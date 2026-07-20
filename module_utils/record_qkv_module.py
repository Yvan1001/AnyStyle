import math
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)



# FluxAttnProcessor2_0
@torch.no_grad()
def flux_attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    context_input_ndim = encoder_hidden_states.ndim
    if context_input_ndim == 4:
        batch_size, channel, height, width = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size = encoder_hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)

    if attn.norm_added_q is not None:
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
    if attn.norm_added_k is not None:
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

    # attention
    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        # YiYi to-do: update uising apply_rotary_emb
        # from ..embeddings import apply_rotary_emb
        # query = apply_rotary_emb(query, image_rotary_emb)
        # key = apply_rotary_emb(key, image_rotary_emb)
        query, key = apply_rope(query, key, image_rotary_emb)

    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, : encoder_hidden_states.shape[1]],
        hidden_states[:, encoder_hidden_states.shape[1] :],
    )

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if context_input_ndim == 4:
        encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states, encoder_hidden_states


# FluxSingleAttnProcessor2_0
@torch.no_grad()
def flux_single_attn_call2_0(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if hasattr(attn, "processor"):
            block = getattr(attn.processor, "block", None)
        else:
            block = None

        recorder = getattr(block, "qkv_recorder", None)

        if recorder is not None:
        # if recorder is None:
            block_idx = block.block_idx
            if recorder.should_record(step_index, block_idx):
                recorder.save(
                    step_index=step_index,
                    block_idx=block_idx,
                    q=query,
                    k=key,
                    v=value
                )


        # Apply RoPE if needed
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)



        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states

@torch.no_grad()    
def FluxSingleTransformerBlockForward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        timestep=None,
        step_index=None
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        self.attn.processor.timestep = timestep
        self.attn.processor.block = self

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            step_index=step_index
        )

        
        
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states




@torch.no_grad()
def FluxTransformer2DModelForward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        step_index: int = None
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    timestep=timestep,
                    step_index=step_index
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)