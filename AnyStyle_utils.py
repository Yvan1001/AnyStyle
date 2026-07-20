
import torch
from module_utils.utils import *
from module_utils.inject_qkv_utils import init_pipeline_inject
from module_utils.record_qkv_utils import init_pipeline_record
import torch


GLOBAL_CONTENT_QKV_DICT = {}
GLOBAL_STYLE_QKV_DICT = {}

class CONTENT_QKVRecorder:
    def __init__(
        self,
        record_timesteps: set,
        max_blocks: int = 38,
        enabled: bool = True,
    ):
        self.record_timesteps = record_timesteps
        self.max_blocks = max_blocks
        self.enabled = enabled

        # os.makedirs(save_dir, exist_ok=True)

    def should_record(self, step_index: int, block_idx: int) -> bool:
        if not self.enabled:
            return False
        if block_idx >= self.max_blocks:
            return False
        
        # if block_idx >= self.max_blocks or block_idx <=9:
        #     return False

        return step_index in self.record_timesteps

    @torch.no_grad()
    def save(self, step_index, block_idx, q, k, v):
        """
        Save QKV into GLOBAL_QKV_DICT (CPU memory)
        """

        key = (int(step_index), int(block_idx))

        GLOBAL_CONTENT_QKV_DICT[key] = {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu()
        }


class STYLE_QKVRecorder:
    def __init__(
        self,
        record_timesteps: set,
        max_blocks: int = 38,
        enabled: bool = True,
    ):
        self.record_timesteps = record_timesteps
        self.max_blocks = max_blocks
        self.enabled = enabled

        # os.makedirs(save_dir, exist_ok=True)

    def should_record(self, step_index: int, block_idx: int) -> bool:
        if not self.enabled:
            return False
        if block_idx >= self.max_blocks:
            return False
        
        # if block_idx >= self.max_blocks or block_idx <=9:
        #     return False

        return step_index in self.record_timesteps

    @torch.no_grad()
    def save(self, step_index, block_idx, q, k, v):
        """
        Save QKV into GLOBAL_QKV_DICT (CPU memory)
        """

        key = (int(step_index), int(block_idx))

        GLOBAL_STYLE_QKV_DICT[key] = {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu()
        }




def Lora_style_denoise(
                pipe,
                content_image_latents,
                DTYPE=torch.bfloat16,
                tar_prompt='photo of a tiger',
                T_steps: int = 28,
                guidance_scale: float = 4.0,
                use_shift_t_sampling=True,
):
    

    device = content_image_latents.device
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,

    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    latent_image_ids = pipe._prepare_latent_image_ids(
        content_image_latents.shape[0],
        content_image_latents.shape[2],
        content_image_latents.shape[3],
        content_image_latents.device, 
        DTYPE,
    )

    init_latents = torch.randn_like(content_image_latents)

    packed_init_latents = pipe._pack_latents(
        init_latents,
        batch_size=init_latents.shape[0],
        num_channels_latents=init_latents.shape[1],
        height=init_latents.shape[2],
        width=init_latents.shape[3],
    )

    packed_content_image_latents = pipe._pack_latents(
        content_image_latents,
        batch_size=content_image_latents.shape[0],
        num_channels_latents=content_image_latents.shape[1],
        height=content_image_latents.shape[2],
        width=content_image_latents.shape[3],
    )

    
    timesteps = get_schedule( 
                num_steps=T_steps,
                image_seq_len=(packed_init_latents.shape[1] ), 
                shift=use_shift_t_sampling,
            )
    guidance_vec = torch.full((packed_init_latents.shape[0],), guidance_scale, device=packed_init_latents.device, dtype=packed_init_latents.dtype)

    record_timesteps = set([i for i in range(T_steps)][10:])
    # record_timesteps = set([i for i in range(T_steps)])
    content_recorder = CONTENT_QKVRecorder(
        record_timesteps=set(t for t in record_timesteps),
        max_blocks=38,
        enabled=True,
    )
    pipe = init_pipeline_record(pipe,content_recorder)

    with pipe.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((packed_init_latents.shape[0],), t_curr, dtype=packed_init_latents.dtype, device=packed_init_latents.device)
            key_packed_latents = t_vec * packed_init_latents + (1 - t_vec) * packed_content_image_latents
            noise_pred = pipe.transformer(
                hidden_states=key_packed_latents,
                timestep=t_vec,
                guidance=guidance_vec,
                encoder_hidden_states=tar_prompt_embeds,
                txt_ids=tar_text_ids,
                img_ids=latent_image_ids,
                pooled_projections=tar_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
                step_index=T_steps-i
            )[0]
            progress_bar.update()


    pipe = init_pipeline_inject(pipe)

    packed_latents = packed_init_latents

    with pipe.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)

            noise_pred = pipe.transformer(
                hidden_states=packed_latents,
                timestep=t_vec,
                guidance=guidance_vec,
                encoder_hidden_states=tar_prompt_embeds,
                txt_ids=tar_text_ids,
                img_ids=latent_image_ids,
                pooled_projections=tar_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
                step_index=T_steps-i,
                record_content_qkv_dict = GLOBAL_CONTENT_QKV_DICT,
                record_style_qkv_dict = GLOBAL_STYLE_QKV_DICT

            )[0]

            packed_latents = packed_latents + (t_prev - t_curr) * noise_pred
            packed_latents = packed_latents.to(DTYPE)
            progress_bar.update()



    latents = pipe._unpack_latents(
            packed_latents,
            height=512,
            width=512,
            vae_scale_factor=pipe.vae_scale_factor,
    )
    latents = latents.to(DTYPE)
    return latents



def Lora_style(pipe,
                content_img_latent,
                tar_prompt,
                DTYPE,
                T_steps: int = 28,
                tar_guidance_scale: float = 1.5,
                ):  
    with torch.no_grad():
        output_latents = Lora_style_denoise(
            pipe, 
            content_image_latents=content_img_latent,
            DTYPE=DTYPE,
            tar_prompt=tar_prompt,
            T_steps=T_steps,
            guidance_scale=tar_guidance_scale,
            use_shift_t_sampling=True,

        )
        out_image = decode_imgs(output_latents, pipe)[0]
    return out_image

