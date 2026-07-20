from .inject_qkv_module import *
from diffusers.models.attention_processor import (
    FluxSingleAttnProcessor2_0
)


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxSingleTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
            layer.forward = FluxSingleTransformerBlockForward.__get__(layer, FluxSingleTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model

def init_pipeline_inject(pipeline):
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxSingleAttnProcessor2_0.__call__ = flux_single_attn_call2_0
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)
    if hasattr(pipeline, "transformer"):
        for idx, block in enumerate(pipeline.transformer.single_transformer_blocks):
            block.block_idx = idx
    return pipeline



