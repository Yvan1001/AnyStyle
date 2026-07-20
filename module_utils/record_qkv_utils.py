from .record_qkv_module import *
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
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

def init_pipeline_record(pipeline,recorder):
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            # FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxSingleAttnProcessor2_0.__call__ = flux_single_attn_call2_0
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)
    if hasattr(pipeline, "transformer"):
        for idx, block in enumerate(pipeline.transformer.single_transformer_blocks):
            block.block_idx = idx
            block.qkv_recorder = recorder
    return pipeline
