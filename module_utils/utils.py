import torch
from typing import Optional, Union
import math
from torch import Tensor



def load_lora_weight_denoise(pipe,style_image_lora_path):
    pipe.load_lora_weights(style_image_lora_path, weight_name="pytorch_lora_weights.safetensors",adapter_name="style_adapter")
    pipe.set_adapters(adapter_names=["style_adapter"],adapter_weights=[1])



def encode_imgs(imgs, pipeline, DTYPE):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=DTYPE)
    return latents

def decode_imgs(latents, pipeline):
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs


def scale_noise(
        scheduler,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def time_shift(mu: float, sigma: float, t: Tensor):

        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    

    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32)

    if shift:

        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

