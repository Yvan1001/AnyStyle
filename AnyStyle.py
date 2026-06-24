import os
from diffusers import FluxPipeline
import argparse
import random
import numpy as np
import gc
from AnyStyle_utils import *
from torchvision import transforms
from PIL import Image


def main(args):
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = args.output_path

    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    pipe = pipe.to(device)
    load_lora_weight_denoise(pipe,"./outputs")


    torch.cuda.empty_cache()
    DTYPE = torch.bfloat16
    T_steps = 28
    tar_guidance_scale = 5
    seed = 14
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tar_prompt = f"A rocket is being launched in watercolor painting style"


    default_transform = transforms.Compose(
        [
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ]
    )

    content_image_float32 = Image.open("asserts/rocket/rocket.jpg").convert('RGB')
    content_image_float32 = default_transform(content_image_float32)


    os.makedirs(output_path, exist_ok=True)

    content_image_float32 = content_image_float32.unsqueeze(0).to(device).to(DTYPE)
    content_img_latent = encode_imgs(content_image_float32, pipe, DTYPE)
    out_image = Lora_style(pipe,
                            content_img_latent,
            
                            tar_prompt,
                            DTYPE,
                            T_steps,
                            tar_guidance_scale,

                            

    )


    out_image.save("result.jpg")


    torch.cuda.empty_cache()
    gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AnyStyle')
    parser.add_argument('--model_path', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--height', type=int, default=512,
                        help='image height')
    parser.add_argument('--width', type=int, default=512,
                        help='image width')
    args = parser.parse_args()
    main(args)