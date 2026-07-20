import os
os.environ["WANDB_MODE"] = "offline"
import subprocess
from pathlib import Path
import torch
import gc


PRETRAINED_MODEL = "black-forest-labs/FLUX.1-dev"
GPU_ID = "0"                                           
ACCELERATE_BIN = str(Path(os.environ["CONDA_PREFIX"]) / "bin" / "accelerate")

def run_command(cmd, desc):
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        return False
    return True


STYLE_OUTPUT_DIR=f"./AnyStyle/outputs"
STYLE_INSTANCE_DIR= "./AnyStyle/asserts/cat"
STYLE_PROMPT= "A cat in watercolor painting style"
STYLE_PROMPT = "in watercolor painting style"


train_cmd = [
    ACCELERATE_BIN, "launch",
    "--num_processes=1",
    "--gpu_ids", GPU_ID,
    "train_dreambooth_lora_flux_style.py",
    "--pretrained_model_name_or_path",PRETRAINED_MODEL,
    "--instance_data_dir", STYLE_INSTANCE_DIR,
    "--instance_prompt",STYLE_PROMPT, 
    "--output_dir", STYLE_OUTPUT_DIR,
    "--mixed_precision","bf16" ,
    "--weighting_scheme=none" ,
    "--resolution=512" ,
    "--train_batch_size=1" ,
    "--repeats=1" ,
    "--learning_rate=5e-4" ,
    "--guidance_scale=1" ,
    "--report_to=wandb" ,
    "--gradient_accumulation_steps=1" ,
    "--gradient_checkpointing" ,
    "--lr_scheduler=constant" ,
    "--lr_warmup_steps=0" ,
    "--rank=64" ,
    "--max_train_steps=200" ,
    "--checkpointing_steps=50" ,
    "--seed=0" ,
    "--do_fp8_training" ,
]

if not run_command(train_cmd, f"fine-tuning---Style"):
    print(f"failed,exit")
    

gc.collect()                    
torch.cuda.empty_cache()
