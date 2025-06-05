import os
import sys
import json
import argparse

# Ensure CUDA_VISIBLE_DEVICES is set early
sys.path.append(".")  # Add project root if needed

import torch
from tqdm import tqdm
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import login
import pyrootutils

# --------- Argument Parser ---------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    return parser.parse_args()

# --------- Constants ---------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_images", "imagic-sd-v1-4")
MODEL_ID = "CompVis/stable-diffusion-v1-4"
HF_TOKEN = "Yout hugging face token"
DEVICE = "cuda"
GUIDANCE_SCALES = [7.5]
ALPHAS = [1.3]
NUM_INFERENCE_STEPS = 50

# --------- Setup Environment ---------
def setup_environment(do_login=True):
    if do_login:
        from huggingface_hub import login
        login(HF_TOKEN)
    pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
    print(f"[Setup] Environment initialized.")

# --------- Load IMAGIC Pipeline ---------
def load_pipeline():
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False,
    )
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        safety_checker=None,
        local_files_only=True,
        custom_pipeline="imagic_stable_diffusion",
        scheduler=scheduler,
    )
    pipe.to(DEVICE)
    return pipe, generator

# --------- Process One Sample ---------
def process_sample(sample, pipe, generator):
    img_id = sample['id']
    prompt = sample['prompt']
    previous_image_path = os.path.join(BASE_DIR, "editing", "previous_resize", os.path.basename(sample['previous_image']))

    if not os.path.exists(previous_image_path):
        print(f"[Warning] Image {previous_image_path} not found. Skipping.")
        return

    # Load original image
    image = Image.open(previous_image_path).convert("RGB")

    # Train the pipeline
    _ = pipe.train(prompt, image=image, generator=generator)

    # Generate edited images
    count=0
    for alpha in ALPHAS:
        for guidance in GUIDANCE_SCALES:
            generated_image = pipe(
                num_inference_steps=NUM_INFERENCE_STEPS,
                alpha=alpha,
                guidance_scale=guidance,
            ).images[0]

            # Crop according to the original image
            previous_image_path = os.path.join(BASE_DIR, sample['previous_image'])
            prev_image_full = Image.open(previous_image_path).convert('RGB')
            crop_width, crop_height = prev_image_full.size
            cropped_image = generated_image.crop((0, 0, crop_width, crop_height))

            save_path = os.path.join(OUTPUT_DIR, f"{img_id}_{count}.png")
            cropped_image.save(save_path)
            count += 1

# --------- Main ---------
def main():
    args = parse_args()
    do_login = os.environ.get("DO_HF_LOGIN", "true").lower() == "true"
    setup_environment(do_login=do_login)

    # Disable diffusers and transformers spam
    import transformers
    import diffusers
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    device = torch.device(DEVICE)

    with open(args.metadata_path, "r") as f:
        all_samples = json.load(f)["samples"]

    samples = all_samples[args.start_idx:args.end_idx]

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id, leave=True, ncols=100):
        pipe, generator = load_pipeline()
        process_sample(sample, pipe, generator)
        del pipe, generator
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
