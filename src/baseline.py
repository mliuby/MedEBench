import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import sys
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    DiffusionPipeline,
    DDIMScheduler
)
from huggingface_hub import login



# === Configuration ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
METADATA_PATH = os.path.join(BASE_DIR, "editing", "editing_metadata.json")
OUTPUT_ROOT = os.path.join(BASE_DIR, "generated_images")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
CACHE_DIR = os.path.join(BASE_DIR, "cache")

HF_TOKEN = "Your huggingface token"
DIFFUSION_STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.join(BASE_DIR, "InstructDiffusion"))
from edit_cli_my import generate_with_instruct_diffusion
# === PIPELINE LOADERS ===
def load_ip2p_pipeline(model_id):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe.to(DEVICE)


def load_imagic_pipeline(model_id):
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        safety_checker=None,
        local_files_only=False,
        custom_pipeline="imagic_stable_diffusion",
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        ),
    )
    return pipe.to(DEVICE)


# === GENERATION ===
def generate_with_ip2p(pipe, sample, output_dir, seeds=3, igs_values=[1.6], gs_values=[7.5], constant_seed=False):
    os.makedirs(output_dir, exist_ok=True)

    img_id = sample["id"]
    prompt = sample["prompt"]
    img_path = os.path.normpath(os.path.join(BASE_DIR, sample["previous_image"]))

    if not os.path.exists(img_path):
        print(f"Image not found for ID {img_id}: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    seed_list = [0] * seeds if constant_seed else [random.randint(0, 9999) for _ in range(seeds)]

    idx = 0
    for seed in seed_list:
        for igs in igs_values:
            for gs in gs_values:
                generator = torch.manual_seed(seed)
                result = pipe(
                    prompt,
                    image=image,
                    num_inference_steps=DIFFUSION_STEPS,
                    image_guidance_scale=igs,
                    guidance_scale=gs,
                    generator=generator
                )
                result.images[0].save(os.path.join(output_dir, f"{img_id}_{idx}.png"))
                idx += 1




def generate_with_imagic(pipe, sample, output_dir, alphas=[1.3], guidance_scales=[7.5], seeds=3, constant_seed=False):
    os.makedirs(output_dir, exist_ok=True)

    img_id = sample["id"]
    prompt = sample["prompt"]
    img_path = os.path.normpath(os.path.join(BASE_DIR, sample["previous_image"]))

    if not os.path.exists(img_path):
        print(f"Image not found for ID {img_id}: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    seed_list = [0] * seeds if constant_seed else [random.randint(0, 9999) for _ in range(seeds)]

    idx = 0


    # Train the pipeline with the input image and prompt
    generator = torch.Generator(device=DEVICE)
    pipe.train(prompt=prompt, image=image, generator=generator)
    for seed in seed_list:    
        for alpha in alphas:
            for gs in guidance_scales:
                # Generate the edited image
                image_edit = pipe(
                    num_inference_steps=DIFFUSION_STEPS,
                    alpha=alpha,
                    guidance_scale=gs,
                    generator=generator
                ).images[0]
                image_edit.save(os.path.join(output_dir, f"{img_id}_{idx}.png"))
                idx += 1



# === PROCESSING ===
def process_model(model_id, model_name, mode="ip2p", sample_slice=None, **kwargs):
    print(f"Loading model: {model_name} ({mode})")

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        samples = json.load(f)["samples"]

    if sample_slice:
        if isinstance(sample_slice, list):
            samples = [samples[i] for i in sample_slice]
        else:
            samples = samples[sample_slice]

    output_dir = os.path.join(OUTPUT_ROOT, model_name)
    print(f"Output directory: {output_dir}")

    if mode == "ip2p":
        pipe = load_ip2p_pipeline(model_id)
        for sample in tqdm(samples, desc=f"Editing with {model_name}"):
            generate_with_ip2p(pipe, sample, output_dir, **kwargs)

    elif mode == "imagic":
        for sample in tqdm(samples, desc=f"Editing with {model_name}"):
            pipe = load_imagic_pipeline(model_id)
            generate_with_imagic(pipe, sample, output_dir, **kwargs)
            del pipe
            torch.cuda.empty_cache()
    elif mode == "instructdiff":
        generate_with_instruct_diffusion(
            samples=samples,
            output_dir=output_dir,
            # In the provided code snippet, the `config_path` parameter is used in the `process_model`
            # function when the `mode` is set to "instructdiff". In this context, `config_path` is
            # used as a reference to the configuration file path for the InstructDiffusion model.
            config_path=model_id,  # model_id is used as config path here
            ckpt_path="checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt",
            vae_ckpt=None,
            cfg_text_list=kwargs.get("cfg_text_list", [5.0]),
            cfg_image_list=kwargs.get("cfg_image_list", [1.25]),
            seeds=kwargs.get("seeds", 3),
            constant_seed=kwargs.get("constant_seed", False),
            steps=DIFFUSION_STEPS
        )
    print(f"All images saved for model: {model_name}")


# === MAIN ===
def main():
    login(HF_TOKEN)

    # InstructPix2Pix
    process_model(
        model_id="timbrooks/instruct-pix2pix",
        model_name="instruct-pix2pix",
        mode="ip2p",
        # sample_slice=slice(None, 5),
        seeds=2,
        igs_values=[1.6],
        gs_values=[7.5],
    )
    torch.cuda.empty_cache()

    # Paint-by-Inpaint
    process_model(
        model_id="paint-by-inpaint/general-base",
        model_name="paint-by-inpaint",
        mode="ip2p",
        # sample_slice=slice(None, 5),
        seeds=2,
        igs_values=[1.7],
        gs_values=[7],
    )
    torch.cuda.empty_cache()

process_model(
    model_id="CompVis/stable-diffusion-v1-4",
    model_name="imagic-sd-v1-4",
    mode="imagic",
    sample_slice=slice(1079, None),
    alphas=[1.3],
    guidance_scales=[7.5],
    seeds=1,
    constant_seed=False  # or True for deterministic output
)


process_model(
    model_id=os.path.join(BASE_DIR, "InstructDiffusion", "configs/instruct_diffusion.yaml"),   # used as config_path
    model_name="instruct-diffusion",
    mode="instructdiff",
    sample_slice=slice(None, 5),  # or any slice you want
    cfg_text_list=[5.0],
    cfg_image_list=[1.25],
    seeds=3,
    constant_seed=False
)



if __name__ == "__main__":
    main()
    
# python src/baseline.py