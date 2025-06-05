import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
# import ImageReward as RM
from einops import rearrange
import clip
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

# === CLIP SIMILARITY CLASS ===
class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        self.size = {
            "RN50x4": 288, "RN50x16": 384, "RN50x64": 448,
            "ViT-L/14@336px": 336
        }.get(name, 224)

        self.model, _ = clip.load(name, device="cuda", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list[str]) -> torch.Tensor:
        text_tokens = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=1, keepdim=True)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        return image_features / image_features.norm(dim=1, keepdim=True)
    


class MedCLIPSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = MedCLIPProcessor()
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()
        self.model.eval().cuda()
    
    def encode_text(self, text: list[str]) -> torch.Tensor:
        # Dummy image to match processor interface
        dummy = Image.new("RGB", (224, 224))
        inputs = self.processor(text=text, images=dummy, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**{k: v.cuda() for k, v in inputs.items()})
        text_embeds = outputs["text_embeds"]
        return F.normalize(text_embeds, dim=1)

    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(text=["None"], images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**{k: v.cuda() for k, v in inputs.items()})        
        img_embeds = outputs["img_embeds"]
        return F.normalize(img_embeds, dim=1)


# === Metric Functions ===

def compute_ssim(ref_np, edited_np):
    return ssim(ref_np, edited_np, channel_axis=-1)

def compute_psnr(ref_np, edited_np):
    return psnr(ref_np, edited_np)

def compute_lpips(edited_img, ref_img, lpips_model):
    return lpips_model(edited_img, ref_img).item()

def compute_image_reward(prompt, edited_path, reward_model):
    with torch.no_grad():
        return reward_model.score(prompt, edited_path)

def compute_clip_scores(clip_model, edited_img, ref_img, prev_img, prompt):
    with torch.no_grad():
        text_embed = clip_model.encode_text([prompt])
        edited_embed = clip_model.encode_image(edited_img)
        ref_embed = clip_model.encode_image(ref_img)
        prev_embed = clip_model.encode_image(prev_img)

        clip_prompt_score = F.cosine_similarity(edited_embed, text_embed).item()
        clip_image_image_score = F.cosine_similarity(edited_embed, ref_embed).item()
        editclip_score = F.cosine_similarity(edited_embed - prev_embed, text_embed).item()

    return clip_prompt_score, clip_image_image_score, editclip_score

def compute_medclip_scores(medclip_model, edited_img_pil, ref_img_pil, prev_img_pil, prompt):
    with torch.no_grad():
        edited_embed = medclip_model.encode_image(edited_img_pil)
        ref_embed = medclip_model.encode_image(ref_img_pil)
        prev_embed = medclip_model.encode_image(prev_img_pil)
        text_embed = medclip_model.encode_text([prompt])

        clip_prompt_score = F.cosine_similarity(edited_embed, text_embed).item()
        clip_image_image_score = F.cosine_similarity(edited_embed, ref_embed).item()
        editclip_score = F.cosine_similarity(edited_embed - prev_embed, text_embed).item()

    return clip_prompt_score, clip_image_image_score, editclip_score



def image_to_numpy(img_tensor):
    return np.clip(np.array(transforms.ToPILImage()(img_tensor.cpu())), 0, 255).astype(np.uint8)

# === Main Evaluation Function ===

def compute_metrics(image_dir, metadata_path):

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    samples = metadata["samples"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = lpips.LPIPS(net='alex').to(device)
    # reward_model = RM.load("ImageReward-v1.0")
    clip_similarity = ClipSimilarity(name="ViT-L/14").to(device)
    medclip_similarity = MedCLIPSimilarity().to(device)

    results = {}
    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Processing {os.path.basename(image_dir)}"):
            sample_id = sample["id"]
            prompt = sample["prompt"]
            detailed_prompt = sample["detailed_prompt"]
            filename = f"{sample_id}_0.png"
            
            ref_dir = os.path.dirname(metadata_path)
            edited_path = os.path.join(image_dir, filename)
            prev_path = os.path.join(ref_dir, "previous", f"{sample_id}.png")
            ref_path = os.path.join(ref_dir, "changed", f"{sample_id}.png")
            mask_path = os.path.join(ref_dir, "previous_mask", f"{sample_id}.png")

            if not all(os.path.exists(p) for p in [edited_path, ref_path, prev_path, mask_path]):
                print(f"Missing files for {sample_id}. Skipping...")
                continue
            
            # Load previous image to get size
            prev_img_pil = Image.open(prev_path).convert("RGB")
            target_size = prev_img_pil.size  # (width, height)

            # Resize other images to match previous
            edited_img_pil = Image.open(edited_path).convert("RGB").resize(target_size, Image.BICUBIC)
            ref_img_pil = Image.open(ref_path).convert("RGB").resize(target_size, Image.BICUBIC)
            mask_pil = Image.open(mask_path).convert("L").resize(target_size, Image.NEAREST)
            mask_tensor = transforms.ToTensor()(mask_pil).unsqueeze(0).to(device)  
            inv_mask = 1.0 - mask_tensor 



            # Convert to PyTorch tensors manually
            to_tensor = transforms.ToTensor()
            edited_img = to_tensor(edited_img_pil).unsqueeze(0).to(device)
            ref_img = to_tensor(ref_img_pil).unsqueeze(0).to(device)
            prev_img = to_tensor(prev_img_pil).unsqueeze(0).to(device)
            edited_masked = edited_img * inv_mask 
            prev_masked = prev_img * inv_mask

            edited_masked_np = image_to_numpy(edited_masked.squeeze(0))
            prev_masked_np = image_to_numpy(prev_masked.squeeze(0))   
            ref_np = image_to_numpy(ref_img.squeeze(0))
            edited_np = image_to_numpy(edited_img.squeeze(0))
            
            clip_prompt_score, clip_image_image_score, editclip_score = compute_clip_scores(
                clip_similarity, edited_img, ref_img, prev_img, prompt
            )
            clip_prompt_score_2, _ , editclip_score_2 = compute_clip_scores(
                clip_similarity, edited_img, ref_img, prev_img, detailed_prompt
            )
            medclip_prompt_score, medclip_img_score, medclip_editclip = compute_medclip_scores(
                medclip_similarity, edited_img_pil, ref_img_pil, prev_img_pil, prompt
            )
            medclip_prompt_score_2, _ , medclip_editclip_2 = compute_medclip_scores(
                medclip_similarity, edited_img_pil, ref_img_pil, prev_img_pil, detailed_prompt
            )
            ssim_masked_val = compute_ssim(prev_masked_np, edited_masked_np)
            psnr_val = compute_psnr(ref_np, edited_np)
            lpips_val = compute_lpips(edited_img, ref_img, lpips_model)
            # ir_score = compute_image_reward(prompt, edited_path, reward_model)
            # ir_score_2 = compute_image_reward(detailed_prompt, edited_path, reward_model)
            
            results[sample_id] = {
                "clip_prompt_score": clip_prompt_score,
                "clip_prompt_score_2": clip_prompt_score_2,
                "clip_image_image_score": clip_image_image_score,
                "editclip": editclip_score,
                "editclip_2": editclip_score_2,
                "medclip_prompt_score": medclip_prompt_score,
                "medclip_prompt_score_2": medclip_prompt_score_2,
                "medclip_image_image_score": medclip_img_score,
                "medclip_editclip": medclip_editclip,
                "medclip_editclip_2": medclip_editclip_2,
                "psnr": psnr_val,
                "lpips": lpips_val,
                "imagereward": ir_score,
                "imagereward2": ir_score_2,
                "masked_ssim": ssim_masked_val
            }

    return results

# === Evaluate All Models ===

def evaluate_all_models(model_dirs, metadata_path, output_path):
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if the file already exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Evaluate each model
    for model_name, image_dir in model_dirs.items():
        if model_name in all_results:
            print(f"✅ Skipping {model_name} (already evaluated)")
            continue

        print(f"\n🔍 Evaluating model: {model_name}")
        model_result = compute_metrics(image_dir, metadata_path)
        all_results[model_name] = model_result

        # Save combined results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=4)

    print(f"\n✅ All model results saved to {output_path}")

# === Run Example ===

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    model_dirs = {
        "gemini_2_flash": os.path.join(base_dir, "generated_images/gemini_2_flash"),
        "seedx": os.path.join(base_dir, "generated_images/seedx"),
        "imagic-sd-v1-4": os.path.join(base_dir, "generated_images/imagic-sd-v1-4"),
        "instruct-pix2pix": os.path.join(base_dir, "generated_images/instruct-pix2pix"),
        "instruct-diffusion": os.path.join(base_dir, "generated_images/instruct-diffusion"),
        "paint-by-inpaint": os.path.join(base_dir, "generated_images/paint-by-inpaint"),
        "icedit": os.path.join(base_dir, "generated_images/icedit")
    }
    evaluate_all_models(
        model_dirs=model_dirs,
        metadata_path=os.path.join(base_dir, "editing/editing_metadata.json"),
        output_path=os.path.join(base_dir, "evaluation_result/all_models_metrics_detail.json")
    )
# conda activate metric
# python src/eval_metrics/metrics.py