import os
import json
import base64
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# === Configuration ===
API_KEY = "Your OpenAI API"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
META_PATH = os.path.join(BASE_DIR, "editing/editing_metadata.json")
META_PATH_dest = os.path.join(BASE_DIR, "editing/editing_metadata.json")
MAX_WORKERS = 8
MAX_RETRIES_GEN = 5

client = OpenAI(api_key=API_KEY)

# === Utilities ===
def encode_image(relative_path):
    image_path = os.path.join(BASE_DIR, relative_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def is_invalid_response(text):
    return "sorry" in text.lower() or "error" in text.lower() or len(text)<5

# === Prompt Generation ===
def generate_detailed_prompt(sample):
    try:
        prev_image_b64 = encode_image(sample["previous_image"])
        changed_image_b64 = encode_image(sample["changed_image"])
        instruction = sample["prompt"]
        system_instruction = (
            "You are a medical image editing assistant. Your task is to analyze the original (left) image and the edited (right) image, "
            "along with a brief user instruction. Based on these inputs, generate a detailed, precise, and clinically relevant description "
            "of the visual change that was applied.\n\n"
            "Your response must clearly state:\n"
            "- The image modality\n"
            "- The action performed and its target entity\n"
            "- The expected appearance or anatomical result after the change\n\n"
            "Format your output as a single formal instruction sentence describing how the right image was derived from the left. "
            "Do not include explanations, formatting, or extra text."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_instruction}\n\nInstruction: {instruction}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_image_b64}", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{changed_image_b64}", "detail": "high"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"ERROR: {str(e)}"

def regenerate_until_valid(sample, max_retries=MAX_RETRIES_GEN):
    for attempt in range(max_retries):
        result = generate_detailed_prompt(sample)
        if not is_invalid_response(result):
            return result
    return result  # last attempt

# === Main Process ===
def process_all_samples(test_mode=False, sample_limit=20):
    # Load metadata
    with open(META_PATH, "r") as f:
        data = json.load(f)
    samples = data["samples"]

    if test_mode:
        samples = samples[:sample_limit]

    print("🔄 Generating detailed prompts with GPT-4o...")

    # Process with multithreading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(regenerate_until_valid, sample): sample
            for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            sample = futures[future]
            sample["detailed_prompt"] = future.result()

    with open(META_PATH_dest, "w") as f:
        json.dump({"samples": samples}, f, indent=4)

    print("✅ Detailed prompts saved to metadata.")

# === Run Script ===
if __name__ == "__main__":
    process_all_samples(test_mode=False)  # set to True for debugging
    
# src/dataset_construction/augment_prompt.py
