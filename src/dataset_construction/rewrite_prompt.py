import json
import re
import base64
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random

# Get the root directory of the project (assuming this script is in src/dataset_construction)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Goes up two levels
META_PATH = os.path.join(BASE_DIR, "editing/editing_metadata_detail.json")
New_META_PATH = os.path.join(BASE_DIR, "editing/editing_metadata_rephrase.json")

# === Configuration ===
# API_KEY = os.getenv("OPENAI_API_KEY") # set OPENAI_API_KEY="sk-..."
API_KEY = "Your OpenAI API"

MAX_WORKERS = 8
MAX_RETRIES_GEN = 10
MAX_RETRIES_PARSE = 3

client = OpenAI(api_key=API_KEY)

def is_invalid_response(text):
    return "sorry" in text.lower() or "error" in text.lower() or len(text)<3


def generate_multiple_rewrites(sample, num_rewrites=3):
    instruction = sample["prompt"]

    system_prompt = (
        "You are tasked with rephrasing prompts. Given an instruction, rewrite it in a clear and natural way. "
        "Keep the meaning the same but change the wording."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Instruction: {instruction}\n\nPlease rephrase it {num_rewrites} different ways."}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=30,
            temperature=0.7
        )
        # Expecting GPT-4o to list the rewrites clearly
        content = response.choices[0].message.content.strip()

        # Simple parsing: Assume each rephrase is separated by a new line or numbered.
        rewrites = [line.strip("0123456789. ") for line in content.split("\n") if line.strip()]
        if len(rewrites) >= num_rewrites:
            selected_rewrite = random.choice(rewrites[:num_rewrites])
            sample["rephrased_prompt"] = selected_rewrite
        else:
            sample["rephrased_prompt"] = instruction  # fallback to original if something weird happens

    except Exception as e:
        print(f"Error rephrasing prompt: {e}")
        sample["rephrased_prompt"] = instruction  # fallback

    return sample

def regenerate_until_valid(sample, max_retries=MAX_RETRIES_GEN):
    for attempt in range(max_retries):
        result = generate_multiple_rewrites(sample)
        if not is_invalid_response(result):
            return result
    return result  # last attempt

# === Update the process ===
def process_all_with_rephrasing(test_mode=False, sample_limit=5):
    with open(META_PATH, "r") as f:
        data = json.load(f)
    samples = data["samples"]

    if test_mode:
        samples = samples[:sample_limit]

    # Step: Generate rephrased prompts
    print("Generating rephrased prompts...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(regenerate_until_valid, s): s for s in samples}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Rephrasing Prompts"):
            pass

    # Save updated samples
    with open(New_META_PATH, "w") as f:
        json.dump({"samples": samples}, f, indent=2)
    print("All rephrased prompts generated and saved.")


if __name__ == "__main__":
    process_all_with_rephrasing(test_mode=False)  # Set test_mode=True if you want a fast test

# python src\dataset_construction\rewrite_prompt.py


