import os
import json
import time
from tqdm import tqdm
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def setup_directories(base_dir, model_name):
    edited_image_dir = os.path.join(base_dir, model_name)
    os.makedirs(edited_image_dir, exist_ok=True)
    return edited_image_dir

def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)

def initialize_client(api_key):
    return genai.Client(api_key=api_key)

def process_image(client, model_id, prompt, prev_image_path, save_path, max_retries=3, retry_delay=5):
    if not os.path.exists(prev_image_path):
        print(f"⚠️ Image {prev_image_path} not found. Skipping.")
        return False

    prev_image = Image.open(prev_image_path).convert("RGB")
    text_input = f"You are good at image editing. Here is the image editing instruction: {prompt}"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[text_input, prev_image],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                ),
            )

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(save_path)
                    print(f"Image saved: {save_path}")
                    return True

            print("No inline data received. Retrying...")

        except Exception as e:
            print(f"Error: {e}")
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit. Waiting {retry_delay}s...")
            time.sleep(retry_delay)

    print(f"Failed to process after {max_retries} attempts.")
    return False

def edit_images(meta_path, output_dir, model_name, api_key, model_id, start_index=999, n_seeds=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    edited_image_dir = setup_directories(output_dir, model_name)
    metadata = load_metadata(meta_path)
    client = initialize_client(api_key)

    for sample in tqdm(metadata['samples'][start_index:], desc="Processing Images", unit="image"):
        img_id = sample['id']
        prompt = sample['prompt']

        for seed in range(n_seeds):
            save_path = os.path.join(edited_image_dir, f"{img_id}_{seed}.png")
            if os.path.exists(save_path):
                continue
            prev_img_path = os.path.join(BASE_DIR, os.path.normpath(sample['previous_image']))
            save_path = os.path.join(edited_image_dir, f"{img_id}_{seed}.png")
            success = process_image(client, model_id, prompt, prev_img_path, save_path)

            if not success:
                print("Fail:", img_id)
                # # Save fallback image
                # if os.path.exists(prev_img_path):
                #     Image.open(prev_img_path).convert("RGB").save(save_path)

    print(f"All images saved to: {edited_image_dir}")
    return edited_image_dir


# Example usage for your environment
if __name__ == "__main__":
    meta_path = os.path.join(BASE_DIR, "editing/editing_metadata_rephrase.json")
    model_name = "gemini_2_flash"
    output_dir = os.path.join(BASE_DIR, "generated_images")
    
    # api_key = "AIzaSyDBfIFXa0SirJybmXwdIuWHfPvsTzRTneU"
    # api_key = "AIzaSyCRxlg6E8GWqLRYDm4dvz7MBkdlKqq21T4"
    api_key = "AIzaSyARTx0tTnSnU-t7kHWOzpzXQdxB_oW0_Eg"

    model_id = "gemini-2.0-flash-exp"

    edited_dir = edit_images(meta_path, output_dir, model_name, api_key, model_id)

# python src/gemini_edit.py 