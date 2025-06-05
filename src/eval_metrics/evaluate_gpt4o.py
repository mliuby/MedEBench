import os
import json
import base64
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ========== Configuration ==========
base_dir = os.path.abspath("../../")
EDIT_META_PATH = "editing/editing_metadata.json"
EVAL_OUTPUT_PATH = "evaluation_result/gpt4o_detailed_scores.json"
API_KEY = "Your OpenAI API Key"
MAX_WORKERS = 4

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

image_cache = {}

# ========== Image Encoding ==========
def encode_image_base64_cached(image_path):
    if image_path not in image_cache:
        with open(image_path, "rb") as f:
            image_cache[image_path] = base64.b64encode(f.read()).decode("utf-8")
    return image_cache[image_path]

# ========== Prompt Builder ==========
def build_gpt4o_prompt(editing_prompt, prev_img, edited_img, gt_img):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": f"I have an image editing task. Here's the editing prompt:\n\"{editing_prompt}\""}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the input image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_img}"}}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the groundtruth image you can refer to:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_img}"}}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the edited image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_img}"}}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "You are a good medical image analysis expert. Evaluate the edit using the following steps:\n\n"
                "**Step 1: Visual Difference Description**\n"
                "Compare the input and edited images. Describe all visible differences between them, including:\n"
                "- Additions, removals, or modifications of visual elements.\n"
                "- Emphasize the extent of the change and specify which anatomical regions were affected.\n"
                "List the differences clearly and item by item.\n\n"
                "**Step 2: Evaluation (Three Scores)**\n"
                "**1. Editing Accuracy (0-10):**\n"
                "- Score strictly based on alignment with the editing prompt. Deduct points for any inaccuracy or missing elements.\n\n"
                "**2. Contextual Preservation (0-10):**\n"
                "- Were parts of the image outside the editing prompt left untouched?\n"
                "- Were there unintended or unnecessary changes to unrelated anatomy or background?\n\n"
                "**3. Visual Quality (0-10):**\n"
                "- Consider clarity, sharpness, blur, artifacts, naturalness, and consistency.\n\n"
                "Please format your evaluation as follows:\n"
                "- Editing Accuracy: [Score]/10\n"
                "- Contextual Preservation: [Score]/10\n"
                "- Visual Quality: [Score]/10"
            )}
        ]}
    ]

# ========== Response Parser ==========
def parse_response(response_text):
    scores = {"editing_accuracy": -1.0, "contextual_preservation": -1.0, "visual_quality": -1.0}
    for line in response_text.split("\n"):
        try:
            if "Editing Accuracy" in line:
                scores["editing_accuracy"] = float(line.split(":")[1].split("/")[0].strip())
            elif "Contextual Preservation" in line:
                scores["contextual_preservation"] = float(line.split(":")[1].split("/")[0].strip())
            elif "Visual Quality" in line:
                scores["visual_quality"] = float(line.split(":")[1].split("/")[0].strip())
        except Exception:
            continue
    return scores

# ========== GPT-4o Evaluation ==========
def evaluate_sample(prompt, prev_path, edit_path, gt_path):
    prev_b64 = encode_image_base64_cached(prev_path)
    edit_b64 = encode_image_base64_cached(edit_path)
    gt_b64 = encode_image_base64_cached(gt_path)
    messages = build_gpt4o_prompt(prompt, prev_b64, edit_b64, gt_b64)

    for attempt in range(1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=400
            )
            content = response.choices[0].message.content
            scores = parse_response(content)
        except Exception as e:
            print(f"[Attempt {attempt+1}] API error: {e}")
            scores = None

        if scores and all(score != -1.0 for score in scores.values()):
            return scores
        else:
            print(f"[Attempt {attempt+1}] Failed or incomplete scores. Retrying...")
            time.sleep(1)

    return {
        "editing_accuracy": -1,
        "contextual_preservation": -1,
        "visual_quality": -1
    }

# ========== Main Evaluation Process ==========
def evaluate_all_models(model_dirs, metadata_path, output_path, test_mode=False, sample_limit=5, use_multithread=True):
    with open(metadata_path) as f:
        metadata_full = {str(s["id"]): s for s in json.load(f)["samples"]}

    if test_mode:
        metadata = dict(list(metadata_full.items())[:sample_limit])
    else:
        metadata = metadata_full

    # Load previous results if resuming
    if os.path.exists(output_path):
        with open(output_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for model_name, model_dir in model_dirs.items():
        print(f"\nEvaluating: {model_name}")
        model_result = {}

        def process_sample(sid, sample):
            edited_path = os.path.join(model_dir, f"{sid}_0.png")
            if not os.path.exists(edited_path):
                return sid, None
            detailed_prompt_scores = evaluate_sample(
                sample["detailed_prompt"],
                os.path.join(base_dir, sample["previous_image"]),
                edited_path,
                os.path.join(base_dir, sample["changed_image"]),
            )
            return sid, {
                "gpt4o_editing_accuracy_detailed": detailed_prompt_scores["editing_accuracy"],
                "gpt4o_contextual_preservation_detailed": detailed_prompt_scores["contextual_preservation"],
                "gpt4o_visual_quality_detailed": detailed_prompt_scores["visual_quality"]
            }

        existing_model_result = all_results.get(model_name, {})
        sids_to_process = [sid for sid in metadata if str(sid) not in existing_model_result]

        if use_multithread:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_sample, sid, metadata[sid]): sid
                    for sid in sids_to_process
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name}"):
                    sid, result = future.result()
                    if result:
                        model_result[sid] = result
                all_results.setdefault(model_name, {}).update(model_result)
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
        else:
            for sid in tqdm(sids_to_process, desc=f"{model_name}"):
                sid, result = process_sample(sid, metadata[sid])
                if result:
                    model_result[sid] = result
            all_results.setdefault(model_name, {}).update(model_result)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

# ========== Run ==========
if __name__ == "__main__":
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
        metadata_path=os.path.join(base_dir, EDIT_META_PATH),
        output_path=os.path.join(base_dir, EVAL_OUTPUT_PATH),
        test_mode=False,
        sample_limit=5,
        use_multithread=True  # Set to False to disable threading
    )
