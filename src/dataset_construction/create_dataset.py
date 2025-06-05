import json
import os

# Get the root directory of the project (assuming this script is in src/dataset_construction)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Goes up two levels

def generate_editing_metadata(
    changed_dir="editing/changed",
    previous_dir="editing/previous",
    prompt_file="editing/prompt.txt",
    output_json="editing/editing_metadata.json"
):
    # Read prompts from prompt.txt
    with open(os.path.join(BASE_DIR, prompt_file), "r") as f:
        prompts = [line.strip().split(".", 1)[1].strip() for line in f.readlines()]

    # Generate metadata
    samples = []
    for i, prompt in enumerate(prompts, start=1):
        sample = {
            "id": i,
            "changed_image": os.path.join(changed_dir, f"{i}.png"),
            "previous_image": os.path.join(previous_dir, f"{i}.png"),
            "prompt": prompt
        }
        samples.append(sample)

    # Final structure
    metadata = {"samples": samples}

    # Save to JSON
    with open(os.path.join(BASE_DIR, "editing/editing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {output_json}")

if __name__ == "__main__":
    generate_editing_metadata()

# python src\dataset_construction\create_dataset.py