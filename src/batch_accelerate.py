import os
import json
import subprocess
import argparse
from huggingface_hub import login

login("Your huggingface token")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, default=os.path.join(BASE_DIR, "editing/editing_metadata.json"))
    parser.add_argument("--worker_script", type=str, default="src/imagic_edit.py")
    args = parser.parse_args()

    # Predefined GPU assignment dictionary
    assignments = {
        0: [879, 884],
        2: [294, 297],
        3: [884, 889]
    }


    with open(args.metadata_path, "r") as f:
        all_samples = json.load(f)["samples"]

    processes = []

    for gpu_id, (start_idx, end_idx) in assignments.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["DO_HF_LOGIN"] = "false"

        command = [
            "python", args.worker_script,
            "--metadata_path", args.metadata_path,
            "--start_idx", str(start_idx),
            "--end_idx", str(end_idx),
        ]
        print(f"Launching process on GPU {gpu_id}: {' '.join(command)}")

        p = subprocess.Popen(command, env=env)
        processes.append(p)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()

# src/batch_accelerate.py