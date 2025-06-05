from PIL import Image
import os
import shutil

def convert_and_move_images(input_directory, output_directory):
    """
    Converts images in input_directory to PNG format and moves them to output_directory.
    Supports common formats like .jpg, .jpeg, .bmp, .gif, .tiff, .webp.
    """
    # Step 1: Convert all supported image formats to .png and save to output
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".png", ".webp")):
            input_path = os.path.join(input_directory, filename)
            try:
                with Image.open(input_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    output_filename = os.path.splitext(filename)[0] + ".png"
                    output_path = os.path.join(output_directory, output_filename)
                    img.save(output_path, "PNG")
                print(f"Converted {filename} to PNG")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Step 2: Move all .png files to the output directory (if any were already .png)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            try:
                shutil.move(input_path, output_path)
                print(f"Moved {filename} to {output_directory}")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

    print("All applicable images have been converted and/or moved.")

# === Example usage ===
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    input_dir = os.path.join(base_dir, "editing/working_raw")
    output_dir = os.path.join(base_dir, "editing/raw")
    
    convert_and_move_images(input_dir, output_dir)
