import tkinter as tk
from PIL import Image, ImageTk
import os

class ImageCropperApp:
    def __init__(self, root, input_path, outputs_path1, outputs_path2, prompt_path):
        self.root = root
        self.input_path = input_path  # Predefined input file path
        self.output_path1 = os.path.join(outputs_path1, os.path.basename(self.input_path))
        self.output_path2 = os.path.join(outputs_path2, os.path.basename(self.input_path))
        self.prompt_path = prompt_path  # Path to the prompt.txt file
        self.image = None
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()

        # Label to show the line from the prompt.txt
        self.filename_label = tk.Label(root, text="", font=("Arial", 14))
        self.filename_label.pack(side="top", pady=10)

        # Initialize the variables
        self.first_click = None
        self.second_click = None
        self.cropped_image = None
        self.save_counter = 0
        self.auto_mode = False  # Toggle for automatic cropping mode

        # Toggle button for automatic cropping mode
        self.auto_mode_button = tk.Button(root, text="Toggle Auto Mode", command=self.toggle_auto_mode)
        self.auto_mode_button.pack(side="left", padx=10)

        # Label to display current mode (Auto Mode: ON/OFF)
        self.mode_label = tk.Label(root, text="Auto Mode: OFF", font=("Arial", 12))
        self.mode_label.pack(side="left", padx=10)

        self.load_image()  # Automatically load the first image from input_path

    def load_image(self):
        # Load image from the predefined input path
        try:
            self.image = Image.open(self.input_path)
            self.image.thumbnail((600, 600))
            self.show_image()
            self.update_filename_label()  # Update the filename label with the line from the prompt file
            self.canvas.bind("<Button-1>", self.on_click)
        except FileNotFoundError:
            print(f"Error: The file {self.input_path} was not found.")
            self.root.quit()

    def show_image(self):
        # Convert the image to a format Tkinter can display
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def update_filename_label(self):
        # Extract the number from the filename (before the extension)
        filename = os.path.basename(self.input_path)
        number_part = filename.split('.')[0]  # Remove the extension, e.g., '89.png' -> '89'

        # Try to find the line in the prompt.txt file that starts with this number
        line_from_prompt = self.get_line_from_prompt(number_part)

        # Update the label with the found line
        self.filename_label.config(text=line_from_prompt)

    def get_line_from_prompt(self, number_part):
        # Read the prompt.txt file and find the line starting with the number_part
        try:
            with open(self.prompt_path, "r") as file:
                for line in file:
                    if line.startswith(number_part):  # Match the line starting with the number
                        return line.strip()  # Remove any extra newlines or spaces
        except FileNotFoundError:
            print(f"Error: The prompt file {self.prompt_path} was not found.")
            return "Prompt file not found."

        return f"No line starting with {number_part} found."

    def on_click(self, event):
        # Handle the first and second click
        if not self.first_click:
            self.first_click = (event.x, event.y)
        elif not self.second_click:
            self.second_click = (event.x, event.y)
            self.crop_image()

    def crop_image(self):
        if self.auto_mode:
            # Automatic cropping mode: Use predefined coordinates
            # if self.save_counter == 0:
            #     # First save: Crop from (3, 3) to (299, 260)
            #     self.first_click = (2, 26)
            #     self.second_click = (229, 310)
            # elif self.save_counter == 1:
            #     # Second save: Crop from (300, 3) to (599, 260)
            #     self.first_click = (235, 26)
            #     self.second_click = (462, 310)
            #             Automatic cropping mode: Use predefined coordinates
            # if self.save_counter == 0:
            #     # First save: Crop from (3, 3) to (299, 260)
            #     self.first_click = (3, 4)
            #     self.second_click = (295, 250)
            # elif self.save_counter == 1:
            #     # Second save: Crop from (300, 3) to (599, 260)
            #     self.first_click = (305, 4)
            #     self.second_click = (597, 250)

            if self.save_counter == 0:
                # First save: Crop from (3, 3) to (299, 260)
                self.first_click = (2, 2)
                self.second_click = (399,503)
            elif self.save_counter == 1:
                # Second save: Crop from (300, 3) to (599, 260)
                self.first_click = (207, 2)
                self.second_click = (408, 167)

        # Get the bounding box from the two clicks
        print(self.first_click)
        print(self.second_click)
        x1, y1 = self.first_click
        x2, y2 = self.second_click
        # Crop the image based on the two click points
        self.cropped_image = self.image.crop((x1, y1, x2, y2))
        # Draw a rectangle on the canvas to visualize the cropping area
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

        if self.cropped_image:
            # Save the cropped image to different output paths based on the save_counter
            if self.save_counter == 0:
                self.cropped_image.save(self.output_path1)
                print(f"Image saved to {self.output_path1}")
                self.save_counter += 1
                self.first_click = None
                self.second_click = None
                self.canvas.bind("<Button-1>", self.on_click)
            elif self.save_counter == 1:
                self.cropped_image.save(self.output_path2)
                print(f"Image saved to {self.output_path2}")
                # Update input and output paths for the next image
                self.reset_interface()
                file_name, ext = os.path.splitext(os.path.basename(self.input_path))
                new_file_name = str(int(file_name) + 1)
                self.input_path = os.path.join(os.path.dirname(self.input_path), f"{new_file_name}{ext}")            
                self.output_path1 = os.path.join(os.path.dirname(self.output_path1), f"{new_file_name}{ext}")
                self.output_path2 = os.path.join(os.path.dirname(self.output_path2), f"{new_file_name}{ext}")
                self.load_image()  # Automatically load the next image after saving

    def toggle_auto_mode(self):
        # Toggle automatic cropping mode
        self.auto_mode = not self.auto_mode
        mode_status = "ON" if self.auto_mode else "OFF"
        # Update the mode label text
        self.mode_label.config(text=f"Auto Mode: {mode_status}")
        print(f"Auto mode is now {mode_status}")

    def reset_interface(self):
        # Reset everything for another round of cropping
        self.canvas.delete("all")
        self.first_click = None
        self.second_click = None
        self.cropped_image = None
        self.save_counter = 0

if __name__ == "__main__":
    # Automatically set paths relative to project root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # MedGen/

    input_file_path = os.path.join(BASE_DIR, "editing/raw/10.png")
    output_files_path1 = os.path.join(BASE_DIR, "editing/previous")
    output_files_path2 = os.path.join(BASE_DIR, "editing/changed")
    prompt_file_path = os.path.join(BASE_DIR, "editing/prompt.txt")

    root = tk.Tk()
    app = ImageCropperApp(root, input_file_path, output_files_path1, output_files_path2, prompt_file_path)
    root.mainloop()

# python src\dataset_construction\manual_crop.py 