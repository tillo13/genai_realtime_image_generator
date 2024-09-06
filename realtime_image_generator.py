import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
import torch
from diffusers import AutoPipelineForImage2Image
import numpy as np
import random
import platform


class ImageGeneratorApp:
    def __init__(self, window, window_title, model_name):
        print("Initializing the Image Generator App...")

        self.window = window
        self.window.title(window_title)

        # Detect the operating system
        self.os = platform.system()
        print(f"Operating System: {self.os}")

        # Set the size of the generated image
        self.image_size = 512  # Assuming the generated image size is 512x512 pixels

        # Set window size to fit the image and controls
        window_width = self.image_size + 40  # Add some padding
        controls_height = 300  # Approximate height needed for controls
        window_height = self.image_size + controls_height + 40  # Add some padding
        self.window.geometry(f'{window_width}x{window_height}')

        # Set up grid layout
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)

        print("Setting up canvas...")
        self.output_canvas = tk.Canvas(window, width=self.image_size, height=self.image_size, bg="white")
        self.output_canvas.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        print("Setting up controls frame...")
        self.controls_frame = ttk.Frame(window)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        print("Setting up UI elements...")
        self.setup_ui()

        self.recording = False
        self.previous_frame = None  # To store the previous frame for img2img feedback
        self.frame_count = 0

        print("Loading model...")
        self.model_name = model_name
        self.load_model()

        print("Setting up window close protocol...")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("Initialization complete.")

        print("Starting image generation automatically...")
        self.toggle_recording()  # Automatically start the image generation process

    def setup_ui(self):
        font_size = ('Helvetica', 10)
        style = ttk.Style()
        style.configure('W.TButton', font=font_size)

        # Default text prompt
        default_prompt = "A photograph of a water drop"

        # Text input prompt
        self.text_input_label = tk.Label(self.controls_frame, text="Prompt:", font=font_size)
        self.text_input_label.pack(pady=5, anchor="w")

        self.text_input = tk.Text(self.controls_frame, width=50, height=3, font=font_size, wrap=tk.WORD)
        self.text_input.insert(tk.END, default_prompt)  # Set default prompt
        self.text_input.pack(pady=5)

        sliders_frame = ttk.Frame(self.controls_frame)
        sliders_frame.pack(pady=5)

        # Adjusted slider ranges and defaults for 2x2 grid
        self.strength_slider = tk.Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Strength", length=200)
        self.strength_slider.set(1.0)  # Set default
        self.strength_slider.grid(row=0, column=0, padx=5, pady=5)

        self.guidance_scale_slider = tk.Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Guidance Scale", length=200)
        self.guidance_scale_slider.set(1.0)  # Set default
        self.guidance_scale_slider.grid(row=0, column=1, padx=5, pady=5)

        self.num_steps_slider = tk.Scale(sliders_frame, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL, label="Num Inference Steps", length=200)
        self.num_steps_slider.set(2)  # Set default
        self.num_steps_slider.grid(row=1, column=0, padx=5, pady=5)

        # New seed slider
        self.seed_slider = tk.Scale(sliders_frame, from_=0, to=10000, resolution=1, orient=tk.HORIZONTAL, label="Seed", length=200)
        self.seed_slider.set(1)  # Set default to 1
        self.seed_slider.grid(row=1, column=1, padx=5, pady=5)

        self.btn_toggle_record = ttk.Button(self.controls_frame, text="Toggle Generation", command=self.toggle_recording, style='W.TButton')
        self.btn_toggle_record.pack(pady=10)

    def load_model(self):
        try:
            print(f"Attempting to load model '{self.model_name}' to GPU...")
            self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_name, torch_dtype=torch.float16, variant="fp16")
            self.pipe.to("cuda")
            self.device = "cuda"
            print(f"Model '{self.model_name}' loaded and moved to GPU (cuda).")
        except Exception as e:
            print(f"Failed to load model '{self.model_name}' to GPU. Falling back to CPU. Error: {e}")
            print(f"Loading model '{self.model_name}' to CPU...")
            self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_name, torch_dtype=torch.float32)
            self.pipe.to("cpu")
            self.device = "cpu"
            print(f"Model '{self.model_name}' loaded and moved to CPU.")

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            print("Generation started...")
            self.generate_images()
        else:
            print("Generation stopped.")

    def generate_images(self):
        if self.recording:
            print("Generating image...")
            self.process_and_display_frame()
            self.window.after(1000, self.generate_images)  # Set interval to 1000 ms (1 second)

    def process_and_display_frame(self):
        prompt = self.text_input.get("1.0", tk.END).strip()
        seed = self.seed_slider.get()

        if prompt:
            torch.manual_seed(seed)
            print(f"Using prompt: {prompt}")
            print(f"Using seed: {seed}")

            # Create an initial image from the prompt if it's the first frame
            if self.previous_frame is None:
                print("Generating initial image...")
                init_image = Image.new('RGB', (self.image_size, self.image_size), color='white')
                transformed_image = self.pipe(prompt=prompt,
                                              image=init_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]
                self.previous_frame = transformed_image
            else:
                print("Generating perturbed and blended image...")
                # Apply random perturbations to the previous frame
                perturbed_image = self.apply_random_perturbations(self.previous_frame)

                # Use the perturbed previous frame for the next iteration
                transformed_image = self.pipe(prompt=prompt,
                                              image=perturbed_image,
                                              strength=self.strength_slider.get(),
                                              guidance_scale=self.guidance_scale_slider.get(),
                                              num_inference_steps=self.num_steps_slider.get()).images[0]

                # Blend the previous and current frames with a reduced effect
                blended_image = self.blend_images(self.previous_frame, transformed_image, alpha=0.1)

                self.previous_frame = blended_image  # Update the previous frame for the next iteration

                print("Displaying blended image...")
                self.display_transformed_image(blended_image)

                # Increment the seed every few frames
                self.frame_count += 1
                if self.frame_count % 20 == 0:  # Change the seed every 20 frames
                    print("Updating seed...")
                    self.seed_slider.set(seed + 1)

    def apply_random_perturbations(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust brightness slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1 + random.uniform(-0.05, 0.05))  # Adjust contrast slightly
        return image

    def blend_images(self, prev_img, curr_img, alpha=0.1):
        prev_array = np.array(prev_img)
        curr_array = np.array(curr_img)
        blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
        return Image.fromarray(blended_array)

    def display_transformed_image(self, transformed_image):
        photo = ImageTk.PhotoImage(transformed_image.resize((self.image_size, self.image_size), Image.LANCZOS))
        self.output_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.output_canvas.image = photo  # Keep a reference!

    def on_closing(self):
        print("Closing application...")
        self.recording = False
        self.window.destroy()


def get_model_name():
    default_model = "stabilityai/sdxl-turbo"
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    def timeout():
        if not submit_called:
            print("Timeout reached, using default model name.")
            model_name_entry.set(default_model)
            submit()

    dialog = tk.Toplevel(root)
    dialog.title("Model Name")

    tk.Label(dialog, text="Enter the name of the Hugging Face model (default: stabilityai/sdxl-turbo):").pack(padx=10, pady=10)
    model_name_entry = tk.StringVar(value=default_model)
    entry = tk.Entry(dialog, textvariable=model_name_entry)
    entry.pack(padx=10, pady=10)

    submit_called = False
    def submit():
        nonlocal submit_called
        submit_called = True
        dialog.destroy()

    tk.Button(dialog, text="Submit", command=submit).pack(padx=10, pady=10)
    dialog.after(5000, timeout)

    root.wait_window(dialog)
    model_name = model_name_entry.get()
    root.destroy()  # Ensure the root window is destroyed
    return model_name


def main():
    model_name = get_model_name()
    print(f"Using model: {model_name}")

    root = tk.Tk()
    app = ImageGeneratorApp(root, "Dynamic image generator..", model_name)
    root.mainloop()


if __name__ == '__main__':
    main()