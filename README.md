# Realtime Image Generator App

## TL;DR
This application generates images based on text prompts using a Hugging Face model. It offers real-time preview and controls to tweak various parameters of the image generation process. Key features include:
- **Smooth Transitions**: Utilizes blending of consecutive frames to ensure a continuous and smooth visual transition. Keeps track of the previous image and applies small random changes to it.
- **Fast Generation**: Leverages GPU acceleration (if available) for rapid image generation.
- **Interactive Controls**: Provides sliders and text input for real-time adjustments to the image generation parameters.
- **Iterative Image Generation**: Uses the previous frame as the base image for generating the next frame, ensuring continuity and smoothness in the transitions.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [User Interface](#user-interface)
- [How It Works](#how-it-works)
  - [Model Loading](#model-loading)
  - [Generating Images](#generating-images)
  - [Image Processing](#image-processing)
  - [Blending Images](#blending-images)
- [Code Explanation](#code-explanation)

## Requirements
- Python 3.7 or higher
- The following Python packages:
  - `tkinter`
  - `PIL` (Pillow)
  - `torch`
  - `diffusers`
  - `numpy`
  - `platform`

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv kumori_venv
    source kumori_venv/bin/activate  # For Unix/macOS
    # or for Windows
    .\kumori_venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application**:
    ```sh
    python faster_dynamics.py
    ```

2. **Input Model Name**: On startup, input the name of the Hugging Face model you want to use or wait for 5 seconds to use the default model (`stabilityai/sdxl-turbo`).

3. **Interact with the UI**: Enter a text prompt and adjust parameters as desired. Click the "Toggle Generation" button to start generating images automatically every second. Click the button again to stop the generation.

## User Interface

The application interface consists of the following elements:

1. **Canvas**: Displays the generated image.
2. **Prompt**: A text box to enter the prompt for image generation.
3. **Sliders**:
    - **Strength**: Affects the strength of the guidance during the generation process.
    - **Guidance Scale**: Adjusts how much the model should focus on the prompt versus the initial image.
    - **Num Inference Steps**: Number of steps in the generation process. More steps can produce higher quality images but take longer.
    - **Seed**: Controls the randomness of the generation. Different seeds produce different images for the same prompt.
4. **Toggle Generation Button**: Starts or stops the image generation process.

## How It Works

### Model Loading

The application attempts to load a specified Hugging Face model onto the GPU for faster image generation. If a GPU is unavailable or if loading on GPU fails, it falls back to CPU.

```python
def load_model(self):
    try:
        self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_name, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.device = "cuda"
    except Exception as e:
        self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_name, torch_dtype=torch.float32)
        self.pipe.to("cpu")
        self.device = "cpu"
```

### Generating Images

The application continuously generates images based on the provided prompt and slider values. If the generation is toggled on, a new image generation process triggers every second.

```python
def generate_images(self):
    if self.recording:
        self.process_and_display_frame()
        self.window.after(1000, self.generate_images)
```

### Image Processing

Each image generation involves:

- **Prompt and Seed**: Extracting and using the text prompt and seed value from the UI.
- **Model Inference**: The model generates an image based on the prompt and previous frame.
- **Image Perturbations**: Introducing small random changes to the previous frame to add variation (a.k.a perturbed image).
- **Blending**: Blending the new image with the previous image to ensure smooth transitions between frames.
- **Displaying the Image**: Updating the canvas with the newly generated image.

```python
def process_and_display_frame(self):
    prompt = self.text_input.get("1.0", tk.END).strip()
    seed = self.seed_slider.get()

    if prompt:
        torch.manual_seed(seed)

        if self.previous_frame is None:
            init_image = Image.new('RGB', (self.image_size, self.image_size), color='white')
            transformed_image = self.pipe(prompt=prompt,
                                          image=init_image,
                                          strength=self.strength_slider.get(),
                                          guidance_scale=self.guidance_scale_slider.get(),
                                          num_inference_steps=self.num_steps_slider.get()).images[0]
            self.previous_frame = transformed_image
        else:
            perturbed_image = self.apply_random_perturbations(self.previous_frame)
            transformed_image = self.pipe(prompt=prompt,
                                          image=perturbed_image,
                                          strength=self.strength_slider.get(),
                                          guidance_scale=self.guidance_scale_slider.get(),
                                          num_inference_steps=self.num_steps_slider.get()).images[0]
            blended_image = self.blend_images(self.previous_frame, transformed_image, alpha=0.1)
            self.previous_frame = blended_image  # Update the previous frame for the next iteration
            self.display_transformed_image(blended_image)
            self.frame_count += 1
            if self.frame_count % 20 == 0:  # Change the seed every 20 frames
                self.seed_slider.set(seed + 1)
```

### Blending Images

The blending process is crucial in ensuring smooth transitions. The method involves combining the previously generated image with the current one using an alpha value, which dictates the blend's influence.

**Track Previous Frame**: The application keeps track of the previous image frame (`self.previous_frame`).
**Apply Perturbations**: Small random changes are applied to the previous frame to create a new base image for model inference.
**Blend Frames**:
- Convert both images to NumPy arrays.
- Use a weighted combination of the current and previous frames based on an alpha value (defaulted to 0.1).
- Convert the blended array back to an image.

```python
def blend_images(self, prev_img, curr_img, alpha=0.1):
    prev_array = np.array(prev_img)
    curr_array = np.array(curr_img)
    blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
    return Image.fromarray(blended_array)
```

By blending images, the application ensures that transitions between frames are smooth, which is particularly useful during continuous image generation. This method maintains continuity and avoids abrupt changes or flickering.

### How Blending Works

Blending in this context refers to the process of combining two images (the previous frame and the current frame) to create a smooth transition between them. This is particularly useful in applications where images are generated continuously over time, such as in this image generation application.

**Step-by-Step Blending Process**:
1. **Track the Previous Frame**: The application keeps track of the previously generated image, stored in a variable called `self.previous_frame`.
2. **Generate the Current Frame**: The current frame is generated using the model based on the provided prompt and certain perturbations applied to the previous frame.
3. **Convert Images to Arrays**: Both the previous and current frames are converted to NumPy arrays. This transformation makes it easy to perform pixel-wise operations, a necessity for blending.
4. **Blend the Frames**: 
    - The blending is achieved by taking a weighted average of the corresponding pixels from the previous and current frames. This involves using an alpha value, which determines the balance between the two frames:
        - An alpha value of 0.1 means 10% of the previous frame is combined with 90% of the current frame.
    - The formula used to blend the frames is: `blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)`
5. **Convert the Blended Array Back to an Image**: The blended NumPy array is then converted back to an image.

**Why Blending is Effective**:
- **Continuity**: By retaining a portion of the previous frame, blending maintains visual continuity, ensuring the transition from one frame to the next is subtle.
- **Smoothness**: Any changes introduced in the current frame are softened by the influence of the previous frame, resulting in a smoother overall visual effect.
- **Consistency**: Helps in maintaining consistency in the generated imagery, especially when minor random perturbations are applied to create variety.

### Code Implementation

Here is the relevant code implementing the blending process:

```python
def blend_images(self, prev_img, curr_img, alpha=0.1):
    prev_array = np.array(prev_img)
    curr_array = np.array(curr_img)
    blended_array = (alpha * prev_array + (1 - alpha) * curr_array).astype(np.uint8)
    return Image.fromarray(blended_array)
```

### Displaying Images

The generated image is resized to fit the canvas and displayed.

```python
def display_transformed_image(self, transformed_image):
    photo = ImageTk.PhotoImage(transformed_image.resize((self.image_size, self.image_size), Image.LANCZOS))
    self.output_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    self.output_canvas.image = photo  # Keep a reference!
```

## Code Explanation

### Main Application Loop

The application uses `tkinter` for the graphical user interface. It dynamically updates the user interface with generated images and allows user interactions through the input prompt and sliders.

```python
def main():
    model_name = get_model_name()
    root = tk.Tk()
    app = ImageGeneratorApp(root, "Image Generator App", model_name)
    root.mainloop()

if __name__ == '__main__':
    main()
```

The `ImageGeneratorApp` class integrates all functionalities, from loading the model to generating and displaying images, with a responsive and interactive GUI.
