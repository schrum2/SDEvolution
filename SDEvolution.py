from image_grid import ImageGridViewer
import tkinter as tk

# Modified generation loop with GUI
def generate_and_display_images(pipe, prompt, start_seed, end_seed, steps):
    root = tk.Tk()
    viewer = ImageGridViewer(root)
    
    for seed in range(start_seed, end_seed + 1):
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt,
            generator=generator,
            num_inference_steps=steps
        ).images[0]
        
        # Add image to viewer
        viewer.add_image(image)
        
        # Update the GUI to show new image
        root.update()
    
    # Start the GUI event loop
    root.mainloop()
    
    # After window is closed, return selected images
    return viewer.get_selected_images()

import torch
from diffusers import StableDiffusionPipeline

#model="runwayml/stable-diffusion-v1-5"
model="stablediffusionapi/deliberate-v2"

guidance_scale=7.5

print(f"Using {model} with guidance scale {guidance_scale}")

pipe = StableDiffusionPipeline.from_pretrained(
    model,
    guidance_scale=guidance_scale,
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Default is PNDMScheduler
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)

#prompt="a photo of an astronaut riding a horse on mars, blazing fast, wind and sand moving back"
#prompt="a photo of a chicken in space"

running = True
while running:
    prompt = input("Image prompt (q to quit): ")
    if prompt == "q": quit()
        
    start_seed = input("Start seed (q to quit): ")
    if start_seed == "q": quit()
    end_seed = input("End seed (q to quit): ")
    if end_seed == "q": quit()

    steps = 20 # input("Num steps (q to quit): ")
    if steps == "q": quit()
    try:
        start_seed = int(start_seed)
        end_seed = int(end_seed)
        steps = int(steps)

        selected_images = generate_and_display_images(
            pipe=pipe,
            prompt=prompt,
            start_seed=start_seed,
            end_seed=end_seed,
            steps=steps
        )

        print(selected_images)

    except ValueError:
        print("Seeds and steps must be non-negative integers")