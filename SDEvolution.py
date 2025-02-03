from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import Genome

# Modified generation loop with GUI
def generate_and_display_images(pipe, genomes):
    root = tk.Tk()
    viewer = ImageGridViewer(root)
    
    for g in genomes:
        generator = torch.Generator("cuda").manual_seed(g.seed)
        if g.image:
            # used saved image from previous generation
            print(f"Use cached image for {g}")
            image = g.image
        else:
            # generate fresh new image
            print(f"Generate new image for {g}")
            image = pipe(
                g.prompt,
                generator=generator,
                guidance_scale=g.guidance_scale,
                num_inference_steps=g.num_inference_steps
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

print(f"Using {model}")

# I disabled the safety checker. There is a risk of NSFW content.
pipe = StableDiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
)
pipe.to("cuda")

# Default is PNDMScheduler
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)

#prompt="a photo of an astronaut riding a horse on mars, blazing fast, wind and sand moving back"
#prompt="a photo of a chicken in space"
#prompt="fighting cats"
#propmt="chickens riding horses"

prompt = input("Image prompt: ")
population_size = 9
steps = 20
guidance_scale = 7.5

genomes = [Genome(prompt, seed, steps, guidance_scale) for seed in range(population_size)]

generation = 0
running = True
while running:
    print(f"Generation {generation}---------------------------")
    selected_images = generate_and_display_images(
        pipe=pipe,
        genomes=genomes
    )

    for (i,image) in selected_images:
        print(f"Selected for survival: {genomes[i]}")
        genomes[i].set_image(image)

    # Pure elitism
    keepers = [genomes[i] for (i,_) in selected_images]

    children = []
    # Fill remaining slots with mutated children
    for i in range(len(keepers), population_size):
        children.append(random.choice(keepers).mutated_child())

    # combined population
    genomes = keepers + children