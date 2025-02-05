from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import Genome
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)

def next_generation(selected_images):
    global genomes
    global generation

    print(f"Generation {generation}---------------------------")
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
    generation += 1

    fill_with_images_from_genomes(genomes)

def fill_with_images_from_genomes(genomes):
    global pipe
    global root
    global viewer
    global refiner_pipe

    viewer.clear_images()
    neg_prompt = "watermark, blur, low quality, worst quality"
    
    for g in genomes:
        generator = torch.Generator("cuda").manual_seed(g.seed)
        if g.image:
            # used saved image from previous generation
            print(f"Use cached image for {g}")
            image = g.image
        else:
            # generate fresh new image
            print(f"Generate new image for {g}")
            pipe.to("cuda")
            with torch.no_grad():
                base_latents = pipe(
                    prompt = prompt,
                    generator=generator,
                    guidance_scale=g.guidance_scale,
                    num_inference_steps=g.num_inference_steps,
                    negative_prompt = neg_prompt,
                    output_type = "latent"
                ).images[0]

            # Empty VRAM
            pipe.to("cpu")
            torch.cuda.empty_cache()

            with torch.no_grad():
                image = refiner_pipe(
                prompt = prompt,
                negative_prompt = neg_prompt,
                image = [base_latents]
            ).images[0]

            # Empty VRAM
            refiner_pipe.to("cpu")
            torch.cuda.empty_cache()

        # Add image to viewer
        viewer.add_image(image)
        
        # Update the GUI to show new image
        root.update()
    
    print("Make selections and click \"Evolve\"")
    # Start the GUI event loop
    root.mainloop()

#model="runwayml/stable-diffusion-v1-5"
model="stabilityai/stable-diffusion-xl-base-1.0"

print(f"Using {model}")

# I disabled the safety checker. There is a risk of NSFW content.
pipe = StableDiffusionXLPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
)
#pipe.to("cuda") # Do later

refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_model_id,
    torch_dtype = torch.float16
)

# Default is PNDMScheduler
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)

refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
    refiner_pipe.scheduler.config
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

root = tk.Tk()
viewer = ImageGridViewer(root, callback_fn=next_generation)
fill_with_images_from_genomes(genomes)
