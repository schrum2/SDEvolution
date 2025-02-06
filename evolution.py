from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import (SDGenome, SDXLGenome)
import torch
from diffusers import EulerDiscreteScheduler

class Evolver:
    def __init__(self):
        self.population_size = 9
        self.steps = 20
        self.guidance_scale = 7.5

    def start_evolution(self):
        self.prompt = input("Image prompt: ")
        self.initialize_population()

        self.generation = 0

        self.root = tk.Tk()
        self.viewer = ImageGridViewer(self.root, callback_fn=self.next_generation)
        self.fill_with_images_from_genomes(self.genomes)

    def next_generation(self,selected_images):
        print(f"Generation {self.generation}---------------------------")
        for (i,image) in selected_images:
            print(f"Selected for survival: {self.genomes[i]}")
            self.genomes[i].set_image(image)

        # Pure elitism
        keepers = [self.genomes[i] for (i,_) in selected_images]

        children = []
        # Fill remaining slots with mutated children
        for i in range(len(keepers), self.population_size):
            children.append(random.choice(keepers).mutated_child())

        # combined population
        self.genomes = keepers + children
        self.generation += 1

        self.fill_with_images_from_genomes(self.genomes)

    def fill_with_images_from_genomes(self,genomes):
        self.viewer.clear_images()
    
        for g in self.genomes:
            
            if g.image:
                # used saved image from previous generation
                print(f"Use cached image for {g}")
                image = g.image
            else:
                image = self.generate_image(g)

            # Add image to viewer
            self.viewer.add_image(image)
        
            # Update the GUI to show new image
            self.root.update()
    
        print("Make selections and click \"Evolve\"")
        # Start the GUI event loop
        self.root.mainloop()

from diffusers import StableDiffusionPipeline

class SDEvolver(Evolver):
    def __init__(self):
        Evolver.__init__(self)

        #model="runwayml/stable-diffusion-v1-5"
        model="stablediffusionapi/deliberate-v2"

        print(f"Using {model}")

        # I disabled the safety checker. There is a risk of NSFW content.
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            safety_checker = None,
            requires_safety_checker = False
        )
        self.pipe.to("cuda")

        # Default is PNDMScheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def initialize_population(self):
        self.genomes = [SDGenome(self.prompt, seed, self.steps, self.guidance_scale) for seed in range(self.population_size)]

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)
        image = self.pipe(
            g.prompt,
            generator=generator,
            guidance_scale=g.guidance_scale,
            num_inference_steps=g.num_inference_steps
        ).images[0]

        return image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)

class SDXLEvolver(Evolver):
    def __init__(self):
        Evolver.__init__(self)

        self.refine_steps = 20
 
        model="stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Using {model}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16
        )

        self.refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.refiner_model,
            torch_dtype = torch.float16
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.refiner_pipe.scheduler.config
        )

    def initialize_population(self):
        self.genomes = [SDXLGenome(self.prompt, seed, self.steps, self.guidance_scale, self.refine_steps) for seed in range(self.population_size)]

    def generate_image(self, g):
        neg_prompt = "watermark, blur, low quality, worst quality"

        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)
        self.pipe.to("cuda")
        with torch.no_grad():
            base_latents = self.pipe(
                prompt = g.prompt,
                generator=generator,
                guidance_scale=g.guidance_scale,
                num_inference_steps=g.num_inference_steps,
                negative_prompt = neg_prompt,
                output_type = "latent"
            ).images[0]

        # Empty VRAM
        self.pipe.to("cpu")
        torch.cuda.empty_cache()

        self.refiner_pipe.to("cuda")
        with torch.no_grad():
            image = self.refiner_pipe(
                prompt = g.prompt,
                negative_prompt = neg_prompt,
                image = [base_latents]
            ).images[0]

        # Empty VRAM
        self.refiner_pipe.to("cpu")
        torch.cuda.empty_cache()

        return image
