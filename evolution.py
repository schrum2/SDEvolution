from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import (SDGenome, SDXLGenome)
import torch
from diffusers import EulerDiscreteScheduler
from abc import ABC, abstractmethod

class Evolver(ABC):
    def __init__(self):
        self.population_size = 9
        self.steps = 20
        self.guidance_scale = 7.5
        self.latents_first = False

        self.evolution_history = []

    def start_evolution(self):
        self.genomes = []
        self.generation = 0

        self.root = tk.Tk()
        self.viewer = ImageGridViewer(
            self.root, 
            callback_fn=self.next_generation,
            initial_prompt="",
            initial_neg_prompt="",
            back_fn=self.previous_generation
        )
        self.fill_with_images_from_genomes(self.genomes)

    def previous_generation(self):
        self.genomes = self.evolution_history.pop()
        self.generation -= 1
        self.fill_with_images_from_genomes(self.genomes)

    @abstractmethod
    def initialize_population(self):
        pass

    def next_generation(self,selected_images,prompt,neg_prompt):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        if selected_images == []:
            print("Resetting population and generations--------------------")
            self.initialize_population()
            self.generation = 0
        else:
            print(f"Generation {self.generation}---------------------------")
            for (i,image) in selected_images:
                print(f"Selected for survival: {self.genomes[i]}")
                #self.genomes[i].set_image(image) # Done earlier, for ALL genomes

            # Track history of all genomes
            self.evolution_history.append(self.genomes.copy())

            # Pure elitism
            keepers = [self.genomes[i] for (i,_) in selected_images]

            children = []
            # Fill remaining slots with mutated children
            for i in range(len(keepers), self.population_size):
                g = random.choice(keepers).mutated_child() # New genome
                # prompts may have changed
                g.prompt = prompt
                g.neg_prompt = neg_prompt
                children.append(g)

            # combined population
            self.genomes = keepers + children
            self.generation += 1

        self.fill_with_images_from_genomes(self.genomes)

    def fill_with_images_from_genomes(self,genomes):
        self.viewer.clear_images()
    
        # SDXL generates new latents first before refining generates images
        if self.latents_first:
            # Do process all genomes while first model is in VRAM
            self.pipe.to("cuda")
            for g in self.genomes:
                g.base_latents = self.generate_latents(g)
                    
            # Empty VRAM so that all latents can be refined next
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
            # Put refiner model in VRAM
            self.refiner_pipe.to("cuda")

        for g in self.genomes:
            
            if g.image:
                # used saved image from previous generation
                print(f"Use cached image for {g}")
                image = g.image
            else:
                image = self.generate_image(g)
                g.set_image(image)

            # Add image to viewer
            self.viewer.add_image(image, g.__str__())
        
            # Update the GUI to show new image
            self.root.update()

        if self.latents_first:
            # Take refiner out of VRAM so base model can do in next generation
            self.refiner_pipe.to("cpu")
            torch.cuda.empty_cache()
    
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
        self.genomes = [SDGenome(self.prompt, self.neg_prompt, seed, self.steps, self.guidance_scale) for seed in range(self.population_size)]

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
    def __init__(self, refine):
        Evolver.__init__(self)

        self.refine_steps = 20
        if refine:
            self.latents_first = True
        else:
            self.latents_first = False 

        model="stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Using {model}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16
        )

        if refine:
            self.refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_model,
                torch_dtype = torch.float16
            )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        if refine:
            self.refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.refiner_pipe.scheduler.config
            )

    def initialize_population(self):
        self.genomes = [SDXLGenome(self.prompt, self.neg_prompt, seed, self.steps, self.guidance_scale, self.refine_steps) for seed in range(self.population_size)]

    def generate_latents(self,g):
        # generate latents first
        print(f"Generate base latents for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)
        with torch.no_grad():
            base_latents = self.pipe(
                prompt = g.prompt,
                generator=generator,
                guidance_scale=g.guidance_scale,
                num_inference_steps=g.num_inference_steps,
                negative_prompt = g.neg_prompt,
                output_type = "latent"
            ).images[0]

        return base_latents

    def generate_image(self, g):
        
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)

        if self.latents_first:
            with torch.no_grad():
                image = self.refiner_pipe(
                    prompt = g.prompt,
                    generator=generator,
                    negative_prompt = g.neg_prompt,
                    num_inference_steps=g.refine_steps, # Actual steps is roughly 1/4th of the value provided here, but the exact reason is not clear
                    image = [g.base_latents]
                ).images[0]
        else:
            with torch.no_grad():
                image = self.pipe(
                    prompt = g.prompt,
                    generator=generator,
                    negative_prompt = g.neg_prompt,
                    num_inference_steps=g.num_inference_steps 
                ).images[0]

        return image
