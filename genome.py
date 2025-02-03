"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

import random

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_GUIDANCE_DELTA = 1.0

genome_id = 0

class Genome:
    def __init__(self, prompt, seed, steps, guidance_scale, randomize = True, parent_id = None):
        self.prompt = prompt
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        if randomize: 
            # Randomize all aspects of picture. Seed will drastically change it
            self.set_seed(random.getrandbits(64))
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
        
        global genome_id
        self.id = genome_id
        genome_id += 1
        self.parent_id = parent_id
        self.image = None

    def set_image(self, image):
        """ save phenotype so code does not have to regenerate """
        self.image = image

    def set_seed(self, new_seed):
        self.seed = new_seed 

    def change_inference_steps(self, delta):
        self.num_inference_steps += delta

    def change_guidance_scale(self, delta):
        self.guidance_scale += delta

    def __str__(self):
        return f"Genome(id={self.id},parent_id={self.parent_id},prompt=\"{self.prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale})"

    def mutate(self):
        if bool(random.getrandbits(1)):
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))

    def mutated_child(self):
        child = Genome(self.prompt, self.seed, self.num_inference_steps, self.guidance_scale, False, self.id)
        child.mutate()
        return child