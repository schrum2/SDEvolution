"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

import random

MUTATE_MAX_STEP_DELTA = 5
MUTATE_MAX_GUIDANCE_DELTA = 0.5

class Genome:
    def __init__(self, prompt, seed, steps, guidance_scale, randomize = True):
        self.prompt = prompt
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        if randomize: self.mutate()

    def set_seed(self, new_seed):
        self.seed = new_seed 

    def change_inference_steps(self, delta):
        self.num_inference_steps += delta

    def change_guidance_scale(self, delta):
        self.guidance_scale += delta

    def __str__(self):
        return f"Genome(prompt=\"{self.prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale})"

    def mutate(self):
        self.set_seed(random.getrandbits(64))
        self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
        self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))