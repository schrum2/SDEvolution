"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

import random
import json
from models import SD_MODEL, SDXL_MODEL

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_REFINE_STEP_DELTA = 20 # Made large: actual steps is just 1/4th of parameter value for some reason
MUTATE_MAX_GUIDANCE_DELTA = 1.0

genome_id = 0

class SDGenome:
    def __init__(self, prompt, neg_prompt, seed, steps, guidance_scale, randomize = True, parent_id = None):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
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
        self.num_inference_steps = max(1, self.num_inference_steps) # do not go below 1 step

    def change_guidance_scale(self, delta):
        self.guidance_scale += delta
        self.guidance_scale = max(1.0, self.guidance_scale) # Do not go below 1.0

    def __str__(self):
        return f"SDGenome(id={self.id},parent_id={self.parent_id},prompt=\"{self.prompt}\",neg_prompt=\"{self.neg_prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale})"

    def metadata(self):
        return {
            "model" : SD_MODEL,
            "id" : self.id,
            "parent_id" : self.parent_id,
            "prompt" : self.prompt,
            "neg_prompt" : self.neg_prompt,
            "seed" : self.seed,
            "num_inference_steps" : self.num_inference_steps,
            "guidance_scale" : self.guidance_scale
        }

    def mutate(self):
        if bool(random.getrandbits(1)):
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))

    def mutated_child(self):
        child = SDGenome(self.prompt, self.neg_prompt, self.seed, self.num_inference_steps, self.guidance_scale, False, self.id)
        child.mutate()
        return child

class SDXLGenome(SDGenome):
    def __init__(self, prompt, neg_prompt, seed, steps, guidance_scale, refine_steps, randomize = True, parent_id = None):
        SDGenome.__init__(self, prompt, neg_prompt, seed, steps, guidance_scale, randomize, parent_id)
        self.refine_steps = refine_steps
        self.base_latents = None

        if randomize: 
            self.change_refine_steps(random.randint(-MUTATE_MAX_REFINE_STEP_DELTA, MUTATE_MAX_REFINE_STEP_DELTA))

    def change_refine_steps(self, delta):
        self.refine_steps += delta
        self.refine_steps = max(1, self.refine_steps) # Do not go below 1 step

    def __str__(self):
        return f"SDXLGenome(id={self.id},parent_id={self.parent_id},prompt=\"{self.prompt}\",neg_prompt=\"{self.neg_prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale},refine_steps={self.refine_steps})"

    def mutate(self):
        if bool(random.getrandbits(1)):
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
            self.change_refine_steps(random.randint(-MUTATE_MAX_REFINE_STEP_DELTA, MUTATE_MAX_REFINE_STEP_DELTA))

    def mutated_child(self):
        child = SDXLGenome(self.prompt, self.neg_prompt, self.seed, self.num_inference_steps, self.guidance_scale, self.refine_steps, False, self.id)
        child.mutate()
        return child
