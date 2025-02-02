"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

class Genome:
    def __init__(self, seed, prompt, steps, guidance_scale):
        self.seed = seed
        self.prompt = prompt
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale

    def change_seed(self, delta):
        self.seed += delta 

    def change_inference_steps(self, delta):
        self.num_inference_steps += delta

    def change_guidance_scale(self, delta):
        self.guidance_scale += delta

    def __str__(self):
        return f"Genome(prompt=\"{self.prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale})"