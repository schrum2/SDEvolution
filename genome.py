"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

import random
import controlnet_aux

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_REFINE_STEP_DELTA = 20 # Made large: actual steps is just 1/4th of parameter value for some reason
MUTATE_MAX_GUIDANCE_DELTA = 1.0
MUTATE_MAX_CONTROLNET_CONDITIONING_SCALE_DELTA = 0.1

ANNOTATED_CONTROL_NETS = [
    ('control_v11p_sd15_canny', controlnet_aux.CannyDetector()), # This takes extra parameters when applied, so I'm not sure it fits in with the rest
    ('control_v11p_sd15_normalbae', controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")),
    ('control_v11p_sd15_mlsd', controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")),
    ('control_v11p_sd15_lineart', controlnet_aux.LineartDetector.from_pretrained("lllyasviel/Annotators")),
    ('control_v11p_sd15s2_lineart_anime', controlnet_aux.LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")),
    ('control_v11p_sd15_openpose', controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators"))
]

ANNOTATED_CONTROL_NETS_MODEL_INDEX = 0
ANNOTATED_CONTROL_NETS_DETECTOR_INDEX = 1

MUTATE_NEW_SEED_THRESHOLD = 0.3
MUTATE_CONTROLNET_THRESHOLD = 0.6

genome_id = 0

class SDGenome:
    def __init__(self, prompt, neg_prompt, seed, steps = 20, guidance_scale = 7, controlnet_index = None, controlnet_scale = 0.5, image = None, randomize = True, parent_id = None):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        self.controlnet_index = controlnet_index
        self.controlnet_scale = controlnet_scale
        self.image = None
        if randomize: 
            # Randomize all aspects of picture. Seed will drastically change it
            self.set_seed(random.getrandbits(64))
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
        
        global genome_id
        self.id = genome_id
        genome_id += 1
        self.parent_id = parent_id

        self.survivor = False # Did not survive from previous generation
        
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

    def change_controlnet_scale(self, delta):
        self.controlnet_scale += delta
        self.controlnet_scale = max(0.0, self.controlnet_scale) # Do not go below 0.0

    def change_controlnet_index(self, index):
        if self.image: # There has to be an image to work with
            self.controlnet_index = index

    def get_controlnet_model(self):
        return ANNOTATED_CONTROL_NETS[self.controlnet_index][ANNOTATED_CONTROL_NETS_MODEL_INDEX] if self.image else None

    def get_controlnet_detector(self):
        return ANNOTATED_CONTROL_NETS[self.controlnet_index][ANNOTATED_CONTROL_NETS_DETECTOR_INDEX] if self.image else None

    def __str__(self):
        return f"SDGenome(id={self.id},parent_id={self.parent_id},prompt=\"{self.prompt}\",neg_prompt=\"{self.neg_prompt}\",seed={self.seed},steps={self.num_inference_steps},guidance={self.guidance_scale},controlnet={self.get_controlnet_model()},controlnet_scale={self.controlnet_scale})"

    def mutate(self):
        rand_num = random.random()
        if rand_num < MUTATE_NEW_SEED_THRESHOLD:
            # will be a big change
            self.set_seed(random.getrandbits(64))
        elif self.image and rand_num < MUTATE_CONTROLNET_THRESHOLD:
            # Derive new image from old via ControlNet
            self.change_controlnet_index(random.randint(0, len(ANNOTATED_CONTROL_NETS) - 1))
            self.change_controlnet_scale(random.uniform(-MUTATE_MAX_CONTROLNET_CONDITIONING_SCALE_DELTA, MUTATE_MAX_CONTROLNET_CONDITIONING_SCALE_DELTA))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))

    def mutated_child(self):
        child = SDGenome(self.prompt, self.neg_prompt, self.seed, self.num_inference_steps, self.guidance_scale, self.controlnet_index, self.controlnet_scale, self.image, False, self.id)
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
