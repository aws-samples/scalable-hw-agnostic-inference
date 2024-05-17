import os
import math
import torch
import logging
import numpy as np
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler


pod_name=os.environ['POD_NAME']
device=os.environ["DEVICE"]
model_id=os.environ['MODEL_ID']
num_inference_steps=int(os.environ['NUM_OF_RUNS_INF'])
height=int(os.environ['HEIGHT'])
width=int(os.environ['WIDTH'])

logger = logging.getLogger(__name__)
DTYPE = torch.bfloat16

class DiffusersHandler(BaseHandler, ABC):
  def __init__(self):
    self.initialized = False

  def initialize(self, ctx):
    self.manifest = ctx.manifest
    logger.info("properties: %s", ctx.system_properties)
    self.pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
    self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    self.initialized = True
    logger.info("Diffusion model from path %s loaded successfully",model_id)
  
  def preprocess(self, requests):
    inputs = []
    for _, data in enumerate(requests):
      input_text = data.get("data")
      if input_text is None:
        input_text = data.get("body")
      if isinstance(input_text, (bytes, bytearray)):
        input_text = input_text.decode("utf-8")
      inputs.append(input_text)
    return inputs

  def inference(self, inputs):
    model_args={'prompt': inputs,'num_inference_steps': num_inference_steps,}
    logger.info("inference with model args: %s", str(model_args))
    inferences = self.pipe(**model_args).images
    inferences = "override"
    return inferences
    
  def postprocess(self, inference_output):
    images = []
    for image in inference_output:
      images.append(np.array(image).tolist())
    return images
