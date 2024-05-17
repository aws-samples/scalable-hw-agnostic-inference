import os
import math
import torch
import logging
import numpy as np
from abc import ABC
from ts.torch_handler.base_handler import BaseHandler
from diffusers import EulerAncestralDiscreteScheduler

pod_name=os.environ['POD_NAME']
device=os.environ["DEVICE"]
model_id=os.environ['MODEL_ID']
num_inference_steps=int(os.environ['NUM_OF_RUNS_INF'])
height=int(os.environ['HEIGHT'])
width=int(os.environ['WIDTH'])

if device=='xla':
  from optimum.neuron import NeuronStableDiffusionPipeline
elif device=='cuda':
  from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)
DTYPE = torch.bfloat16

class DiffusersHandler(BaseHandler, ABC):
  def __init__(self):
    self.initialized = False

  def initialize(self, ctx):
    self.manifest = ctx.manifest
    print("properties:",ctx.system_properties)
    if device=='xla':
      self.pipe = NeuronStableDiffusionPipeline.from_pretrained(compiled_model_id)
    elif device=='cuda':
      self.pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
      self.pipe.unet.to(memory_format=torch.channels_last)
      self.pipe.vae.to(memory_format=torch.channels_last)
      self.pipe.unet = torch.compile(
        self.pipe.unet,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs"
      )
      self.pipe.text_encoder = torch.compile(
        self.pipe.text_encoder,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
      self.pipe.vae.decoder = torch.compile(
        self.pipe.vae.decoder,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
      self.pipe.vae.post_quant_conv = torch.compile(
        self.pipe.vae.post_quant_conv,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
    self.pipe.enable_attention_slicing()
    self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    self.initialized = True
    print("Diffusion model from path ",model_id," loaded successfully; model state is ",self.initialized)
  
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
    print("inference with model args:",str(model_args))
    inferences = self.pipe(**model_args).images
    return inferences
    
  def postprocess(self, inference_output):
    inference_output = [1, 2, 3, 4, 5]
    print("postprocess inference_output:",str(inference_output))
    images = []
    for image in inference_output:
      images.append(np.array(image).tolist())
    return images