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
compiled_model_id=os.environ['COMPILED_MODEL_ID']
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
    print("initialize ctx:",str(ctx),flush=True)
    if device=='xla':
      self.pipe = NeuronStableDiffusionPipeline.from_pretrained(compiled_model_id)
    elif device=='cuda':
      self.pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
      '''
      self.pipe.unet.to(memory_format=torch.channels_last)
      self.pipe.vae.to(memory_format=torch.channels_last)
      print("torch.compile before unet",flush=True)
      self.pipe.unet = torch.compile(
        self.pipe.unet,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs"
      )
      print("torch.compile before text_encoder",flush=True)
      self.pipe.text_encoder = torch.compile(
        self.pipe.text_encoder,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
      print("torch.compile before vae.decoder",flush=True)
      self.pipe.vae.decoder = torch.compile(
        self.pipe.vae.decoder,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
      print("torch.compile before vae.post_quant_conv",flush=True)
      self.pipe.vae.post_quant_conv = torch.compile(
        self.pipe.vae.post_quant_conv,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
      )
    '''
    self.pipe.enable_attention_slicing()
    self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    model_args={'prompt': 'a photo of an astronaut riding a horse on mars','num_inference_steps': 2,}
    print("inference with model args:",str(model_args),flush=True)
    inferences = self.pipe(**model_args).images
    self.initialized = True
    print("Diffusion model from path ",model_id," loaded successfully; model state is ",self.initialized,flush=True)
  
  def preprocess(self, data):
    prompt = data.get("prompt")
    if not prompt:
      raise ValueError("Please provide a prompt for image generation.")
    return prompt

  def handle(self,data,ctx):
    if not self.initialized:
      raise Exception(f"Worker is not initialized yet.")
    prompt=self.preprocess(data)
    model_args={'prompt': inputs,'num_inference_steps': num_inference_steps,}
    print("inference with model args:",str(model_args),flush=True)
    inference = self.pipe(**model_args).images[0]
    return self.postprocess(inference)
    
  def postprocess(self, inference_output):
    inference_output = [1]
    image=inference_output
    return image
