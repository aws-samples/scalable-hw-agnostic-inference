import os
import math
import time
import random
import gradio as gr
from matplotlib import image as mpimg
from fastapi import FastAPI
import torch

pod_name=os.environ['POD_NAME']
model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
compiled_model_id=os.environ['COMPILED_MODEL_ID']
num_inference_steps=int(os.environ['NUM_OF_RUNS_INF'])


# Define datatype
DTYPE = torch.bfloat16

if device=='xla':
  from optimum.neuron import NeuronStableDiffusionPipeline 
elif device=='cuda':
  from diffusers import StableDiffusionPipeline

from diffusers import EulerAncestralDiscreteScheduler

if device=='xla':
  pipe = NeuronStableDiffusionPipeline.from_pretrained(compiled_model_id)
elif device=='cuda':
  pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
  pipe.enable_attention_slicing()
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
  pipe.unet.to(memory_format=torch.channels_last)
  pipe.vae.to(memory_format=torch.channels_last)
  pipe.unet = torch.compile(
    pipe.unet, 
    fullgraph=True, 
    mode="max-autotune"
  )

  pipe.text_encoder = torch.compile(
    pipe.text_encoder,
    fullgraph=True,
    mode="max-autotune",
  )

  pipe.vae.decoder = torch.compile(
    pipe.vae.decoder,
    fullgraph=True,
    mode="max-autotune",
  )

  pipe.vae.post_quant_conv = torch.compile(
    pipe.vae.post_quant_conv,
    fullgraph=True,
    mode="max-autotune-no-cudagraphs",
  )

def text2img(prompt):
  start_time = time.time()
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  image = pipe(**model_args).images[0]
  total_time =  time.time()-start_time
  return image, str(total_time)

app = FastAPI()
io = gr.Interface(fn=text2img,inputs=["text"],
    outputs = [gr.Image(height=512, width=512), "text"],
    title = 'Stable Diffusion 2.1 in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is Stable Diffusion 2.1 pod " + pod_name + " in AWS EC2 " + device + " instance; try /serve"}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
