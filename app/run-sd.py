import os
import math
import time
import random
import gradio as gr
from matplotlib import image as mpimg
from fastapi import FastAPI
import torch
from pydantic import BaseModel,ConfigDict
from typing import Optional
from PIL import Image
import base64
from io import BytesIO

pod_name=os.environ['POD_NAME']
model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
compiled_model_id=os.environ['COMPILED_MODEL_ID']
num_inference_steps=int(os.environ['NUM_OF_RUNS_INF'])

            
# Define datatype
DTYPE = torch.bfloat16

if device=='xla':
  from optimum.neuron import NeuronStableDiffusionPipeline 
elif device=='cuda' or device=='triton':
  from diffusers import StableDiffusionPipeline

from diffusers import EulerAncestralDiscreteScheduler

if device=='xla':
  pipe = NeuronStableDiffusionPipeline.from_pretrained(compiled_model_id)
elif device=='cuda' or device=='triton':
  pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None,torch_dtype=DTYPE).to("cuda")
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
  if device=='triton':
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.unet = torch.compile(
      pipe.unet, 
      fullgraph=True, 
      mode="max-autotune-no-cudagraphs"
    )

    pipe.text_encoder = torch.compile(
      pipe.text_encoder,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )

    pipe.vae.decoder = torch.compile(
      pipe.vae.decoder,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )

    pipe.vae.post_quant_conv = torch.compile(
      pipe.vae.post_quant_conv,
      fullgraph=True,
      mode="max-autotune-no-cudagraphs",
    )

  pipe.enable_attention_slicing()

def text2img(prompt):
  start_time = time.time()
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  image = pipe(**model_args).images[0]
  total_time =  time.time()-start_time
  return image, str(total_time)

#warm up inference to trigger model compilation
prompt="portrait photo of a old warrior chief"
model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
image = pipe(**model_args).images[0]
print("type(pipe(**model_args).images[0]):"+str(type(pipe(**model_args).images[0])))

app = FastAPI()

io = gr.Interface(fn=text2img,inputs=["text"],
    outputs = [gr.Image(height=512, width=512), "text"],
    title = 'Stable Diffusion 2.1 in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is Stable Diffusion 2.1 pod " + pod_name + " in AWS EC2 " + device + " instance; try /serve or /load/1"}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

class Item(BaseModel):
  prompt: str
  response: Optional[Image.Image]=None
  latency: float = 0.0
  model_config = ConfigDict(arbitrary_types_allowed=True)

@app.post("/genimage")
def generate_text_post(item: Item):
  item.response,item.latency=text2img(item.prompt)
  img=item.response
  buffered = BytesIO()
  img.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return {"prompt":item.prompt,"response":img_str,"latency":item.latency}


@app.get("/load/{n_inf}")
def load(n_inf: int):
  prompt = "a photo of an astronaut riding a horse on mars"
  num_inference_steps = n_inf
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  pipe(**model_args).images[0]
  return {"message": "1"}

app = gr.mount_gradio_app(app, io, path="/serve")
