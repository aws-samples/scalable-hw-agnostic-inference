from optimum.neuron import NeuronModelForObjectDetection
from optimum.neuron import pipeline
from transformers import AutoImageProcessor
import torch
import os
import math
import time
import random
import gradio as gr
from matplotlib import image as mpimg
from fastapi import FastAPI
import torch
from huggingface_hub import login
from pydantic import BaseModel

pod_name=os.environ['POD_NAME']
compiled_model_id=os.environ['COMPILED_MODEL_ID']
device=os.environ["DEVICE"]
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()

#TBD remove after issue resolves - https://github.com/huggingface/optimum-neuron/issues/710
os.environ["XLA_FLAGS"] = ""
os.environ["TF_XLA_FLAGS"] = ""

login(hf_token,add_to_git_credential=True)

if device=='xla':
  from optimum.neuron import NeuronModelForImageClassification
elif device=='cuda':
  print(f"TBD")
elif device=='cpu': 
  print(f"TBD")


def detect_obj_image(url):
  start_time = time.time()
  response = pipe(url) 
  total_time = time.time()-start_time
  return response,total_time

if device=='xla':
  preprocessor = AutoImageProcessor.from_pretrained(compiled_model_id)
  model=NeuronModelForObjectDetection.from_pretrained(compiled_model_id)
  pipe = pipeline("object-detection",model=model,feature_extractor=preprocessor)
elif device=='cuda': 
  print(f"TBD")
elif device=='cpu': 
  print(f"TBD")

#warmup
pipe("http://images.cocodataset.org/val2017/000000039769.jpg")

app = FastAPI()
io = gr.Interface(fn=detect_obj_image,inputs=["text"],
    outputs = ["text","text"],
    title = compiled_model_id + ' in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is" + compiled_model_id + " pod " + pod_name + " in AWS EC2 " + device + " instance; try /detectobj http post with image url; /serve "}

class Item(BaseModel):
  prompt: str
  response: str=None
  latency: float=0.0

@app.post("/detectobj")
def classify_image_post(item: Item):
  item.response,item.latency=classify_image(item.prompt)
  return {"image":item.prompt,"response":item.response,"latency":item.latency}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
