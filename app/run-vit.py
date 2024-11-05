from transformers import AutoImageProcessor,pipeline
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
from PIL import Image
import requests

pod_name=os.environ['POD_NAME']
compiled_model_id=os.environ['COMPILED_MODEL_ID']
device=os.environ["DEVICE"]
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()

#TBD remove after issue resolves - https://github.com/huggingface/optimum-neuron/issues/710
#os.environ["XLA_FLAGS"] = ""
#os.environ["TF_XLA_FLAGS"] = ""

#login(hf_token,add_to_git_credential=True)

if device=='xla':
  from optimum.neuron import NeuronModelForImageClassification
elif device=='cuda':
  # print(f"TBD")
  from transformers import ViTImageProcessor, ViTForImageClassification
elif device=='cpu': 
  print(f"TBD")


def classify_image(url):
  start_time = time.time()

  if device=='xla':
    image = Image.open(requests.get(url, stream=True).raw)
    preprocessor = AutoImageProcessor.from_pretrained(compiled_model_id)
    model=NeuronModelForImageClassification.from_pretrained(compiled_model_id)
    pipe = pipeline("image-classification",model=model,feature_extractor=preprocessor)
    response = pipe(image)[0]['label']
    # response = model.config.id2label
  elif device=='cuda': 
    # print(f"TBD")
    image = Image.open(requests.get(url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained(compiled_model_id)
    model = ViTForImageClassification.from_pretrained(compiled_model_id)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    response = model.config.id2label[predicted_class_idx]

  elif device=='cpu': 
    # print(f"TBD")
    image = Image.open(requests.get(url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained(compiled_model_id)
    model = ViTForImageClassification.from_pretrained(compiled_model_id)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    response = model.config.id2label[predicted_class_idx]
  
   
  total_time = time.time()-start_time
  return response,total_time

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
classify_image(image_url)
#warmup
# pipe("http://images.cocodataset.org/val2017/000000039769.jpg")

app = FastAPI()
io = gr.Interface(fn=classify_image,inputs=["text"],
    outputs = ["text","text"],
    title = compiled_model_id + ' in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is" + compiled_model_id + " pod " + pod_name + " in AWS EC2 " + device + " instance; try /imgcls http post with image url; /serve "}

class Item(BaseModel):
  prompt: str
  response: str=None
  latency: float=0.0

@app.post("/imgcls")
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