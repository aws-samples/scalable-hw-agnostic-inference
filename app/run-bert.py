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
from fastapi.middleware.cors import CORSMiddleware

pod_name=os.environ['POD_NAME']
model_id=os.environ['MODEL_ID']
compiled_model_id=os.environ['COMPILED_MODEL_ID']
device=os.environ["DEVICE"]
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()

login(hf_token,add_to_git_credential=True)

if device=='xla':
  from optimum.neuron import NeuronModelForSequenceClassification
  model=NeuronModelForSequenceClassification.from_pretrained(compiled_model_id)
elif device=='cuda':
  from transformers import AutoModelForSequenceClassification
  model=AutoModelForSequenceClassification.from_pretrained(model_id).to('cuda')
elif device=='cpu': 
  from transformers import AutoModelForSequenceClassification
  model=AutoModelForSequenceClassification.from_pretrained(model_id)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


def classify_sentiment(prompt):
  start_time = time.time()
  if device=='xla':
    inputs = tokenizer(prompt, return_tensors="pt")
  elif device=='cuda':
    # model = model.to('cuda')  # Move the model to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')  # Move inputs to GPU
    logs = model(**inputs).logits
    sent = model.config.id2label[logs.argmax().item()]
    total_time =  time.time()-start_time
    return sent, total_time
  elif device=='cpu':
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
  
  logits = model(**inputs).logits 
  sentiment = model.config.id2label[logits.argmax().item()]
  total_time =  time.time()-start_time
  return sentiment,total_time

# classify_sentiment("Hamilton is widely celebrated as the best musical of recent years, captivating audiences with its brilliant blend of history, hip-hop, and powerful storytelling.")
classify_sentiment("Hamilton is overrated and fails to live up to the hype as the best musical of past years.")


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

io = gr.Interface(fn=classify_sentiment,inputs=["text"],
    outputs = ["text","text"],
    title = model_id + ' in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is" + model_id + " pod " + pod_name + " in AWS EC2 " + device + " instance; try /load/{n_runs}/infer/{n_inf}; /gentext http post with user prompt "}

class Item(BaseModel):
  prompt: str
  response: str=None
  latency: float=0.0

@app.post("/sentiment")
def classify_text_post(item: Item):
  item.response,item.latency=classify_sentiment(item.prompt)
  return {"prompt":item.prompt,"response":item.response,"latency":item.latency}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
