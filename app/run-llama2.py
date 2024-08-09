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

if device=='xla':
  from optimum.neuron import NeuronModelForCausalLM
elif device=='cuda':
  from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer

if device=='xla':
  model = NeuronModelForCausalLM.from_pretrained(model_id)
elif device=='cuda': 
  #model = AutoModelForCausalLM.from_pretrained(model_id,load_in_8bit=True,device_map="auto")
  quantization_config = BitsAndBytesConfig(llm_int8_threshold=200.0,load_in_8bit=True)
  model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,device_map = 'auto',quantization_config=quantization_config,)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-chat-hf")

def gentext(prompt):
  start_time = time.time()
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(**inputs,max_new_tokens=128,do_sample=True,temperature=0.9,top_k=50,top_p=0.9)
  outputs = outputs[0, inputs.input_ids.size(-1):]
  response = tokenizer.decode(outputs, skip_special_tokens=True)
  total_time =  time.time()-start_time
  return str(response), str(total_time)

app = FastAPI()
io = gr.Interface(fn=gentext,inputs=["text"],
    outputs = ["text","text"],
    title = model_id + ' in AWS EC2 ' + device + ' instance; pod name ' + pod_name)

@app.get("/")
def read_main():
  return {"message": "This is" + model_id + " pod " + pod_name + " in AWS EC2 " + device + " instance; try /load/{n_runs}/infer/{n_inf} e"}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
