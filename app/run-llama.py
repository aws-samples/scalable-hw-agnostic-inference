import os
import math
import time
import random
import gradio as gr
from matplotlib import image as mpimg
from fastapi import FastAPI
import torch
from huggingface_hub import login

pod_name=os.environ['POD_NAME']
model_id=os.environ['MODEL_ID']
compiled_model_id=os.environ['COMPILED_MODEL_ID']
device=os.environ["DEVICE"]
hf_token=os.environ['HUGGINGFACE_TOKEN']
max_new_tokens=int(os.environ['MAX_NEW_TOKENS'])

login(hf_token,add_to_git_credential=True)

if device=='xla':
  from optimum.neuron import NeuronModelForCausalLM
elif device=='cuda':
  from transformers import AutoModelForCausalLM,BitsAndBytesConfig
  quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

def gentext(prompt):
  start_time = time.time()
  if device=='xla':
    inputs = tokenizer(prompt, return_tensors="pt")
  elif device=='cuda':
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

  outputs = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=True,use_cache=True,temperature=0.7,top_k=50,top_p=0.9)
  outputs = outputs[0, inputs.input_ids.size(-1):]
  response = tokenizer.decode(outputs, skip_special_tokens=True)
  total_time =  time.time()-start_time
  return str(response), str(total_time)

if device=='xla':
  model = NeuronModelForCausalLM.from_pretrained(compiled_model_id,use_cache=True,use_auth_token=hf_token)
elif device=='cuda': 
  model = AutoModelForCausalLM.from_pretrained(model_id,use_cache=True,device_map='auto',torch_dtype=torch.float16,quantization_config=quantization_config,)
  
gentext("write a poem")


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
