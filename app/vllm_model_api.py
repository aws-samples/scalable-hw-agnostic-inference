import asyncio
import traceback
import math
import boto3
import time
import argparse
import torch
import torch.nn as nn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Union
from huggingface_hub import login
from starlette.responses import StreamingResponse
import base64
from vllm import LLM,SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
#from vllm import LLM
from sentence_transformers import SentenceTransformer
import yaml
import copy
from itertools import count

_generate_sem = asyncio.Semaphore(1)
_req_ctr = count(1)
cw_namespace='hw-agnostic-infer'
default_max_new_tokens=50
cloudwatch = boto3.client('cloudwatch', region_name='us-west-2')

app_name=os.environ['APP']
nodepool=os.environ['NODEPOOL']
pod_name = os.environ['POD_NAME']
hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
repo_id=os.environ['MODEL_ID']
stream_enabled = os.environ["STREAM_ENABLED"].lower() == "true"
os.environ['NEURON_COMPILED_ARTIFACTS']=repo_id


with open("/vllm_config.yaml", "r") as file:
  vllm_config=yaml.safe_load(file)
  for bad in ("show_progress", "disable_log_stats", "use_tqdm"): 
    vllm_config.pop(bad, None)
login(hf_token, add_to_git_credential=True)

base_params = SamplingParams(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_tokens=default_max_new_tokens,
)
base_params.stream = stream_enabled 
base_params.stream_chunk_size = 8

async def gentext(prompt: str, max_new_tokens: int):
    params = copy.copy(base_params)
    params.max_tokens = max_new_tokens
    params.stream = stream_enabled          # True → stream, False → batch

    start  = time.time()
    ttft   = None
    text   = ""

    params.stream = False

    if stream_enabled:
      #params.stream = True
      req_id = f"r{next(_req_ctr)}"
      async for out in model.generate(prompt, params, req_id):
        chunk = out.outputs[0].text
        if ttft is None and chunk:          # first token
            ttft = time.time() - start
        text += chunk
    else:
      async with _generate_sem:
        outputs = await asyncio.to_thread(model.generate,prompt,params)      
      #print(f"DEBUG: in gentext under batch; outputs:{outputs}")
      text=outputs[0].outputs[0].text
      first=outputs[0]
      metrics = getattr(first, "metrics", None)
      if metrics and getattr(metrics, "first_token_time", None):
        ttft = (metrics.first_token_time - metrics.arrival_time) * 1_000
      else:
        ttft = (time.time() - start) * 1_000        
    return text, ttft, time.time() - start

def cw_pub_metric(metric_name,metric_value,metric_unit):
  response = cloudwatch.put_metric_data(
    Namespace=cw_namespace,
    MetricData=[
      {
        'MetricName':metric_name,
        'Value':metric_value,
        'Unit':metric_unit,
       },
    ]
  )
  print(f"DEBUG: in pub_deployment_counter - metric_name:{metric_name}; metric_value:{metric_value}; metric_unit:{metric_unit};response:{response}")
  return response

login(hf_token, add_to_git_credential=True)

async def benchmark(n_runs, test_name,model,prompt,max_new_tokens):
    response_text,ttft,total_time=await gentext(prompt,max_new_tokens)
    latency_collector = LatencyCollector()

    for _ in range(n_runs):
        latency_collector.pre_hook()
        response_text,ttft,total_time=await gentext(prompt,max_new_tokens)
        #print(f"DEBUG: in benchmark:response_text to {prompt} is:{response_text}; ttft is {ttft}; and total_time is {total_time}")
        latency_collector.hook()

    p0_latency_ms = latency_collector.percentile(0) * 1000
    p50_latency_ms = latency_collector.percentile(50) * 1000
    p90_latency_ms = latency_collector.percentile(90) * 1000
    p95_latency_ms = latency_collector.percentile(95) * 1000
    p99_latency_ms = latency_collector.percentile(99) * 1000
    p100_latency_ms = latency_collector.percentile(100) * 1000

    report_dict = dict()
    report_dict["Latency P0"] = f'{p0_latency_ms:.1f}'
    report_dict["Latency P50"]=f'{p50_latency_ms:.1f}'
    report_dict["Latency P90"]=f'{p90_latency_ms:.1f}'
    report_dict["Latency P95"]=f'{p95_latency_ms:.1f}'
    report_dict["Latency P99"]=f'{p99_latency_ms:.1f}'
    report_dict["Latency P100"]=f'{p100_latency_ms:.1f}'

    report = f'RESULT FOR {test_name}:'
    for key, value in report_dict.items():
        report += f' {key}={value}'
    print(report)
    return report

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

class GenerateRequest(BaseModel):
    max_new_tokens: int
    prompt: str

class GenerateBenchmarkRequest(BaseModel):
    n_runs: int
    max_new_tokens: int
    prompt: str

class GenerateResponse(BaseModel):
    text: str = Field(..., description="Base64-encoded text")
    execution_time: float

class GenerateBenchmarkResponse(BaseModel):
    report: str = Field(..., description="Benchmark report")

def load_model():
    if stream_enabled:                    
        ea = AsyncEngineArgs(**vllm_config)
        return AsyncLLMEngine.from_engine_args(ea)
    else:                                
        return LLM(**vllm_config)

model = load_model()
app = FastAPI()

@app.on_event("startup")
async def _warmup():
  await benchmark(10,"warmup",model,"What model are you?",default_max_new_tokens)

@app.post("/benchmark",response_model=GenerateBenchmarkResponse) 
async def generate_benchmark_report(request: GenerateBenchmarkRequest):
  try:
      with torch.no_grad():
        test_name=f'benchmark:{app_name} on {nodepool} with {request.max_new_tokens} output tokens'
        response_report=await benchmark(request.n_runs,test_name,model,request.prompt,request.max_new_tokens)
        report_base64 = base64.b64encode(response_report.encode()).decode()
      return GenerateBenchmarkResponse(report=report_base64)
  except Exception as e:
      traceback.print_exc()
      raise HTTPException(status_code=500, detail=f"{e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text_post(request: GenerateRequest):
  try:
      with torch.no_grad():
        response_text,ttft,total_time=await gentext(request.prompt,request.max_new_tokens)
      counter_metric=app_name+'-counter'
      await asyncio.to_thread(cw_pub_metric,counter_metric,1,'Count')
      counter_metric=nodepool
      await asyncio.to_thread(cw_pub_metric,counter_metric,1,'Count')
      latency_metric=app_name+'-latency'
      await asyncio.to_thread(cw_pub_metric,latency_metric,total_time,'Seconds')
      if ttft is not None:
        ttft_metric=app_name+'-ttft'
        await asyncio.to_thread(cw_pub_metric,ttft_metric,total_time,'Milliseconds')
      text_base64 = base64.b64encode(response_text.encode()).decode()
      return GenerateResponse(text=text_base64, execution_time=total_time)
  except Exception as e:
      traceback.print_exc()
      raise HTTPException(status_code=500, detail=f"text serialization failed: {e}")

# Health and readiness endpoints
@app.get("/health")
def healthy():
    return {"message": f"{pod_name} is healthy"}

@app.get("/readiness")
def ready():
    return {"message": f"{pod_name} is ready"}
