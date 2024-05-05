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
model_dir=os.environ['COMPILER_WORKDIR_ROOT']

# Define datatype
DTYPE = torch.bfloat16

if device=='xla':
  from optimum.neuron import NeuronStableDiffusionPipeline
elif device=='cuda':
  from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def benchmark(n_runs, test_name, model, model_inputs):
    if not isinstance(model_inputs, tuple):
        model_inputs = model_inputs

    warmup_run = model(**model_inputs)

    latency_collector = LatencyCollector()

    for _ in range(n_runs):
        latency_collector.pre_hook()
        res = model(**model_inputs)
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

    report = f'RESULT FOR {test_name} on {pod_name}:'
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

if device=='xla':
  pipe = NeuronStableDiffusionPipeline.from_pretrained(model_dir)
elif device=='cuda':
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to("cuda")
  #pipe.enable_attention_slicing
  '''
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
  pipe.unet.to(memory_format=torch.channels_last)
  pipe.vae.to(memory_format=torch.channels_last)

  pipe.unet = torch.compile(pipe.unet, 
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
  '''
def text2img(prompt):
  start_time = time.time()
  num_inference_steps=20
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  image = pipe(**model_args).images[0]
  total_time =  time.time()-start_time
  r1 = random.randint(0,99999)
  imgname="image"+str(r1)+".png"
  image.save(imgname)
  image = mpimg.imread(imgname)
  return image, str(total_time)

#warmup
prompt = "a photo of an astronaut riding a horse on mars"
num_inference_steps=2
model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
image = pipe(**model_args).images[0]

app = FastAPI()
io = gr.Interface(fn=text2img,inputs=["text"],
    outputs = [gr.Image(height=512, width=512), "text"],
    title = 'Stable Diffusion 2.1 in AWS EC2 ' + device + ' instance')

@app.get("/")
def read_main():
  return {"message": "This is Stable Diffusion 2.1 pod " + pod_name + " in AWS EC2 " + device + " instance; try /load/{n_runs}/infer/{n_inf} or /serve"}

@app.get("/load/{n_runs}/infer/{n_inf}")
def load(n_runs: int,n_inf: int):
  prompt = "a photo of an astronaut riding a horse on mars"
  #num_inference_steps = n_inf
  if device=='xla':
    num_inference_steps = 3
  elif device=='cuda':
    num_inference_steps = 1
  model_args={'prompt': prompt,'num_inference_steps': num_inference_steps,}
  report=benchmark(n_runs, "stable_diffusion_512", pipe, model_args)
  return {"message": "benchmark report:"+report}

@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

app = gr.mount_gradio_app(app, io, path="/serve")
