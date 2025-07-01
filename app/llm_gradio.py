import gradio as gr
import requests
from PIL import Image
import io
import os
from fastapi import FastAPI
import base64
import asyncio
import httpx
import traceback
import json
import time

MODELS_FILE_PATH=os.environ['MODELS_FILE_PATH']

def load_models_config():
  try:
    with open(MODELS_FILE_PATH, "r") as f:
      models = json.load(f)
      return models
  except Exception as e:
    print(f"Error loading models config: {e}")
    return []

app = FastAPI()

models = load_models_config()

for model in models:
    svc = model['service']
    model['url'] = f"http://{svc}.default.svc.cluster.local:8000"

async def fetch_text(client, url, prompt, model_name, max_tokens, temperature):
    endpoint = f"{url}/v1/completions"
    payload = {
      "model": model_name,
      "prompt": [prompt],
      "max_tokens": max_tokens,
      "temperature": temperature,
      "stream": True, 
    }
    start = time.time()
    ttft = None
    last = start
    tpot = []
    text_accum = ""
    try:
        async with client.stream("POST", endpoint, json=payload, timeout=60.0) as resp:
          resp.raise_for_status()
          async for line in resp.aiter_lines():
            if not line.startswith("data:"):
              continue
            data_str = line[len("data:"):].strip()
            if not data_str or data_str == "[DONE]":
              continue
            chunk_obj = json.loads(data_str)
            chunk = chunk_obj["choices"][0]["text"]

            now = time.time()
            if ttft is None:
              ttft = now - start
            else:
              tpot.append(now - last)
            last = now
            text_accum += chunk
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"
    total = last - start
    tpot_avg = (sum(tpot) / len(tpot)) if tpot else 0.0
    metrics = (
        f"TTFT={ttft*1000:.1f}ms, "
        f"TPUT_avg={tpot_avg*1000:.1f}ms, "
        f"total={total:.2f}s"
    )
    return text_accum, metrics

async def fetch_benchmark(client, url, prompt, model_name, n_runs=1, max_tokens=32, temperature=0.0):
    ttfts = []
    tputs = []
    totals  = []
    output_text = ""
    for i in range(n_runs):
        start = time.time()
        total = None
        ttft = None
        last = start
        tpot = []
        text_accum = ""
        payload = {
          "model": model_name,
          "prompt": [prompt],
          "max_tokens": max_tokens,
          "temperature": temperature,
          "stream": True,
        }
        try:
            async with client.stream("POST", f"{url}/v1/completions", json=payload, timeout=300.0) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    obj = json.loads(data_str)
                    chunk = obj["choices"][0]["text"]

                    now = time.time()
                    if ttft is None:
                        ttft = now - start
                    else:
                        tpot.append(now - last)
                    last = now
                    text_accum += chunk
        except Exception as e:
            traceback.print_exc()
            return None, f"Error in run {i+1}: {e}"
        run_total = last - start
        totals.append(run_total)
        ttfts.append(ttft or 0.0)
        tputs.append((sum(tpot)/len(tpot)) if tpot else 0.0)
        if i == 0:
            output_text = text_accum

    # percentile helper
    def pct(arr, p):
        a = sorted(arr)
        idx = min(int(len(a)*p/100), len(a)-1)
        return a[idx]

    percs = [0, 50, 90, 95, 99]
    ttft_vals = [pct(ttfts, p)*1000 for p in percs]
    tput_vals = [pct(tputs, p)*1000 for p in percs]
    avg_total_ms = sum(totals)/len(totals)*1000

    ttft_report = ", ".join(f"P{p}={v:.1f}ms" for p, v in zip(percs, ttft_vals))
    tput_report = ", ".join(f"P{p}={v:.1f}ms" for p, v in zip(percs, tput_vals))
    report = (
      f"TTFT: {ttft_report}; "
      f"TPUT: {tput_report}; "
      f"AvgTotal={avg_total_ms:.1f}ms"
    )

    return output_text, report
'''
    latencies = []
    first_text = None
    for i in range(n_runs):
        payload = {
          "model": model_name,
          "prompt": [prompt],
          "max_tokens": max_tokens,
          "temperature": temperature,
        }
        start = time.time()
        try:
            resp = await client.post(f"{url}/v1/completions", json=payload, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            traceback.print_exc()
            return None, f"Error run {i+1}: {e}"

        elapsed = time.time() - start
        latencies.append(elapsed)

        if i == 0:
            first_text = data["choices"][0]["text"]

    # compute percentiles
    latencies.sort()
    def pct(p):
        idx = min(int(len(latencies)*p/100), len(latencies)-1)
        return latencies[idx]

    p0 = pct(0)
    p50 = pct(50)
    p90 = pct(90)
    p95 = pct(95)
    p99 = pct(99)
    p100 = pct(100)

    # format in milliseconds
    report = (
      f"P0={p0*1000:.1f}ms, "
      f"P50={p50*1000:.1f}ms, "
      f"P90={p90*1000:.1f}ms, "
      f"P95={p95*1000:.1f}ms, "
      f"P99={p99*1000:.1f}ms, "
      f"P100={p100*1000:.1f}ms"
    )

    return first_text or "", report
'''
async def call_model_api(model_name,prompt, task_type, n_runs, max_new_tokens, temperature):
    async with httpx.AsyncClient() as client:
      if task_type == "fetch_text":
        tasks = [
          fetch_text(client, model['url'],prompt,model_name,max_new_tokens,temperature,)
          for model in models
        ]
      else: 
        tasks = [
            fetch_benchmark(client,model['url'],prompt,model_name,n_runs,max_new_tokens,temperature,)
            for model in models
        ]
      results = await asyncio.gather(*tasks)
    texts = []
    exec_times = []
    for text,exec_time in results:
      texts.append(text)
      exec_times.append(exec_time)
    return texts + exec_times

@app.get("/health")
def healthy():
    return {"message": "Service is healthy"}

@app.get("/readiness")
def ready():
    return {"message": "Service is ready"}

with gr.Blocks() as interface:
    gr.Markdown(f"# LLM Text Generation App and Benchmark App")
    gr.Markdown("Enter a prompt to generate text using different models.")

    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Textbox(
              label="Model Name",
              value="meta-llama/Llama-3.1-8B",
              placeholder="e.g. meta-llama/Llama-3.1-8B"
            )
            prompt = gr.Textbox(label="Prompt", lines=10, placeholder="Enter your prompt here...",elem_id="prompt-box")
            #generate_button = gr.Button("Generate Text",variant="primary")
            task_type = gr.Dropdown(label="Task Type",choices=["fetch_text", "fetch_benchmark"],value="fetch_text",interactive=True)
            n_runs_box = gr.Number(label="Number of Runs (Benchmark)",value=1)
            max_new_tokens_box = gr.Number(label="Max New Tokens",value=32)
            temperature_box = gr.Number(label="Temperature", value=0.0)
            generate_button = gr.Button("Run Task", variant="primary")
        
        with gr.Column(scale=2):
            text_components = []
            exec_time_components = []

            with gr.Row(equal_height=True):
              for idx, model in enumerate(models):
                 with gr.Column(scale=1,min_width=300):
                     text = gr.Textbox(label=f"Text from {model['name']}",interactive=False)
                     exec_time = gr.Textbox(label=f"Execution Time ({model['name']})",interactive=False,lines=1,placeholder="Execution time will appear here...")
                     text_components.append(text)
                     exec_time_components.append(exec_time)

    # callback for the button
    generate_button.click(
        fn=call_model_api,
        inputs=[model_name,prompt, task_type, n_runs_box, max_new_tokens_box, temperature_box],
        outputs=text_components + exec_time_components,
        api_name="generate_text"
    )
app = gr.mount_gradio_app(app, interface, path="/serve")
