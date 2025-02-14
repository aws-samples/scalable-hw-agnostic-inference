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

app = FastAPI()

model_id_a=os.environ['MODEL_ID_A']
model_id_b=os.environ['MODEL_ID_B']
model_id_c=os.environ['MODEL_ID_C']

models = [
    {
        'name': 'Deepseek8B',
        'host_env': 'DS_R1_8B_SERVICE_HOST',
        'port_env': 'DS_R1_8B_SERVICE_PORT'
    },
    {
        'name': model_id_a,
        'host_env': 'DS_R1_70B_A_SERVICE_HOST',
        'port_env': 'DS_R1_70B_A_SERVICE_PORT'
    },
    {
        'name': model_id_b,
        'host_env': 'DS_R1_70B_B_SERVICE_HOST',
        'port_env': 'DS_R1_70B_B_SERVICE_PORT'
    },
    {
        'name': model_id_c,
        'host_env': 'DS_R1_8B_SERVICE_HOST',
        'port_env': 'DS_R1_8B_SERVICE_PORT'
    }
]

for model in models:
    host = os.environ[model['host_env']]
    port = os.environ[model['port_env']]
    model['url'] = f"http://{host}:{port}"

async def fetch_text(client,url,prompt,max_new_tokens=32):
    endpoint = f"{url}/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens
    }
    try:
        response = await client.post(endpoint, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        response_text = base64.b64decode(data['text']).decode('utf-8')
        execution_time = data.get('execution_time', 0)
        return response_text, f"{execution_time:.2f} seconds"
    except httpx.RequestError as e:
        traceback.print_exc()
        return None, f"Request Error: {str(e)}"
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"

async def fetch_benchmark(client, url, prompt, n_runs=1, max_new_tokens=32):
    endpoint = f"{url}/benchmark"
    payload = {
        "prompt": prompt,
        "n_runs": n_runs,
        "max_new_tokens": max_new_tokens
    }
    try:
        response = await client.post(endpoint, json=payload, timeout=300.0)
        response.raise_for_status()
        data = response.json()

        response_text = base64.b64decode(data['report']).decode('utf-8')
        execution_time = data.get('execution_time', 0)

        return response_text, f"{execution_time:.2f} seconds"
    except httpx.RequestError as e:
        traceback.print_exc()
        return None, f"Request Error: {str(e)}"
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"

async def call_model_api(prompt,task_type,n_runs,max_new_tokens):
    async with httpx.AsyncClient() as client:
      if task_type == "fetch_text":
        tasks = [
            fetch_text(client, model['url'], prompt,max_new_tokens)
            for model in models
        ]
      else: 
        tasks = [
            fetch_benchmark(client, model['url'], prompt, n_runs, max_new_tokens)
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
            prompt = gr.Textbox(label="Prompt", lines=10, placeholder="Enter your prompt here...",elem_id="prompt-box")
            #generate_button = gr.Button("Generate Text",variant="primary")
            task_type = gr.Dropdown(label="Task Type",choices=["fetch_text", "fetch_benchmark"],value="fetch_text",interactive=True)
            n_runs_box = gr.Number(label="Number of Runs (Benchmark)",value=1)
            max_new_tokens_box = gr.Number(label="Max New Tokens",value=32)
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
        inputs=[prompt,task_type, n_runs_box, max_new_tokens_box],
        outputs=text_components + exec_time_components,
        api_name="generate_text"
    )
app = gr.mount_gradio_app(app, interface, path="/serve")
