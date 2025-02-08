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

model_id=os.environ['MODEL_ID']

models = [
    {
        'name': 'Deepseek8B',
        'host_env': 'DS_R1_8B_SERVICE_HOST',
        'port_env': 'DS_R1_8B_SERVICE_PORT'
    },
    {
        'name': 'Deepseek70B',
        'host_env': 'DS_R1_70B_SERVICE_HOST',
        'port_env': 'DS_R1_70B_SERVICE_PORT'
    }
]

for model in models:
    host = os.environ[model['host_env']]
    port = os.environ[model['port_env']]
    model['url'] = f"http://{host}:{port}/generate"

async def fetch_text(client, url, prompt):
    payload = {
        "prompt": prompt
    }
    try:
        response = await client.post(url, json=payload, timeout=60.0)
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

async def call_model_api(prompt):
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_text(client, model['url'], prompt)
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
    gr.Markdown(f"# {model_id} Text Generation App")
    gr.Markdown("Enter a prompt to generate text using different models.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=10, placeholder="Enter your prompt here...",elem_id="prompt-box")
            generate_button = gr.Button("Generate Text",variant="primary")
        
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
        inputs=[prompt],
        outputs=text_components + exec_time_components,
        api_name="generate_text"
    )
app = gr.mount_gradio_app(app, interface, path="/serve")
