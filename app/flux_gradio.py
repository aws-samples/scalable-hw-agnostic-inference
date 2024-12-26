import gradio as gr
import requests
from PIL import Image
import io
import os
from fastapi import FastAPI
import base64

app = FastAPI()

model_id=os.environ['MODEL_ID']

model_1='256x144'
model_api_host1=os.environ['FLUX_NEURON_256X144_MODEL_API_SERVICE_HOST']
model_api_port1=os.environ['FLUX_NEURON_256X144_MODEL_API_SERVICE_PORT']
MODEL_API_URL1 = f"http://{model_api_host1}:{model_api_port1}/generate"

model_2='1024x576'
model_api_host2=os.environ['FLUX_NEURON_1024X576_MODEL_API_SERVICE_HOST']
model_api_port2=os.environ['FLUX_NEURON_1024X576_MODEL_API_SERVICE_PORT']
MODEL_API_URL2 = f"http://{model_api_host2}:{model_api_port2}/generate"

model_3='512x512'
model_api_host3=os.environ['FLUX_NEURON_512X512_MODEL_API_SERVICE_HOST']
model_api_port3=os.environ['FLUX_NEURON_512X512_MODEL_API_SERVICE_PORT']
MODEL_API_URL3 = f"http://{model_api_host3}:{model_api_port3}/generate"

def call_model_api(prompt, num_inference_steps):
    results = {}
    for idx, (model_i, url) in enumerate([
         (model_1,MODEL_API_URL1),
         (model_2,MODEL_API_URL2),
         (model_3,MODEL_API_URL3)],start=1):
         try:
           payload = {
             "prompt": prompt,
             "num_inference_steps": int(num_inference_steps)
           }
        
           response = requests.post(url, json=payload)
           response.raise_for_status()  # Raise an error for bad status codes
           data = response.json()
           image_bytes = base64.b64decode(data['image']) 
           image = Image.open(io.BytesIO(image_bytes))
        
           execution_time = data['execution_time']
           results[f"image_{idx}"] = image
           results[f"time_{idx}"] = f"{execution_time:.2f} seconds" 
         except requests.exceptions.RequestException as e:
           results[f"image_{idx}"] = None
           results[f"time_{idx}"] = f"Request Error: {str(e)}"
         except Exception as e:
           results[f"image_{idx}"] = None
           results[f"time_{idx}"] = f"Error: {str(e)}"
    return (
         results.get("image_1"),results.get("time_1"),
         results.get("image_2"),results.get("time_2"),
         results.get("image_3"),results.get("time_3")
       )

@app.get("/health")
def healthy():
    return {"message": "Service is healthy"}

@app.get("/readiness")
def ready():
    return {"message": "Service is ready"}

title=gr.Markdown(f"Image Generation via {model_id} Pipeline")

interface = gr.Interface(
    fn=call_model_api,
    inputs=[
        gr.Textbox(label="Prompt", lines=1, placeholder="Enter your prompt here..."),
        gr.Number(label="Inference Steps", value=10, precision=0, 
                  info="Enter the number of inference steps; higher number takes more time but produces better image")
    ],
    outputs=[
        gr.Image(label=f"Image from {model_1}", height=256, width=144),
        gr.Textbox(label=f"Execution Time ({model_1})"),
        gr.Image(label=f"Image from {model_2}", height=1024, width=576),
        gr.Textbox(label=f"Execution Time ({model_2})"),
        gr.Image(label=f"Image from {model_3}", height=512, width=512),
        gr.Textbox(label=f"Execution Time ({model_3})"),
    ],
    description="Enter a prompt and specify the number of inference steps to generate an image using the model pipeline."
)

app = gr.mount_gradio_app(app,title + interface, path="/serve")
