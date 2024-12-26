import gradio as gr
import requests
from PIL import Image
import io
import os
from fastapi import FastAPI
import base64

app = FastAPI()

model_id=os.environ['MODEL_ID']
model_api_host=os.environ['FLUX_NEURON_256X144_MODEL_API_SERVICE_HOST']
model_api_port=os.environ['FLUX_NEURON_256X144_MODEL_API_SERVICE_PORT']
MODEL_API_URL = f"http://{model_api_host}:{model_api_port}/generate"

def call_model_api(prompt, num_inference_steps):
    try:
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "num_inference_steps": int(num_inference_steps)
        }
        
        # Send POST request to the FastAPI Model Pipeline
        response = requests.post(MODEL_API_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        
        data = response.json()
        image_bytes = base64.b64decode(data['image']) 
        image = Image.open(io.BytesIO(image_bytes))
        
        execution_time = data['execution_time']
        
        return image, f"{execution_time:.2f} seconds"
    except requests.exceptions.RequestException as e:
        # Handle request errors
        return None, f"Request Error: {str(e)}"
    except Exception as e:
        # Handle other errors
        return None, f"Error: {str(e)}"

@app.get("/health")
def healthy():
    return {"message": "Service is healthy"}

@app.get("/readiness")
def ready():
    return {"message": "Service is ready"}

interface = gr.Interface(
    fn=call_model_api,
    inputs=[
        gr.Textbox(label="Prompt", lines=2, placeholder="Enter your prompt here..."),
        gr.Number(label="Inference Steps", value=10, precision=0, 
                  info="Enter the number of inference steps; higher number takes more time but produces better image")
    ],
    outputs=[
        gr.Image(height=512, width=512), 
        gr.Textbox(label="Execution Time")
    ],
    title=f"Image Generation via {model_id} Pipeline",
    description="Enter a prompt and specify the number of inference steps to generate an image using the model pipeline."
)

app = gr.mount_gradio_app(app, interface, path="/serve")

