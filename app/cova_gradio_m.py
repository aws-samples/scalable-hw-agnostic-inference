import os, io, base64, asyncio, json
from typing import Tuple, List
import gradio as gr
import httpx
from PIL import Image
from fastapi import FastAPI
import json

MODELS_FILE_PATH=os.environ['MODELS_FILE_PATH']

def load_models_config():
  try:
    with open(MODELS_FILE_PATH, "r") as f:
      models = json.load(f)
      return models
  except Exception as e:
    print(f"Error loading models config: {e}")
    return []

models = load_models_config()

app = FastAPI()
'''
IMAGE_MODELS = [
    dict(
        name="512 × 512",
        host_env="FLUX_NEURON_512X512_MODEL_API_SERVICE_HOST",
        port_env="FLUX_NEURON_512X512_MODEL_API_SERVICE_PORT",
        height=512,
        width=512,
        # caption backend for this image size
        caption_host_env="MLLAMA_32_11B_VLLM_TRN1_SERVICE_HOST",
        caption_port_env="MLLAMA_32_11B_VLLM_TRN1_SERVICE_PORT",
        # number of caption tokens to ask for per request
        caption_max_new_tokens=64,
    )
]
'''
for m in models:
    m["image_url"]   = f'http://{os.environ[m["host_env"]]}:{os.environ[m["port_env"]]}/generate'
    m["caption_url"] = f'http://{os.environ[m["caption_host_env"]]}:{os.environ[m["caption_port_env"]]}/generate'

async def post_json(client: httpx.AsyncClient, url: str, payload: dict, timeout: float = 60.0):
    start = asyncio.get_event_loop().time()
    r = await client.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    elapsed = asyncio.get_event_loop().time() - start
    return r.json(), elapsed

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

async def fetch_end_to_end(
    client         : httpx.AsyncClient,
    model_cfg      : dict,
    prompt         : str,
    num_steps      : int
) -> Tuple[Image.Image, str, str]:
    # Generate the image
    img_payload = {"prompt": prompt, "num_inference_steps": int(num_steps)}
    img_json, img_latency = await post_json(client, model_cfg["image_url"], img_payload)
    image = Image.open(io.BytesIO(base64.b64decode(img_json["image"])))

    # vLLM to describe that image
    img_b64 = pil_to_base64(image)
    caption_prompt = f"Describe the content of this image (base64 PNG follows): {img_b64}"
    cap_payload = {"prompt": caption_prompt,
                   "max_new_tokens": model_cfg["caption_max_new_tokens"]}
    cap_json, cap_latency = await post_json(client, model_cfg["caption_url"], cap_payload)
    caption = base64.b64decode(cap_json["text"]).decode()

    return image, f"{img_latency:.2f}s", caption, f"{cap_latency:.2f}s"

async def orchestrate_calls(prompt: str, num_steps: int):
    async with httpx.AsyncClient() as client:
        tasks = [fetch_end_to_end(client, cfg, prompt, num_steps) for cfg in models]
        results = await asyncio.gather(*tasks)

    # Flatten results for gradio →  [img, img_lat, caption, cap_lat] * N
    flat: List = []
    for tup in results:
        flat.extend(tup)
    return flat

with gr.Blocks() as interface:
    gr.Markdown("# ⚡ Flux Image-Gen + vLLM(Llama3.2 Multimodal Models) Caption Demo")
    gr.Markdown("Enter a text prompt ➜ model draws an image ➜ LLM describes the image.")

    with gr.Row():
        # user controls
        with gr.Column(scale=1):
            prompt_in   = gr.Textbox(lines=1, label="Prompt")
            steps_in    = gr.Number(label="Inference Steps", value=10, precision=0)
            btn_generate = gr.Button("Generate", variant="primary")

        # results
        with gr.Column(scale=2):
            img_out_components:  list = []
            img_lat_components:  list = []
            cap_out_components:  list = []
            cap_lat_components:  list = []

            for cfg in models:
                with gr.Group():
                    gr.Markdown(f"### {cfg['name']}")
                    img = gr.Image(height=cfg["height"]//2,
                                   width=cfg["width"]//2,
                                   interactive=False)
                    lat = gr.Markdown()
                    cap = gr.Markdown()
                    cap_lat = gr.Markdown()
                    img_out_components.append(img)
                    img_lat_components.append(lat)
                    cap_out_components.append(cap)
                    cap_lat_components.append(cap_lat)

    # wire them all up
    btn_generate.click(
        orchestrate_calls,
        inputs=[prompt_in, steps_in],
        outputs=(
            img_out_components +
            img_lat_components +
            cap_out_components +
            cap_lat_components
        ),
        api_name="generate_and_caption",
    )

app = gr.mount_gradio_app(app, interface, path="/serve")

@app.get("/health")
def healthy():
    return {"message": "Service is healthy"}

@app.get("/readiness")
def ready():
    return {"message": "Service is ready"}
