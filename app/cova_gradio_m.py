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

for m in models:
    m["image_url"]   = f'http://{os.environ[m["host_env"]]}:{os.environ[m["port_env"]]}/generate'
    m["caption_url"] = f'http://{os.environ[m["caption_host_env"]]}:{os.environ[m["caption_port_env"]]}/generate'
    m["encoder_url"] = f'http://{os.environ[m["encoder_host_env"]]}:{os.environ[m["encoder_port_env"]]}/generate'

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

    # Generate the caption
    img_b64 = pil_to_base64(image)
    cap_payload = {
       "prompt": "Describe this image",
       "image":  img_b64,
       "max_new_tokens": model_cfg["caption_max_new_tokens"],
    }
    cap_json, cap_latency = await post_json(client, model_cfg["caption_url"], cap_payload)
    caption = base64.b64decode(cap_json["text"]).decode()
    
    #Generate the embeddings
    if "encoder_url" in model_cfg:
       #enc_json, enc_latency = await post_json(
       #  client, model_cfg["encoder_url"], {"text": caption}
       #)
       #encoded = enc_json.get("encoded", str(enc_json))
       enc_payload = {
         "prompt": caption,
         "max_new_tokens": model_cfg.get("encoder_max_new_tokens", 256)
       }
       enc_json, enc_latency = await post_json(client, model_cfg["encoder_url"], enc_payload)
       encoded = base64.b64decode(enc_json["text"]).decode()
       enc_latency_s = f"{enc_latency:.2f}s"
    return (image, f"{img_latency:.2f}s", caption, f"{cap_latency:.2f}s",encoded,enc_latency_s,)

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
    gr.Markdown("# ⚡ Flux Image-Gen + vLLM(Multimodal Models) Caption + T5 Encoder Demo")
    gr.Markdown("Enter a text prompt ➜ model draws an image ➜ LLM describes the image. ➜ Generate embeddings")

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
            enc_out_components:  list = []
            enc_lat_components:  list = []

            for cfg in models:
                with gr.Group():
                    gr.Markdown(f"### {cfg['name']}")
                    img = gr.Image(height=cfg["height"]//2,
                                   width=cfg["width"]//2,
                                   interactive=False)
                    lat = gr.Markdown()
                    cap = gr.Markdown()
                    cap_lat = gr.Markdown()
                    enc = gr.Markdown()
                    enc_lat = gr.Markdown()
                    img_out_components.append(img)
                    img_lat_components.append(lat)
                    cap_out_components.append(cap)
                    cap_lat_components.append(cap_lat)
                    enc_out_components.append(enc)
                    enc_lat_components.append(enc_lat)


    # wire them all up
    btn_generate.click(
        orchestrate_calls,
        inputs=[prompt_in, steps_in],
        outputs=(
            img_out_components +
            img_lat_components +
            cap_out_components +
            cap_lat_components +
            enc_out_components +
            enc_lat_components
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
