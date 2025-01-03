import math
import time
import argparse
import torch
import torch.nn as nn
import torch_neuronx
import neuronx_distributed
import os
import gradio as gr
from fastapi import FastAPI
from huggingface_hub import login

from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Any, Dict, Optional, Union


prompt= "A cat holding a sign that says hello world" 
num_inference_steps=10

model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
pod_name=os.environ['POD_NAME']
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
height=int(os.environ['HEIGHT'])
width=int(os.environ['WIDTH'])
max_sequence_length=int(os.environ['MAX_SEQ_LEN'])
guidance_scale=float(os.environ['GUIDANCE_SCALE'])

login(hf_token,add_to_git_credential=True)

COMPILER_WORKDIR_ROOT=os.environ['COMPILER_WORKDIR_ROOT']

TEXT_ENCODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_1/compiled_model/model.pt')
TEXT_ENCODER_2_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_2/compiled_model/model.pt')
VAE_DECODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'decoder/compiled_model/model.pt')

EMBEDDERS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/embedders')
OUT_LAYERS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/out_layers')
SINGLE_TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/single_transformer_blocks')
TRANSFORMER_BLOCKS_DIR = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/transformer_blocks')


class NeuronFluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        config,
        x_embedder,
        context_embedder
    ):
        super().__init__()
        with torch_neuronx.experimental.neuron_cores_context(start_nc=4,
                                                             nc_count=8):
            self.embedders_model = \
                  neuronx_distributed.trace.parallel_model_load(EMBEDDERS_DIR)
            self.transformer_blocks_model = \
                neuronx_distributed.trace.parallel_model_load(
                    TRANSFORMER_BLOCKS_DIR)
            self.single_transformer_blocks_model = \
                neuronx_distributed.trace.parallel_model_load(
                    SINGLE_TRANSFORMER_BLOCKS_DIR)
            self.out_layers_model = \
                neuronx_distributed.trace.parallel_model_load(
                    OUT_LAYERS_DIR)
        self.config = config
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.device = torch.device("cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        hidden_states = self.x_embedder(hidden_states)

        hidden_states, temb, image_rotary_emb = self.embedders_model(
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            txt_ids,
            img_ids
        )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        image_rotary_emb = image_rotary_emb.type(torch.bfloat16)

        encoder_hidden_states, hidden_states = self.transformer_blocks_model(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb
        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states],
                                  dim=1)

        hidden_states = self.single_transformer_blocks_model(
            hidden_states,
            temb,
            image_rotary_emb
        )

        hidden_states = hidden_states.to(torch.bfloat16)

        return self.out_layers_model(
            hidden_states,
            encoder_hidden_states,
            temb
        )


class NeuronFluxCLIPTextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        output = self.encoder(emb)
        output = CLIPEncoderOutput(output)
        return output


class CLIPEncoderOutput():
    def __init__(self, dictionary):
        self.pooler_output = dictionary["pooler_output"]


class NeuronFluxT5TextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        return torch.unsqueeze(self.encoder(emb)["last_hidden_state"], 1)


def load_model(
        prompt,
        height,
        width,
        max_sequence_length,
        num_inference_steps):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    with torch_neuronx.experimental.neuron_cores_context(start_nc=0):
        pipe.text_encoder = NeuronFluxCLIPTextEncoderModel(
            pipe.text_encoder.dtype,
            torch.jit.load(TEXT_ENCODER_PATH))
    with torch_neuronx.experimental.neuron_cores_context(start_nc=8):
        pipe.text_encoder_2 = NeuronFluxT5TextEncoderModel(
            pipe.text_encoder_2.dtype,
            torch.jit.load(TEXT_ENCODER_2_PATH))
    pipe.transformer = NeuronFluxTransformer2DModel(
        pipe.transformer.config,
        pipe.transformer.x_embedder,
        pipe.transformer.context_embedder)
    with torch_neuronx.experimental.neuron_cores_context(start_nc=8):
        pipe.vae.decoder = torch.jit.load(VAE_DECODER_PATH)

    return pipe

def benchmark(n_runs, test_name, model, model_inputs):
    if not isinstance(model_inputs, tuple):
        model_inputs = model_inputs

    warmup_run = model(**model_inputs)

    latency_collector = LatencyCollector()

    for _ in range(n_runs):
        latency_collector.pre_hook()
        res = model(**model_inputs)
        image=res.images[0]
        image.save(os.path.join(COMPILER_WORKDIR_ROOT, "flux-dev.png"))
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

def text2img(prompt,num_inference_steps):
  start_time = time.time()
  model_args={'prompt':prompt,'height':height,'width':width,'max_sequence_length':max_sequence_length,'num_inference_steps': int(num_inference_steps),'guidance_scale':guidance_scale}
  image = model(**model_args).images[0]
  total_time =  time.time()-start_time
  return image, str(total_time)


#if __name__ == '__main__':
model=load_model(
        prompt,
        height,
        width,
        max_sequence_length,
        num_inference_steps
)
model_inputs={'prompt':prompt,'height':height,'width':width,'max_sequence_length':max_sequence_length,'num_inference_steps': num_inference_steps,'guidance_scale':guidance_scale}
test_name="flux1-dev-10runs"
benchmark(1,test_name,model,model_inputs)

app = FastAPI()
@app.get("/health")
def healthy():
  return {"message": pod_name + "is healthy"}

@app.get("/readiness")
def ready():
  return {"message": pod_name + "is ready"}

io = gr.Interface(
        fn=text2img,
        inputs=[
            gr.Textbox(label="Prompt", lines=2, placeholder="Enter your prompt here..."),
            gr.Textbox(label="Inference-steps", lines=1, placeholder="Enter the number of inference steps; high number takes more time but produces better image")
            ],
        outputs = [
            gr.Image(height=height, width=width), 
            gr.Textbox(label="Execution Time")
            ],
        title=f"{model_id} in AWS EC2 {device} instance; pod name {pod_name}"
)
app = gr.mount_gradio_app(app, io, path="/serve")
