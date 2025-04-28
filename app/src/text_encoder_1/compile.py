import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from model import TracingCLIPTextEncoderWrapper
from huggingface_hub import login
from huggingface_hub import whoami
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
try:
  user_info = whoami()
  print(f"Already logged in as {user_info['name']}")
except:
  login(hf_token,add_to_git_credential=True)

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
DTYPE=torch.bfloat16

def trace_text_encoder():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe

    text_encoder = TracingCLIPTextEncoderWrapper(text_encoder)

    emb = torch.zeros((1, 77), dtype=torch.int64)

    text_encoder_neuron = torch_neuronx.trace(
        text_encoder.neuron_text_encoder,
        emb,
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'),
        compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

    torch_neuronx.async_load(text_encoder_neuron)

    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, 'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT,
                                         'compiled_model/model.pt')
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    del text_encoder
    del text_encoder_neuron


if __name__ == '__main__':
    trace_text_encoder()

