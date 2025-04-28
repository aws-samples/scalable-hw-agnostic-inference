import math
import time
import torch
import os
import sys
import yaml
import requests
from PIL import Image
from vllm import LLM, SamplingParams, TextPrompt
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from huggingface_hub import create_repo,upload_folder,login,snapshot_download

hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
repo_id=os.environ['MODEL_ID']
os.environ['NEURON_COMPILED_ARTIFACTS']=repo_id
os.environ['VLLM_NEURON_FRAMEWORK']='neuronx-distributed-inference'
login(hf_token,add_to_git_credential=True)

config_path = "/vllm_config.yaml"
with open(config_path, 'r') as f:
    model_vllm_config_yaml = f.read()

model_vllm_config = yaml.safe_load(model_vllm_config_yaml)

class LatencyCollector:
    def __init__(self):
        self.latency_list = []

    def record(self, latency_sec):
        self.latency_list.append(latency_sec)

    def percentile(self, percent):
        if not self.latency_list:
            return 0.0
        latency_list = sorted(self.latency_list)
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

    def report(self, test_name="Batch Inference"):
        print(f"\n📊 LATENCY REPORT for {test_name}")
        for p in [0, 50, 90, 95, 99, 100]:
            value = self.percentile(p) * 1000
            print(f"Latency P{p}: {value:.2f} ms")


def get_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

# Model Inputs
PROMPTS = ["What is in this image? Tell me a story",
            "What is the recipe of mayonnaise in two sentences?" ,
            "Describe this image",
            "What is the capital of Italy famous for?",
          ]
IMAGES = [get_image("https://github.com/meta-llama/llama-models/blob/main/models/resources/dog.jpg?raw=true"),
          torch.empty((0,0)),
          get_image("https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/nxd-inference-block-diagram.jpg"),
          torch.empty((0,0)),
         ]
SAMPLING_PARAMS = [dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256),
                   dict(top_k=1, temperature=0.9, top_p=1.0, max_tokens=256),
                   dict(top_k=10, temperature=0.9, top_p=0.5, max_tokens=512),
                   dict(top_k=10, temperature=0.75, top_p=0.5, max_tokens=1024),
                  ]


def get_VLLM_mllama_model_inputs(prompt, single_image, sampling_params):
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)
    inputs["multi_modal_data"] = {"image": input_image}
    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)
    return inputs, sampling_params

def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


llm_model = LLM(**model_vllm_config)
latency_collector = LatencyCollector()

assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), \
f"""Text, image prompts and sampling parameters should have the same batch size,
    got {len(PROMPTS)}, {len(IMAGES)}, and {len(SAMPLING_PARAMS)}"""

batched_inputs = []
batched_sample_params = []
for i in range(1,21):
  for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(pmpt, img, params)
        # test batch-size = 1
        start_time = time.time()
        outputs = llm_model.generate(inputs, sampling_params)
        latency_sec = time.time() - start_time
        latency_collector.record(latency_sec)
        print_outputs(outputs)
        batched_inputs.append(inputs)
        batched_sample_params.append(sampling_params)

latency_collector.report("MLLAMA")
