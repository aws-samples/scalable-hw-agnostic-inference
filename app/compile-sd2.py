import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
model_dir=os.environ['COMPILER_WORKDIR_ROOT']
height=int(os.environ['HEIGHT'])
width=int(os.environ['WIDTH'])
batch_size=int(os.environ['BATCH_SIZE'])
hf_token=os.environ['HUGGINGFACE_TOKEN']
hf_repo=os.environ['HUGGINGFACE_REPO']

from huggingface_hub.hf_api import HfFolder
from optimum.neuron import NeuronStableDiffusionPipeline

HfFolder.save_token(hf_token)
compiler_args = {"auto_cast": "none", "auto_cast_type": "bf16","inline_weights_to_neff": "True"}
input_shapes = {"batch_size": batch_size, "height": height, "width": width}
stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_id, export=True, **compiler_args, **input_shapes)
stable_diffusion.save_pretrained(model_dir)
stable_diffusion.push_to_hub(model_dir,repository_id=hf_repo,use_auth_token=True)

