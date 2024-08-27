import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
model_id=os.environ['MODEL_ID']
model_dir=os.environ['COMPILER_WORKDIR_ROOT']
batch_size=int(os.environ['BATCH_SIZE'])
num_cores=int(os.environ['NUM_CORES'])
sequence_length=int(os.environ['SEQUENCE_LENGTH'])
auto_cast_type=os.environ['AUTO_CAST_TYPE']
hf_token=os.environ['HUGGINGFACE_TOKEN']
hf_repo=os.environ['HUGGINGFACE_REPO']

from huggingface_hub.hf_api import HfFolder
from huggingface_hub import login
from optimum.neuron import NeuronModelForCausalLM

hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
login(hf_token,add_to_git_credential=True)

compiler_args = {"num_cores": num_cores, "auto_cast_type": auto_cast_type}
input_shapes = {"batch_size": ,batch_size "sequence_length": sequence_length}
model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        **compiler_args,
        **input_shapes)

model.save_pretrained(model_dir)
model.push_to_hub(model_dir,repository_id=hf_repo)
