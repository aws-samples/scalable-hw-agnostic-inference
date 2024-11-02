import os
os.environ["XLA_FLAGS"] = ""
os.environ["TF_XLA_FLAGS"] = ""

model_id=os.environ['MODEL_ID']
device=os.environ["DEVICE"]
model_dir=os.environ['COMPILER_WORKDIR_ROOT']
batch_size=int(os.environ['BATCH_SIZE'])
hf_token=os.environ['HUGGINGFACE_TOKEN']
hf_repo=os.environ['HUGGINGFACE_REPO']

from huggingface_hub.hf_api import HfFolder
from optimum.neuron import NeuronModelForObjectDetection
from optimum.neuron import pipeline
from transformers import AutoImageProcessor

HfFolder.save_token(hf_token)
model = NeuronModelForObjectDetection.from_pretrained(model_id,export=True,batch_size=batch_size) 
model.save_pretrained(model_dir)
model.push_to_hub(model_dir,repository_id=hf_repo,use_auth_token=True)

print(f"Testing the compiled model")
my_model=NeuronModelForObjectDetection.from_pretrained(hf_repo)
preprocessor = AutoImageProcessor.from_pretrained(hf_repo)
my_pipe = pipeline("object-detection", model=my_model, feature_extractor=preprocessor)
pipe("https://farm7.staticflickr.com/6198/6121789455_82a4b0e32e_z.jpg")
pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
