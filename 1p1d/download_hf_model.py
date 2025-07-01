from huggingface_hub import login,snapshot_download
import os
repo_id=os.environ['COMPILED_MODEL_ID']
os.environ['NEURON_COMPILED_ARTIFACTS']=repo_id
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
login(hf_token, add_to_git_credential=True)
snapshot_download(repo_id=repo_id,local_dir="/"+repo_id,token=hf_token)
print(f"Repository '{repo_id}' downloaded to '{repo_id}'.")

