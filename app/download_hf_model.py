from huggingface_hub import login,snapshot_download
import os
hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
repo_id=os.environ['COMPILED_MODEL_ID']
repo_dir=os.environ['NEURON_COMPILED_ARTIFACTS']
login(hf_token, add_to_git_credential=True)
snapshot_download(repo_id=repo_id,local_dir=repo_dir,token=hf_token)
print(f"Repository '{repo_id}' downloaded to '{repo_dir}'.")
