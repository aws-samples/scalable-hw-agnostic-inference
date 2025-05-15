import os
from huggingface_hub import create_repo,upload_folder,login

hf_token=os.environ['HUGGINGFACE_TOKEN'].strip()
repo_id=os.environ['MODEL_ID']
login(hf_token, add_to_git_credential=True)

def push_compiled_model_to_hf(
    local_dir: str,
    repo_id: str,
    commit_message: str,
    token: str = None,
):
    create_repo(
        repo_id=repo_id,
        token=token,
        exist_ok=True,
        private=False
    )

    upload_folder(
        folder_path=local_dir,
        path_in_repo="",
        repo_id=repo_id,
        commit_message=commit_message
    )


push_compiled_model_to_hf(
  local_dir=repo_id,
  repo_id=repo_id,
  commit_message=f"Add NxD compiled model {repo_id} for vLLM; after converting checkpoints"
)

