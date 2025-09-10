from huggingface_hub import snapshot_download

# Replace with your base model repo (example: Meta-Llama/Llama-3.2-1B)
base_model_repo = "meta-llama/Llama-3.2-1B-Instruct"

# Download all model files locally
local_dir = snapshot_download(
    repo_id=base_model_repo,
    resume_download=True,
    local_dir="models/llama-3.2-1b",   # folder where model gets stored
    local_dir_use_symlinks=False       # set to True if you prefer symlinks
)

print(f"Model downloaded to: {local_dir}")