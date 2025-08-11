from huggingface_hub import snapshot_download

# Download the whole model repo from Hugging Face
local_dir = snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="../models/TinyLlama-1.1B-Chat-v1.0",
    local_dir_use_symlinks=False   # ensures actual files are copied
)

print(f"Model downloaded to: {local_dir}")
