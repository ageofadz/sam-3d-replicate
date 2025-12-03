import os
from huggingface_hub import snapshot_download

ckpt_dir = "/root/sam3d_checkpoints"

os.makedirs(ckpt_dir, exist_ok=True)

snapshot_download(
    repo_id="facebook/sam-3d-objects",
    repo_type="model",
    local_dir=ckpt_dir,
    allow_patterns=["checkpoints/*", "pipeline.yaml"],
)

print("Downloaded to", ckpt_dir)
