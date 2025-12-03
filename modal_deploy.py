import modal

app = modal.App("sam-3d-objects")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "python3.10 -m ensurepip --upgrade",
        "python3.10 -m pip install --upgrade pip setuptools wheel",
        "python3.10 -m pip install appdirs",
        "git clone https://github.com/ageofadz/sam-3d-replicate.git /src",
        "git clone https://github.com/facebookresearch/sam-3d-objects.git /src/sam-3d-objects",
        "python3.10 -m pip install --no-build-isolation nvidia-pyindex==1.0.9",
        "python3.10 -m pip install -r /src/requirements.txt",
        "python3.10 -m pip install --no-build-isolation -r /src/sam-3d-objects/requirements.inference.txt"
    )
)

@app.cls(
    image=image,
    gpu="a10g",
    timeout=600,
    secrets=[modal.Secret.from_name("hf-token")],
)
class SAM3DModel:
    @modal.enter()
    def setup(self):
        import sys
        import os
        
        os.environ["CONDA_PREFIX"] = "/usr/local"
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        
        sys.path.insert(0, "/src/sam-3d-objects/notebook")
        
        from huggingface_hub import snapshot_download
        from inference import Inference
        
        tag = "hf"
        checkpoints_dir = f"/src/sam-3d-objects/checkpoints/{tag}"
        os.makedirs(checkpoints_dir, exist_ok=True)

        snapshot_download(
            repo_id="facebook/sam-3d-objects",
            token=os.getenv("HF_TOKEN"),
            local_dir=checkpoints_dir,
            local_dir_use_symlinks=False,
        )

        config_path = os.path.join(checkpoints_dir, "pipeline.yaml")
        self.inference = Inference(config_path, compile=False)
    
    @modal.method()
    def predict(self, image_bytes: bytes, mask_bytes: bytes, mask_index: int = 0, seed: int = 42):
        import tempfile
        import sys
        sys.path.insert(0, "/src/sam-3d-objects/notebook")
        from inference import load_image, load_single_mask
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_f:
            img_f.write(image_bytes)
            img_path = img_f.name
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as mask_f:
            mask_f.write(mask_bytes)
            mask_path = mask_f.name
        
        img = load_image(img_path)
        m = load_single_mask(mask_path, index=mask_index)
        output = self.inference(img, m, seed=seed)
        
        out_path = "/tmp/output.ply"
        output["gs"].save_ply(out_path)
        
        with open(out_path, "rb") as f:
            return f.read()

@app.local_entrypoint()
def main(image_path: str, mask_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with open(mask_path, "rb") as f:
        mask_bytes = f.read()
    
    model = SAM3DModel()
    result = model.predict.remote(image_bytes, mask_bytes)
    
    with open("output.ply", "wb") as f:
        f.write(result)
    print("Saved output to output.ply")
