import modal

app = modal.App("sam-3d-objects")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone --recursive https://github.com/ageofadz/sam-3d-replicate.git /src",
        "cd /src && git pull origin main",
        "cd /src && git submodule update --init --recursive",
        "pip install -r /src/requirements.txt"
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
        from huggingface_hub import snapshot_download
        
        sys.path.append("/src/sam-3d-objects/notebook")
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
        sys.path.append("/src/sam-3d-objects/notebook")
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

