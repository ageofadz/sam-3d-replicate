import sys
import os
from pathlib import Path

sys.path.append("/src/sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask
from huggingface_hub import snapshot_download

class Predictor:
    def __init__(self):
        self.inference = None
    
    def setup(self):
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

    def predict(self, image_path: str, mask_path: str, mask_index: int = 0, seed: int = 42):
        img = load_image(image_path)
        m = load_single_mask(mask_path, index=mask_index)
        output = self.inference(img, m, seed=seed)
        
        out_path = "/tmp/output.ply"
        output["gs"].save_ply(out_path)
        
        return out_path

