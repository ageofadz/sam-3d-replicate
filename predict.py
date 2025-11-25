from typing import Any
from cog import BasePredictor, Input, Path
import sys
import os

from huggingface_hub import snapshot_download

sys.path.append("/src/sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask


class Predictor(BasePredictor):
    def setup(self) -> None:
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

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask: Path = Input(
            description="Mask (e.g. PNG) for the object to reconstruct. For now required.",
            default=None,
        ),
        mask_index: int = Input(
            description="Index of the mask if mask is a multi-mask file",
            default=0,
        ),
        seed: int = Input(description="Random seed", default=42),
    ) -> Path:
        img = load_image(str(image))

        if mask is None:
            raise ValueError("Mask file is required for now.")

        m = load_single_mask(str(mask), index=mask_index)

        output = self.inference(img, m, seed=seed)

        out_path = Path("/tmp/output.ply")
        output["gs"].save_ply(str(out_path))

        return out_path