import base64
import io
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import tempfile

# Lazy singleton
_inference = None
_pipeline_cfg = None

CHECKPOINT_DIR = "/models/sam3d"


def _ensure_checkpoints() -> str:
    """
    Download checkpoints once into /models/sam3d.
    Relies on HF_TOKEN env var.
    """
    global _pipeline_cfg

    if _pipeline_cfg is not None:
        return _pipeline_cfg

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pipeline_yaml = os.path.join(CHECKPOINT_DIR, "pipeline.yaml")

    if not os.path.exists(pipeline_yaml):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN env var not set in container")

        snapshot_download(
            repo_id="facebook/sam-3d-objects",
            repo_type="model",
            local_dir=CHECKPOINT_DIR,
            allow_patterns=["checkpoints/*", "pipeline.yaml"],
            token=hf_token,
        )

    _pipeline_cfg = pipeline_yaml
    return _pipeline_cfg


def _ensure_inference():
    """
    Lazily import and construct Inference once per container.
    """
    global _inference
    if _inference is not None:
        return _inference

    # Avoid side-effecty __init__ if needed
    os.environ["LIDRA_SKIP_INIT"] = "true"

    cfg_path = _ensure_checkpoints()

    from sam3d_objects.pipeline.inference_pipeline import Inference

    _inference = Inference(cfg_path, compile=False)
    return _inference


def _decode_b64_field(item: Dict, *keys: str) -> bytes:
    """
    Accepts:
    - raw base64
    - data:...;base64,<data>
    - missing '=' padding
    """
    s = None
    for key in keys:
        if key in item and item[key]:
            s = item[key]
            break

    if s is None:
        raise ValueError(f"Missing required field (one of {keys})")

    if s.startswith("data:"):
        s = s.split(",", 1)[1]

    s = s.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)

    return base64.b64decode(s)


def run_sam3d(image_bytes: bytes, mask_bytes: bytes, seed: int) -> bytes:
    """
    Core inference: bytes -> numpy -> Inference -> .ply bytes
    """
    inference = _ensure_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    output = inference(image_np, mask_np, seed=seed)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    output["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)
    return ply_bytes


app = FastAPI()


@app.post("/infer")
def infer(payload: Dict):
    try:
        image_bytes = _decode_b64_field(payload, "image_b64", "image")
        mask_bytes = _decode_b64_field(payload, "mask_b64", "mask")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad base64 input: {e}")

    seed = int(payload.get("seed", 42))

    try:
        ply_bytes = run_sam3d(image_bytes, mask_bytes, seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return StreamingResponse(
        io.BytesIO(ply_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="splat.ply"'},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
