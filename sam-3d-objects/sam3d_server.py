# sam3d_server.py
import base64
import io
import os
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

from sam3d_objects.pipeline.inference_pipeline import Inference


# ---------- config / checkpoints ----------

HF_REPO_ID = "facebook/sam-3d-objects"
HF_PIPELINE_YAML = "pipeline.yaml"
HF_CHECKPOINT_DIR = "/opt/sam3d/checkpoints"  # arbitrary path inside container


def _ensure_checkpoints() -> str:
    """
    Make sure pipeline.yaml + checkpoints exist.
    Downloads from HF to HF_CHECKPOINT_DIR if missing.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(HF_CHECKPOINT_DIR, exist_ok=True)
    cfg_path = os.path.join(HF_CHECKPOINT_DIR, HF_PIPELINE_YAML)
    if os.path.exists(cfg_path):
        return cfg_path

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var is not set inside the container.")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=HF_CHECKPOINT_DIR,
        allow_patterns=["checkpoints/*", HF_PIPELINE_YAML],
        token=token,
    )
    return cfg_path


_inference: Inference | None = None


def get_inference() -> Inference:
    """
    Lazy-init global Inference object on first request.
    Runs on GPU if torch sees CUDA.
    """
    global _inference
    if _inference is not None:
        return _inference

    cfg_path = _ensure_checkpoints()
    _inference = Inference(cfg_path, compile=False)
    return _inference


# ---------- request / response models ----------

class Sam3DRequest(BaseModel):
    # any of these can be full data URLs or raw base64
    image: str | None = None
    image_b64: str | None = None
    mask: str | None = None
    mask_b64: str | None = None
    seed: int | None = 42


# ---------- helpers ----------

def _decode_b64_field(item: Dict, *keys: str) -> bytes:
    """
    Try multiple keys in order, accept:
    - raw base64
    - data:...;base64,<data>
    - strings with whitespace/newlines
    - missing '=' padding
    """
    s = None
    for key in keys:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            s = val
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


def _run_sam3d(image_bytes: bytes, mask_bytes: bytes, seed: int) -> bytes:
    inference = get_inference()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    output = inference(image_np, mask_np, seed=seed)

    import tempfile

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    output["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)
    return ply_bytes


# ---------- FastAPI app ----------

app = FastAPI(title="SAM-3D Objects API", version="0.1.0")


@app.get("/health")
def health():
    gpu = os.environ.get("NVIDIA_VISIBLE_DEVICES", "unknown")
    return {"status": "ok", "gpu": gpu}


@app.post("/run")
def run_sam3d(req: Sam3DRequest):
    try:
        payload = req.model_dump()
        image_bytes = _decode_b64_field(payload, "image_b64", "image")
        mask_bytes = _decode_b64_field(payload, "mask_b64", "mask")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    seed = int(req.seed or 42)

    try:
        ply_bytes = _run_sam3d(image_bytes, mask_bytes, seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="splat.ply"'},
    )
