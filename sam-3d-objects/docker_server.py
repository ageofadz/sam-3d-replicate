import base64
import io
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
import numpy as np

from sam3d_objects.pipeline.inference_pipeline import Inference

app = FastAPI()

# Lazy singleton
_inference = None
_cfg_path = "/checkpoints/pipeline.yaml"  # we will bind-mount this later or adjust


def _ensure_inference():
    global _inference
    if _inference is not None:
        return _inference

    # If you want to use HF inside container, set HF_TOKEN as an env var
    # and download checkpoints to /checkpoints in an init script instead.
    if not os.path.exists(_cfg_path):
        raise RuntimeError(f"Missing pipeline config at {_cfg_path}")

    _inference = Inference(_cfg_path, compile=False)
    return _inference


def _decode_b64_field(item: Dict, *keys: str) -> bytes:
    s = None
    for key in keys:
        if key and key in item and item[key]:
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


@app.post("/run")
def run_sam3d(item: Dict):
    try:
        image_bytes = _decode_b64_field(item, "image_b64", "image")
        mask_bytes = _decode_b64_field(item, "mask_b64", "mask")
        seed = int(item.get("seed", 42))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    image_np = np.array(img)
    mask_np = (np.array(mask) > 0).astype("uint8")

    inference = _ensure_inference()
    out = inference(image_np, mask_np, seed=seed)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp.close()
    out["gs"].save_ply(tmp.name)

    with open(tmp.name, "rb") as f:
        ply_bytes = f.read()

    os.unlink(tmp.name)

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename=\"splat.ply\"'},
    )
