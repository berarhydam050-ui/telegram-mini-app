import base64
import binascii
import io
import inspect
import os
import sys
import textwrap
import threading
import types
import traceback
from typing import Any
import modal

# ============================================================
# 1. CONFIGURATION & INFRASTRUCTURE
# ============================================================

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"

DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_IMAGE_PIXELS = 40_000_000
FOREGROUND_SCALE = 0.85
REMESH_MODE = "none"

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry(
    "rhydam12/sf3d-gpu-worker:latest",
    add_python="3.10",
)

# ============================================================
# 2. BULLETPROOF IMAGE DECODING & PREPROCESSING
# ============================================================

def decode_image(data: Any):
    """Flexible image decoder accepting dicts, raw base64, or Data URI schemes."""
    from PIL import Image, ImageOps

    if isinstance(data, dict):
        data = (
            data.get("image")
            or data.get("image_base64")
            or data.get("imageBase64")
            or data.get("data")
        )

    if not isinstance(data, str) or not data.strip():
        raise ValueError("No valid base64 image payload provided.")

    # Strip Data URI scheme if present
    if "," in data and "data:" in data[:30]:
        data = data.split(",", 1)[1]

    data = data.strip()

    try:
        raw = base64.b64decode(data, validate=False)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 string: {exc}") from exc

    if not raw:
        raise ValueError("Decoded image buffer is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions are invalid.")
            if width * height > MAX_IMAGE_PIXELS:
                source.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
            
            source.load()
            result = ImageOps.exif_transpose(source).convert("RGBA")
            return result
    except Exception as exc:
        raise ValueError(f"Could not parse image format: {exc}") from exc


def remove_background_and_center(image):
    """Safely extracts foreground and centers it on a square canvas."""
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(remove_background_and_center, "_session", None)
    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    try:
        rgba = remove(image, session=session).convert("RGBA")
        alpha = rgba.getchannel("A")
        bbox = alpha.getbbox()
    except Exception:
        # Fallback if rembg fails
        rgba = image.convert("RGBA")
        bbox = rgba.getbbox()

    if bbox is None:
        rgba = image.convert("RGBA")
        bbox = (0, 0, rgba.width, rgba.height)

    foreground = rgba.crop(bbox)
    side = max(foreground.width, foreground.height)
    target_side = max(1, int(side * FOREGROUND_SCALE))
    scale = target_side / max(foreground.width, foreground.height)
    new_size = (max(1, int(foreground.width * scale)), max(1, int(foreground.height * scale)))
    foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

    canvas_side = max(new_size)
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    position = ((canvas_side - foreground.width) // 2, (canvas_side - foreground.height) // 2)
    canvas.alpha_composite(foreground, dest=position)
    return canvas

# ============================================================
# 3. WORKER CLASS WITH SNAPSHOT SAFETY NETS
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=5,
)
class SF3DModel:

    @modal.enter(snap=True)
    def load_model(self):
        import torch
        import torch.cuda.amp
        import torch.nn.functional as F
        from huggingface_hub import login

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in Modal worker container.")

        self.device = torch.device(GPU_DEVICE)

        # Safety Net 1: AMP Keyword Filter Wrapper
        _orig_custom_fwd = torch.cuda.amp.custom_fwd
        _orig_custom_bwd = torch.cuda.amp.custom_bwd

        def safe_custom_fwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return _orig_custom_fwd(*args, **kwargs)

        def safe_custom_bwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return _orig_custom_bwd(*args, **kwargs)

        if not hasattr(torch, "amp"):
            torch.amp = types.ModuleType("amp")

        torch.amp.custom_fwd = safe_custom_fwd
        torch.amp.custom_bwd = safe_custom_bwd
        torch.cuda.amp.custom_fwd = safe_custom_fwd
        torch.cuda.amp.custom_bwd = safe_custom_bwd

        # Safety Net 2: File System Patching
        network_path = "/app/stable-fast-3d/sf3d/models/network.py"
        if os.path.exists(network_path):
            try:
                with open(network_path, "r") as f:
                    code = f.read()
                code = code.replace("from torch.amp import custom_bwd, custom_fwd", "from torch.cuda.amp import custom_bwd, custom_fwd")
                code = code.replace('device_type="cuda"', '')
                code = code.replace("device_type='cuda'", '')
                with open(network_path, "w") as f:
                    f.write(code)
            except Exception as e:
                print(f"Warning: Failed to patch network.py: {e}")

        # Safety Net 3: Texture Baker C++ Safe Wrapper
        baker_path = "/opt/conda/lib/python3.10/site-packages/texture_baker/baker.py"
        if os.path.exists(baker_path):
            try:
                with open(baker_path, "r") as f:
                    baker_code = f.read()
                if "cpu_safe_wrapper" not in baker_code:
                    patch_header = """import torch\n\ndef cpu_safe_wrapper(fn):\n    def wrapper(*args, **kwargs):\n        args_cpu = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]\n        kwargs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}\n        res = fn(*args_cpu, **kwargs_cpu)\n        if isinstance(res, torch.Tensor):\n            return res.cuda()\n        if isinstance(res, (tuple, list)):\n            return type(res)(x.cuda() if isinstance(x, torch.Tensor) else x for x in res)\n        return res\n    return wrapper\n\n"""
                    baker_code = patch_header + baker_code.replace(
                        "torch.ops.texture_baker_cpp.rasterize",
                        "cpu_safe_wrapper(torch.ops.texture_baker_cpp.rasterize)"
                    ).replace(
                        "torch.ops.texture_baker_cpp.interpolate",
                        "cpu_safe_wrapper(torch.ops.texture_baker_cpp.interpolate)"
                    )
                    with open(baker_path, "w") as f:
                        f.write(baker_code)
            except Exception as e:
                print(f"Warning: Failed to patch baker.py: {e}")

        # Safety Net 4: Global Grid Sample Tensor Alignment
        if not getattr(F.grid_sample, "_is_snapshot_safe", False):
            _orig_grid_sample = F.grid_sample
            def _safe_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
                dev = input.device
                if grid.device != dev:
                    grid = grid.to(dev)
                return _orig_grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            _safe_grid_sample._is_snapshot_safe = True
            F.grid_sample = _safe_grid_sample

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)

        if "/app/stable-fast-3d" not in sys.path:
            sys.path.append("/app/stable-fast-3d")

        from sf3d.system import SF3D
        from rembg import new_session

        self.model = SF3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.model.eval()
        self.model.to(self.device)

        remove_background_and_center._session = new_session("u2net")
        models_volume.commit()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        print("SNAPSHOT INITIALIZATION COMPLETE")

    @modal.method()
    def process_image(self, item: dict):
        import torch
        import torch.cuda.amp

        # Post-Thaw CUDA State Re-binding
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            torch.cuda.empty_cache()

        self.model.to(self.device)

        # Enforce AMP wrappers on post-thaw calls
        _orig_custom_fwd = getattr(torch.cuda.amp, "_orig_custom_fwd", torch.cuda.amp.custom_fwd)
        _orig_custom_bwd = getattr(torch.cuda.amp, "_orig_custom_bwd", torch.cuda.amp.custom_bwd)

        def safe_custom_fwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return _orig_custom_fwd(*args, **kwargs)

        def safe_custom_bwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return _orig_custom_bwd(*args, **kwargs)

        if not hasattr(torch, "amp"):
            torch.amp = types.ModuleType("amp")

        torch.amp.custom_fwd = safe_custom_fwd
        torch.amp.custom_bwd = safe_custom_bwd

        original = decode_image(item)
        processed = remove_background_and_center(original)

        texture_res = int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION))
        remesh_val = item.get("remesh", REMESH_MODE)
        if remesh_val == "none":
            remesh_val = None

        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mesh, _ = self.model.run_image(
                        processed,
                        bake_resolution=texture_res,
                        remesh=remesh_val,
                    )

            output = io.BytesIO()
            mesh.export(output, file_type="glb", include_normals=True)
            glb_bytes = output.getvalue()

            if not glb_bytes:
                raise RuntimeError("SF3D engine generated empty GLB buffer.")

            encoded = base64.b64encode(glb_bytes).decode("ascii")

            # Universal response dictionary supporting all possible frontend key lookups
            return {
                "success": True,
                "model": encoded,
                "model_base64": encoded,
                "glb": encoded,
                "result": encoded,
                "format": "glb",
                "mime_type": "model/gltf-binary",
                "texture_resolution": texture_res,
                "remesh": str(remesh_val),
            }

        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"3D Generation Error: {exc}") from exc
        finally:
            torch.cuda.empty_cache()

# ============================================================
# 4. FASTAPI WEBHOOK WITH CORS & FLEXIBLE ENDPOINTS
# ============================================================

@app.function(image=image, scaledown_window=2)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(title="SF3D Image-to-3D API")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.get("/")
    @web_app.get("/health")
    async def health():
        return {"status": "ok", "service": APP_NAME, "model": MODEL_ID}

    @web_app.post("/")
    @web_app.post("/generate")
    async def run_generate(request: Request):
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("Request body must be a valid JSON object.")

            model_instance = SF3DModel()
            result = await model_instance.process_image.remote.aio(data)
            return result

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"SF3D processing failed: {type(exc).__name__}: {exc}",
            ) from exc

    return web_app
        
