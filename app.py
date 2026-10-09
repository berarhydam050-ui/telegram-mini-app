import base64
import binascii
import io
import os
import sys
import types
import traceback
from typing import Any
import modal

# ============================================================
# 1. CONFIGURATION
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
# 2. IMAGE PREPROCESSING
# ============================================================

def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64") or data.get("data")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("No valid base64 image payload provided.")
    if "," in data and "data:" in data[:30]:
        data = data.split(",", 1)[1]
    
    data = data.strip()
    raw = base64.b64decode(data, validate=False)
    if not raw:
        raise ValueError("Decoded image buffer is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Could not parse image format: {exc}") from exc


def remove_background_and_center(image):
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(remove_background_and_center, "_session", None)
    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    rgba = remove(image, session=session).convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
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
# 3. CPU-SNAPSHOT SCALE-TO-ZERO WORKER
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=15,          # Keeps warm for 15 mins after first use
    enable_memory_snapshot=True,  # CPU snapshot ON
    # Notice: GPU snapshot is intentionally REMOVED here
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=5,
)
class SF3DModel:

    @modal.enter(snap=True)
    def freeze_to_cpu(self):
        """STEP 1: Runs during deployment. Freezes model into standard RAM."""
        import torch
        import os
        import sys
        from huggingface_hub import login

        print("--- STARTING CPU MEMORY SNAPSHOT ---")
        
        # 1. Blind PyTorch to CUDA so SF3D doesn't crash trying to find a GPU
        self._orig_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        torch.set_default_device('cpu')

        # 2. Patch files BEFORE importing SF3D
        network_path = "/app/stable-fast-3d/sf3d/models/network.py"
        if os.path.exists(network_path):
            with open(network_path, "r") as f:
                code = f.read()
            code = code.replace("from torch.amp import custom_bwd, custom_fwd", "from torch.cuda.amp import custom_bwd, custom_fwd")
            code = code.replace('device_type="cuda"', '')
            code = code.replace("device_type='cuda'", '')
            with open(network_path, "w") as f:
                f.write(code)

        baker_path = "/opt/conda/lib/python3.10/site-packages/texture_baker/baker.py"
        if os.path.exists(baker_path):
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

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)

        # 3. Import and load model safely into CPU RAM
        if "/app/stable-fast-3d" not in sys.path:
            sys.path.append("/app/stable-fast-3d")
        from sf3d.system import SF3D

        self.model = SF3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
            device_map="cpu"
        )
        print("--- CPU MEMORY SNAPSHOT COMPLETE ---")


    @modal.enter(snap=False)
    def thaw_to_gpu(self):
        """STEP 2: Runs instantly on Cold Start. Pushes model from RAM to VRAM in ~3 seconds."""
        import torch
        import torch.cuda.amp
        from rembg import new_session

        print("--- COLD START DETECTED: THAWING TO GPU ---")
        
        # 1. Restore PyTorch's ability to see the GPU
        if hasattr(self, "_orig_cuda_available"):
            torch.cuda.is_available = self._orig_cuda_available
        
        torch.set_default_device("cuda")
        self.device = torch.device(GPU_DEVICE)

        # 2. Apply Safe AMP Wrappers for PyTorch
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

        # 3. Push Model to GPU VRAM and Init background remover
        self.model.to(self.device)
        self.model.eval()
        remove_background_and_center._session = new_session("u2net")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        print("--- GPU THAW COMPLETE. READY. ---")


    @modal.method()
    def process_image(self, item: dict):
        import torch
        import torch.cuda.amp

        # Re-enforce AMP patches during execution
        if not hasattr(torch, "amp"):
            torch.amp = types.ModuleType("amp")
        torch.amp.custom_fwd = getattr(torch.cuda.amp, "custom_fwd", torch.amp.custom_fwd)
        torch.amp.custom_bwd = getattr(torch.cuda.amp, "custom_bwd", torch.amp.custom_bwd)

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

            return {
                "success": True,
                "model": encoded,
                "model_base64": encoded,
                "format": "glb",
            }
        finally:
            torch.cuda.empty_cache()

# ============================================================
# 4. FASTAPI WEBHOOK
# ============================================================

@app.function(image=image, scaledown_window=15)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/")
    @web_app.post("/generate")
    async def run_generate(request: Request):
        try:
            data = await request.json()
            model_instance = SF3DModel()
            return await model_instance.process_image.remote.aio(data)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc))

    return web_app
