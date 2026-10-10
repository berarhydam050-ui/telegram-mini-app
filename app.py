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

# Container image setup with FastAPI explicitly included
image = (
    modal.Image.from_registry("rhydam12/sf3d-gpu-worker:latest", add_python="3.10")
    .pip_install("fastapi[standard]")
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
    scaledown_window=15,          
    enable_memory_snapshot=True,  
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=5,
)
class SF3DModel:

    @modal.enter(snap=True)
    def freeze_to_cpu(self):
        """STEP 1: Safely mocks missing ComfyUI imports, patches network.py, and freezes model into CPU RAM."""
        import torch
        import sys
        
        print("--- STARTING CPU MEMORY SNAPSHOT ---")
        
        # 1. Inject dynamic in-memory mocks for missing ComfyUI imports
        class MockModule(types.ModuleType):
            def __getattr__(self, name):
                return MockModule(name)
            def __call__(self, *args, **kwargs):
                return self

        sys.modules["comfy"] = MockModule("comfy")
        sys.modules["comfy.model_management"] = MockModule("comfy.model_management")
        sys.modules["folder_paths"] = MockModule("folder_paths")

        # 2. Patch network.py directly inside the container
        path = "/app/stable_fast_3d/sf3d/models/network.py"
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            content = content.replace("from torch.amp", "from torch.cuda.amp")
            content = content.replace('device_type="cuda"', "")
            content = content.replace("device_type='cuda'", "")
            with open(path, "w") as f:
                f.write(content)
            print("Successfully patched network.py container-side!")

        # 3. Blind PyTorch to CUDA so SF3D loads into RAM during build
        self._orig_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        torch.set_default_device('cpu')

        # 4. Import SF3D safely
        if "/app/stable_fast_3d" not in sys.path:
            sys.path.append("/app/stable_fast_3d")
            
        from stable_fast_3d.sf3d.system import SF3D
        
        # 5. Load model weights into memory snapshot
        print("Loading SF3D pipeline state into frozen RAM snapshot...")
        self.pipeline = SF3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
            cache_dir=CACHE_DIR
        )
        self.pipeline.eval()
        print("--- CPU SNAPSHOT COMPLETED SUCCESSFULLY ---")

    @modal.enter(snap=False)
    def hydrate_to_gpu(self):
        """STEP 2: Restores instantly from snapshot and mounts weights to A10G VRAM."""
        import torch
        from rembg import new_session
        
        print("--- RESTORING SNAPSHOT / WAKING INSTANCE ---")
        torch.cuda.is_available = self._orig_cuda_available
        torch.set_default_device("cuda")
        
        self.pipeline.to(GPU_DEVICE)
        remove_background_and_center._session = new_session("u2net")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device(GPU_DEVICE))
        print("--- MODEL PIPELINE ACTIVE ON CUDA VRAM ---")

    @modal.method()
    def process_mesh(self, item: Any) -> dict:
        """Runs SF3D inference and returns a base64 encoded GLB mesh."""
        import torch
        
        pil_img = decode_image(item)
        processed_img = remove_background_and_center(pil_img)
        
        texture_res = int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION)) if isinstance(item, dict) else DEFAULT_TEXTURE_RESOLUTION
        remesh_val = item.get("remesh", REMESH_MODE) if isinstance(item, dict) else REMESH_MODE
        if remesh_val == "none":
            remesh_val = None

        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mesh, _ = self.pipeline.run_image(
                        processed_img,
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
                "glb": encoded,
                "format": "glb",
            }
        finally:
            torch.cuda.empty_cache()

# ============================================================
# 4. ASGI WEB APP ENDPOINT
# ============================================================

@app.function(image=image, scaledown_window=15)
@modal.asgi_app()
def generate():
    """Exposes the scale-to-zero backend pool via standard FastAPI ASGI routing."""
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
            model_worker = SF3DModel()
            return await model_worker.process_mesh.remote.aio(data)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
    
