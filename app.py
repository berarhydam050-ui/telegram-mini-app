import base64
import io
import os
import sys
import traceback
from typing import Any
import modal

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("rhydam12/sf3d-gpu-worker:latest", add_python="3.10")
    .pip_install("fastapi[standard]")
)

def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64") or data.get("data")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("No valid base64 image payload provided.")
    if "," in data and "data:" in data[:30]:
        data = data.split(",", 1)[1]
    
    raw = base64.b64decode(data.strip(), validate=False)
    if not raw:
        raise ValueError("Decoded image buffer is empty.")

    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGBA")

def remove_background_and_center(image):
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(remove_background_and_center, "_session", None)
    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    rgba = remove(image, session=session).convert("RGBA")
    bbox = rgba.getchannel("A").getbbox() or (0, 0, rgba.width, rgba.height)

    foreground = rgba.crop(bbox)
    side = max(foreground.width, foreground.height)
    target_side = max(1, int(side * 0.85))
    scale = target_side / max(foreground.width, foreground.height)
    new_size = (max(1, int(foreground.width * scale)), max(1, int(foreground.height * scale)))
    foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

    canvas_side = max(new_size)
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    position = ((canvas_side - foreground.width) // 2, (canvas_side - foreground.height) // 2)
    canvas.alpha_composite(foreground, dest=position)
    return canvas

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=15,          
    enable_memory_snapshot=True,  # Enables 5-second instant resume from RAM snapshot!
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
)
class SF3DModel:
    @modal.enter(snap=True)
    def freeze_to_cpu(self):
        import torch
        
        os.environ["HF_HOME"] = CACHE_DIR

        # Create file-based module stubs so inspect/snapshot doesn't crash
        os.makedirs("/app/comfy", exist_ok=True)
        with open("/app/comfy/__init__.py", "w") as f: f.write("")
        with open("/app/comfy/model_management.py", "w") as f: f.write("")
        with open("/app/folder_paths.py", "w") as f: f.write("")
        if "/app" not in sys.path:
            sys.path.append("/app")

        # Patch network.py for PyTorch compatibility
        net_path = "/app/stable_fast_3d/sf3d/models/network.py"
        if os.path.exists(net_path):
            with open(net_path, "r") as f:
                content = f.read()
            content = content.replace("from torch.amp", "from torch.cuda.amp")
            content = content.replace('device_type="cuda"', "")
            content = content.replace("device_type='cuda'", "")
            with open(net_path, "w") as f:
                f.write(content)

        # Temporarily force CPU during snapshot build
        self._orig_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        torch.set_default_device('cpu')

        if "/app/stable_fast_3d" not in sys.path:
            sys.path.append("/app/stable_fast_3d")
            
        from stable_fast_3d.sf3d.system import SF3D
        print("Freezing SF3D pipeline into memory snapshot...")
        self.pipeline = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors"
        )
        self.pipeline.eval()

    @modal.enter(snap=False)
    def hydrate_to_gpu(self):
        import torch
        from rembg import new_session
        
        torch.cuda.is_available = self._orig_cuda_available
        torch.set_default_device("cuda")
        
        self.pipeline.to("cuda:0")
        remove_background_and_center._session = new_session("u2net")
        torch.cuda.empty_cache()

    @modal.method()
    def process_mesh(self, item: Any) -> dict:
        import torch
        img = remove_background_and_center(decode_image(item))
        
        texture_res = int(item.get("texture_resolution", 2048)) if isinstance(item, dict) else 2048
        remesh_val = item.get("remesh", "none") if isinstance(item, dict) else "none"
        if remesh_val == "none":
            remesh_val = None

        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mesh, _ = self.pipeline.run_image(
                    img,
                    bake_resolution=texture_res,
                    remesh=remesh_val,
                )
            buf = io.BytesIO()
            mesh.export(buf, file_type="glb", include_normals=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return {
                "success": True,
                "model": encoded,
                "model_base64": encoded,
                "glb": encoded,
                "format": "glb",
            }
        finally:
            torch.cuda.empty_cache()

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
            return await SF3DModel().process_mesh.remote.aio(data)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
        
