import base64
import binascii
import io
import os
import sys
import types
import traceback
from typing import Any
import modal

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


def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("Provide an image as a base64 string.")
    if data.startswith("data:"):
        header, separator, data = data.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Invalid base64 data URI.")
    if len(data) > 180_000_000:
        raise ValueError("Encoded image exceeds the input limit.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data.") from exc
    if not raw:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 40-megapixel limit.")
            source.load()
            result = ImageOps.exif_transpose(source).convert("RGBA")
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc
    return result


def remove_background_and_center(image):
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(remove_background_and_center, "_session", None)
    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    rgba = remove(image, session=session).convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("No foreground subject was detected.")

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
        from huggingface_hub import login

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the worker.")

        self.device = torch.device(GPU_DEVICE)

        # Fix torch.amp import error directly in Python runtime memory
        if hasattr(torch, "amp"):
            torch.amp.custom_fwd = getattr(torch.amp, "custom_fwd", torch.cuda.amp.custom_fwd)
            torch.amp.custom_bwd = getattr(torch.amp, "custom_bwd", torch.cuda.amp.custom_bwd)
        else:
            torch.amp = types.ModuleType("amp")
            torch.amp.custom_fwd = torch.cuda.amp.custom_fwd
            torch.amp.custom_bwd = torch.cuda.amp.custom_bwd

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)

        # CPU Safe Wrapper for texture_baker C++ ops
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
        print("Snapshot model initialization complete.")

    @modal.method()
    def process_image(self, item: dict):
        import torch
        import torch.cuda.amp

        # Re-apply memory patch on wake-up
        if hasattr(torch, "amp"):
            torch.amp.custom_fwd = getattr(torch.amp, "custom_fwd", torch.cuda.amp.custom_fwd)
            torch.amp.custom_bwd = getattr(torch.amp, "custom_bwd", torch.cuda.amp.custom_bwd)

        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            torch.cuda.empty_cache()

        self.model.to(self.device)

        original = decode_image(item)
        processed = remove_background_and_center(original)

        texture_resolution = int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION))
        remesh_option = item.get("remesh", REMESH_MODE)

        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mesh, _ = self.model.run_image(
                        processed,
                        bake_resolution=texture_resolution,
                        remesh=remesh_option if remesh_option != "none" else None,
                    )

            output = io.BytesIO()
            mesh.export(output, file_type="glb", include_normals=True)
            glb_bytes = output.getvalue()

            if not glb_bytes:
                raise RuntimeError("SF3D returned an empty GLB.")

            encoded = base64.b64encode(glb_bytes).decode("ascii")

            return {
                "success": True,
                "format": "glb",
                "mime_type": "model/gltf-binary",
                "model_base64": encoded,
                "texture_resolution": texture_resolution,
                "remesh": remesh_option,
            }

        except Exception:
            traceback.print_exc()
            raise
        finally:
            torch.cuda.empty_cache()


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
    async def health():
        return {"status": "ok", "service": APP_NAME, "model": MODEL_ID}

    @web_app.post("/")
    async def run_generate(request: Request):
        try:
            data = await request.json()
            model_instance = SF3DModel()
            result = await model_instance.process_image.remote.aio(data)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"SF3D generation failed: {type(exc).__name__}: {exc}",
            ) from exc

    return web_app
        
