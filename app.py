import base64
import binascii
import io
import json
import os
import re
import sys
import types
import traceback
import uuid
from pathlib import Path
from typing import Any
import modal


APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"

DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_IMAGE_PIXELS = 40_000_000
# Slightly more margin so heads/feet are less clipped in SF3D
FOREGROUND_SCALE = 0.85
REMESH_MODE = "none"

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
MAX_STORED_GLB = 80 * 1024 * 1024
API_KEY = os.environ.get("MODAL_API_KEY")

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


def _get_rembg_session():
    """Prefer a stronger cutout model when available."""
    from rembg import new_session

    session = getattr(_get_rembg_session, "_session", None)
    if session is not None:
        return session

    # isnet-general-use usually gives cleaner edges than u2net
    for model_name in ("isnet-general-use", "u2net"):
        try:
            session = new_session(model_name)
            _get_rembg_session._session = session
            print(f"rembg session loaded: {model_name}")
            return session
        except Exception as exc:
            print(f"rembg model {model_name} unavailable: {exc}")

    session = new_session("u2net")
    _get_rembg_session._session = session
    return session


def _sharpen_alpha(rgba, threshold=24):
    """Harder alpha edges reduce blocky fringe artifacts in SF3D."""
    from PIL import Image
    r, g, b, a = rgba.split()
    # Binary-ish mask: kill soft fringe pixels
    a = a.point(lambda p: 255 if p >= threshold else 0)
    return Image.merge("RGBA", (r, g, b, a))


def remove_background_and_center(image):
    from PIL import Image
    from rembg import remove

    session = _get_rembg_session()
    rgba = remove(image, session=session).convert("RGBA")
    rgba = _sharpen_alpha(rgba)

    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("No foreground subject was detected.")

    subject = rgba.crop(bbox)
    canvas_size = 1280
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    max_dim = int(canvas_size * FOREGROUND_SCALE)
    subject.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    paste_x = (canvas_size - subject.width) // 2
    paste_y = (canvas_size - subject.height) // 2
    canvas.paste(subject, (paste_x, paste_y), subject)
    return canvas


def job_status_path(job_id: str) -> Path:
    return Path(CACHE_DIR) / "jobs" / f"{job_id}.json"


def write_job(job_id: str, payload: dict):
    path = job_status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    models_volume.commit()


def read_job(job_id: str):
    path = job_status_path(job_id)
    try:
        models_volume.reload()
    except Exception:
        pass
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@app.cls(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=1,  # Starter: one GPU job at a time avoids stuck pending queue
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

        network_path = "/app/stable-fast-3d/sf3d/models/network.py"
        if os.path.exists(network_path):
            with open(network_path, "r") as f:
                code = f.read()
            code = code.replace("from torch.amp import custom_bwd, custom_fwd", "from torch.cuda.amp import custom_bwd, custom_fwd")
            code = code.replace('device_type="cuda"', '')
            code = code.replace("device_type='cuda'", '')
            with open(network_path, "w") as f:
                f.write(code)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)

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

        self.model = SF3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.model.eval()
        self.model.to(self.device)

        # Warm rembg session inside the snapshot
        _get_rembg_session()
        models_volume.commit()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        print("Snapshot model initialization complete.")

    @modal.method()
    def process_image(self, item: dict):
        import torch
        import torch.cuda.amp

        job_id = item.get("job_id")
        if job_id:
            write_job(job_id, {"status": "running", "job_id": job_id})

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

        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            torch.cuda.empty_cache()

        self.model.to(self.device)

        try:
            original = decode_image(item)
            processed = remove_background_and_center(original)

            texture_resolution = int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION))
            texture_resolution = max(512, min(texture_resolution, 2048))
            remesh_option = item.get("remesh", REMESH_MODE)

            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    mesh, _ = self.model.run_image(
                        processed,
                        bake_resolution=texture_resolution,
                        remesh=remesh_option if remesh_option != "none" else None,
                    )

            glb_bytes = mesh.export(file_type="glb")
            if not glb_bytes:
                raise RuntimeError("SF3D returned an empty GLB.")

            asset_id = str(uuid.uuid4())
            asset_path = Path(CACHE_DIR) / "outputs" / f"{asset_id}.glb"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(glb_bytes)
            models_volume.commit()

            result = {
                "success": True,
                "status": "done",
                "format": "glb",
                "mime_type": "model/gltf-binary",
                "asset_id": asset_id,
                "download_url": f"/download/{asset_id}",
                "texture_resolution": texture_resolution,
                "remesh": remesh_option,
                "job_id": job_id,
            }

            if item.get("return_base64", True):
                result["model"] = base64.b64encode(glb_bytes).decode("utf-8")

            if job_id:
                status = {
                    "status": "done",
                    "job_id": job_id,
                    "asset_id": asset_id,
                    "download_url": f"/download/{asset_id}",
                }
                if item.get("return_base64", True) and len(result.get("model", "")) < 16_000_000:
                    status["model"] = result["model"]
                write_job(job_id, status)

            return result

        except Exception as exc:
            traceback.print_exc()
            if job_id:
                write_job(job_id, {
                    "status": "error",
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise
        finally:
            torch.cuda.empty_cache()


@app.function(
    image=image,
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    timeout=60,
)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request, Header
    from fastapi.responses import Response
    from fastapi.middleware.cors import CORSMiddleware
    import time

    web_app = FastAPI(title="SF3D Image-to-3D API")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
    )

    def reload_models():
        try:
            models_volume.reload()
        except Exception:
            traceback.print_exc()

    def glb_path(asset_id: str) -> Path:
        return Path(CACHE_DIR) / "outputs" / f"{asset_id}.glb"

    def check_api_key(x_api_key: str | None):
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @web_app.get("/")
    async def health():
        return {"status": "ok", "service": APP_NAME, "model": MODEL_ID}

    @web_app.get("/job/{job_id}")
    async def get_job(job_id: str):
        if not UUID_RE.fullmatch(job_id):
            raise HTTPException(status_code=400, detail="Bad job id")
        for _ in range(3):
            data = read_job(job_id)
            if data:
                return data
            time.sleep(0.4)
        raise HTTPException(status_code=404, detail="Job not found")

    @web_app.api_route("/download/{asset_id}", methods=["GET", "HEAD"])
    async def download_asset(asset_id: str):
        if not UUID_RE.fullmatch(asset_id):
            raise HTTPException(status_code=400, detail="Bad asset id")
        path = glb_path(asset_id)
        for _ in range(5):
            reload_models()
            if path.is_file():
                break
            time.sleep(0.8)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found or expired")
        filename = f"form3d-{asset_id}.glb"
        data = path.read_bytes()
        return Response(
            content=data,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "private, max-age=86400",
                "X-Content-Type-Options": "nosniff",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @web_app.post("/")
    async def run_generate(request: Request, x_api_key: str | None = Header(None)):
        check_api_key(x_api_key)
        try:
            data = await request.json()
            data.setdefault("texture_resolution", DEFAULT_TEXTURE_RESOLUTION)
            data.setdefault("remesh", REMESH_MODE)
            data.setdefault("return_base64", True)

            async_mode = data.get("async", True)
            job_id = str(uuid.uuid4())
            data["job_id"] = job_id

            write_job(job_id, {"status": "queued", "job_id": job_id})

            model_instance = SF3DModel()
            call = model_instance.process_image.spawn(data)

            if not async_mode:
                result = call.get()
                return result

            return {
                "success": True,
                "status": "queued",
                "job_id": job_id,
                "poll_url": f"/job/{job_id}",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"SF3D generation failed: {type(exc).__name__}: {exc}",
            ) from exc

    return web_app
