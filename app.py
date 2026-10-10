
import base64
import functools
import io
import os
import sys
import traceback
from typing import Any

import modal


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"
SF3D_REPO = "/app/stable_fast_3d"

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "rhydam12/sf3d-gpu-worker:latest",
        add_python="3.10",
    )
    .pip_install("fastapi[standard]")
)


# ============================================================
# PYTORCH AMP COMPATIBILITY PATCH
# Must run before importing SF3D.
# ============================================================

def patch_legacy_cuda_amp(torch):
    amp = torch.cuda.amp

    for name in ("custom_fwd", "custom_bwd"):
        original = getattr(amp, name)

        if getattr(original, "_sf3d_compat_patched", False):
            continue

        def make_wrapper(original_decorator):
            @functools.wraps(original_decorator)
            def wrapper(*args, **kwargs):
                kwargs.pop("device_type", None)
                return original_decorator(*args, **kwargs)

            wrapper._sf3d_compat_patched = True
            return wrapper

        setattr(amp, name, make_wrapper(original))

    print("[SF3D] AMP compatibility patch applied.")
    print(f"[SF3D] PyTorch: {torch.__version__}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def decode_image(data: Any):
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

    data = data.strip()

    if data.startswith("data:") and "," in data[:200]:
        data = data.split(",", 1)[1]

    raw = base64.b64decode(data, validate=False)

    if not raw:
        raise ValueError("Decoded image buffer is empty.")

    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGBA")


def remove_background_and_center(image):
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(
        remove_background_and_center,
        "_session",
        None,
    )

    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    rgba = remove(image, session=session).convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()

    if bbox is None:
        raise ValueError("No foreground object detected.")

    foreground = rgba.crop(bbox)

    # Scale the foreground to leave a little margin.
    longest_side = max(foreground.width, foreground.height)
    target_side = max(1, int(longest_side * 0.85))
    scale = target_side / longest_side

    new_size = (
        max(1, int(foreground.width * scale)),
        max(1, int(foreground.height * scale)),
    )

    foreground = foreground.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )

    canvas_side = max(new_size)

    canvas = Image.new(
        "RGBA",
        (canvas_side, canvas_side),
        (0, 0, 0, 0),
    )

    position = (
        (canvas_side - foreground.width) // 2,
        (canvas_side - foreground.height) // 2,
    )

    canvas.alpha_composite(foreground, dest=position)
    return canvas


# ============================================================
# SINGLE-USE GPU WORKER WITH MEMORY SNAPSHOTS
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=900,
    startup_timeout=900,

    # Request shutdown after each GPU job.
    single_use_containers=True,
    scaledown_window=0,

    # Snapshot model initialization for faster cold starts.
    enable_memory_snapshot=True,
    experimental_options={
        "enable_gpu_snapshot": True,
    },
)
class SF3DModel:

    @modal.enter(snap=True)
    def load_model(self):
        import torch
        import types

        os.environ["HF_HOME"] = CACHE_DIR

        # Patch AMP before importing any SF3D code.
        patch_legacy_cuda_amp(torch)

        # Preserve compatibility with installations that import
        # optional ComfyUI modules during SF3D initialization.
        sys.modules.setdefault(
            "comfy",
            types.ModuleType("comfy"),
        )
        sys.modules.setdefault(
            "comfy.model_management",
            types.ModuleType("comfy.model_management"),
        )
        sys.modules.setdefault(
            "folder_paths",
            types.ModuleType("folder_paths"),
        )

        if SF3D_REPO not in sys.path:
            sys.path.insert(0, SF3D_REPO)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable on this Modal worker."
            )

        # Import only after applying the AMP compatibility patch.
        from stable_fast_3d.sf3d.system import SF3D

        print("[SF3D] Loading model on A10G...")

        self.pipeline = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        self.pipeline.to("cuda:0")
        self.pipeline.eval()

        print("[SF3D] Model loaded; ready for snapshot.")

    @modal.method()
    def process_mesh(self, item: Any) -> dict:
        import torch
        import time

        started = time.perf_counter()

        try:
            if not isinstance(item, (dict, str)):
                raise ValueError(
                    "Request must be an object or base64 image string."
                )

            img = remove_background_and_center(
                decode_image(item)
            )

            texture_res = (
                int(item.get("texture_resolution", 2048))
                if isinstance(item, dict)
                else 2048
            )

            # Prevent unreasonable texture-resolution requests.
            if texture_res not in (512, 1024, 2048, 4096):
                raise ValueError(
                    "texture_resolution must be 512, 1024, "
                    "2048, or 4096."
                )

            remesh_val = (
                item.get("remesh", "none")
                if isinstance(item, dict)
                else "none"
            )

            if remesh_val == "none":
                remesh_val = None

            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ),
            ):
                mesh, _ = self.pipeline.run_image(
                    img,
                    bake_resolution=texture_res,
                    remesh=remesh_val,
                )

            output = io.BytesIO()

            mesh.export(
                output,
                file_type="glb",
                include_normals=True,
            )

            encoded = base64.b64encode(
                output.getvalue()
            ).decode("ascii")

            elapsed = round(
                time.perf_counter() - started,
                3,
            )

            print(f"[SF3D] Generation completed in {elapsed}s")

            return {
                "success": True,
                "model": encoded,
                "model_base64": encoded,
                "glb": encoded,
                "format": "glb",
                "generation_seconds": elapsed,
            }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================
# HTTP API
# ============================================================

@app.function(
    image=image,
    scaledown_window=0,
    timeout=900,
    startup_timeout=900,
)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": APP_NAME,
        }

    @web_app.post("/")
    @web_app.post("/generate")
    async def run_generate(request: Request):
        try:
            data = await request.json()

            result = await SF3DModel().process_mesh.remote.aio(
                data
            )

            return result

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            traceback.print_exc()

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    return web_app
