import base64
import binascii
import io
import inspect
import os
import re
import sys
import textwrap
import threading
import traceback
from typing import Any

import modal


# ============================================================
# 1. MODAL CONFIGURATION
# ============================================================

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"

DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_IMAGE_PIXELS = 40_000_000
FOREGROUND_SCALE = 0.85

REMESH_MODE = os.environ.get("SF3D_REMESH_MODE", "quad")

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)

image = modal.Image.from_registry(
    "rhydam12/sf3d-gpu-worker:latest",
    add_python="3.10",
)


# ============================================================
# 2. IMAGE DECODING AND PREPROCESSING
# ============================================================

def decode_image(data: Any):
    """Decode base64 input safely and normalize EXIF orientation."""
    from PIL import Image, ImageOps

    if isinstance(data, dict):
        data = (
            data.get("image")
            or data.get("image_base64")
            or data.get("imageBase64")
        )

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
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc

    return result


def remove_background_and_center(image):
    """Remove the background and fit the foreground on a square canvas."""
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
    new_size = (
        max(1, int(foreground.width * scale)),
        max(1, int(foreground.height * scale)),
    )

    foreground = foreground.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )

    canvas_side = max(new_size)
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))

    position = (
        (canvas_side - foreground.width) // 2,
        (canvas_side - foreground.height) // 2,
    )
    canvas.alpha_composite(foreground, dest=position)

    return canvas


# ============================================================
# 3. FAIL-SAFE SF3D CUDA SOURCE PATCH
# ============================================================

_PATCH_LOCK = threading.Lock()
_PATCHED_METHODS = []


def patch_query_triplane():
    """
    Fail-safe patcher for SF3D query_triplane implementation.
    Catches any mismatch or error so startup never fails.
    """
    try:
        import sf3d.system as system_module

        with _PATCH_LOCK:
            for _, method_name, owner in _PATCHED_METHODS:
                if owner is system_module:
                    return

            owners = []

            for _, cls in inspect.getmembers(system_module, inspect.isclass):
                method = getattr(cls, "query_triplane", None)
                if callable(method):
                    owners.append((cls, "query_triplane", method))

            module_function = getattr(system_module, "query_triplane", None)
            if callable(module_function):
                owners.append((system_module, "query_triplane", module_function))

            if not owners:
                import sf3d
                import pkgutil
                import importlib

                for item in pkgutil.walk_packages(
                    sf3d.__path__, prefix="sf3d."
                ):
                    try:
                        mod = importlib.import_module(item.name)
                    except Exception:
                        continue

                    for _, cls in inspect.getmembers(mod, inspect.isclass):
                        method = getattr(cls, "query_triplane", None)
                        if callable(method):
                            owners.append((cls, "query_triplane", method))

            if not owners:
                print("WARNING: Could not locate query_triplane(). Skipping patch safely.")
                return

            patched_count = 0

            for owner, method_name, original in owners:
                if getattr(original, "_sf3d_cuda_patched", False):
                    patched_count += 1
                    continue

                try:
                    source = inspect.getsource(original)
                    source = textwrap.dedent(source)
                except (OSError, TypeError):
                    continue

                if "positions = positions.to(device=triplanes.device)" in source:
                    original._sf3d_cuda_patched = True
                    patched_count += 1
                    continue

                lines = source.splitlines()
                output = []
                inserted = False

                for line in lines:
                    if (
                        not inserted
                        and "grid_sample(" in line
                        and "positions" in source
                        and "triplanes" in source
                    ):
                        indent = line[:len(line) - len(line.lstrip())]
                        output.append(
                            indent
                            + "positions = positions.to(device=triplanes.device)"
                        )
                        inserted = True

                    output.append(line)

                if not inserted:
                    continue

                patched_source = "\n".join(output) + "\n"

                namespace = getattr(original, "__globals__", {})
                namespace = dict(namespace)

                try:
                    exec(
                        compile(
                            patched_source,
                            inspect.getsourcefile(original) or "<sf3d-patch>",
                            "exec",
                        ),
                        namespace,
                    )

                    replacement = namespace[method_name]
                    replacement._sf3d_cuda_patched = True

                    if inspect.ismethod(original):
                        replacement = replacement.__get__(
                            owner, owner if inspect.isclass(owner) else type(owner)
                        )

                    setattr(owner, method_name, replacement)
                    patched_count += 1

                except Exception as patch_exc:
                    print(f"WARNING: Failed to apply patch sub-routine: {patch_exc}")

            if patched_count > 0:
                _PATCHED_METHODS.append(
                    (system_module.__name__, "query_triplane", system_module)
                )
                print("Applied SF3D query_triplane device-sync patch successfully.")
            else:
                print("WARNING: query_triplane() found, but source structure did not match patch pattern. Continuing safely.")

    except Exception as exc:
        print(f"WARNING: patch_query_triplane encountered an exception, bypassing safely: {exc}")


# ============================================================
# 4. MODAL SF3D WORKER
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=900,
    max_containers=5,
)
class SF3DWorker:

    @modal.enter(snap=True)
    def load_model(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the worker.")

        self.device = torch.device(GPU_DEVICE)

        # Run fail-safe patcher
        patch_query_triplane()

        from sf3d.system import SF3D
        from rembg import new_session

        self.model = SF3D.from_pretrained(MODEL_ID)
        self.model.eval()
        self.model.to(self.device)

        remove_background_and_center._session = new_session("u2net")
        models_volume.commit()

        self._enforce_cuda()
        torch.cuda.synchronize(self.device)

    def _enforce_cuda(self, *inputs):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable after worker startup.")

        device = torch.device(GPU_DEVICE)

        if self.device != device:
            self.device = device

        self.model.to(device)

        for name, parameter in self.model.named_parameters():
            if parameter.device != device:
                parameter.data = parameter.data.to(device)

        for name, buffer in self.model.named_buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device)

        def check_input(value, name="input"):
            if torch.is_tensor(value) and value.device.type != "cuda":
                raise RuntimeError(
                    f"{name} is on {value.device}; expected CUDA."
                )

            if isinstance(value, dict):
                for key, item in value.items():
                    check_input(item, f"{name}.{key}")
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    check_input(item, f"{name}[{index}]")

        for index, value in enumerate(inputs):
            check_input(value, f"input[{index}]")

        torch.cuda.synchronize(device)

    @modal.method()
    def process_image(self, data):
        import torch

        original = decode_image(data)
        processed = remove_background_and_center(original)

        self._enforce_cuda()

        try:
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    mesh, _ = self.model.run_image(
                        processed,
                        bake_resolution=DEFAULT_TEXTURE_RESOLUTION,
                        remesh=REMESH_MODE,
                    )

            self._enforce_cuda()

            output = io.BytesIO()
            mesh.export(output, file_type="glb")
            glb_bytes = output.getvalue()

            if not glb_bytes:
                raise RuntimeError("SF3D returned an empty GLB.")

            encoded = base64.b64encode(glb_bytes).decode("ascii")

            return {
                "success": True,
                "format": "glb",
                "mime_type": "model/gltf-binary",
                "model_base64": encoded,
                "texture_resolution": DEFAULT_TEXTURE_RESOLUTION,
                "remesh": REMESH_MODE,
            }

        except Exception:
            traceback.print_exc()
            raise


# ============================================================
# 5. FASTAPI ASGI WEBHOOK (LAZILY IMPORTED)
# ============================================================

@app.function(
    image=image,
    scaledown_window=2,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    web_app = FastAPI(title="SF3D Image-to-3D API")

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    class GenerateRequest(BaseModel):
        image: str = Field(
            ...,
            description="Base64 image or data:image/...;base64 URI",
        )

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": APP_NAME,
            "model": MODEL_ID,
        }

    @web_app.post("/generate")
    async def generate(request: GenerateRequest):
        try:
            result = await SF3DWorker().process_image.remote.aio(
                {"image": request.image}
            )
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
                
