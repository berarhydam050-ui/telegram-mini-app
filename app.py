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

# Official Stable Fast 3D default. The object fills this fraction
# of a square frame; the rest is transparent padding.
DEFAULT_FOREGROUND_RATIO = 0.85
DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_INPUT_SIDE = 2048

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)

# ============================================================
# IMAGE DEFINITION
# Forces compilation of the custom CUDA extension (texture_baker)
# with nvcc during image build so the CUDA kernel is available
# after snapshot restore on A10G.
# ============================================================

image = (
    modal.Image.from_registry(
        "rhydam12/sf3d-gpu-worker:latest",
        add_python="3.10",
    )
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchvision",
        "rembg",
        "Pillow",
        "numpy",
        "trimesh",
        "ninja",  # speeds up / stabilizes CUDA extension builds
    )
    .run_commands(
        # Fail early if nvcc is missing
        "which nvcc || (echo 'ERROR: nvcc not found in base image' && exit 1)",

        # Force rebuild of stable_fast_3d + its CUDA/C++ extensions
        f"""
        if [ -d {SF3D_REPO} ]; then
            cd {SF3D_REPO}
            echo "=== Building stable_fast_3d with CUDA support ==="
            FORCE_CUDA=1 \\
            TORCH_CUDA_ARCH_LIST="8.6" \\
            pip install --no-build-isolation -e . -v
        else
            echo "ERROR: {SF3D_REPO} not found" && exit 1
        fi
        """,

        # Sanity check that the CUDA operator actually exists
        """
        python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
try:
    from texture_baker import texture_baker_cpp
    print('texture_baker_cpp imported successfully')
    print('Has rasterize:', hasattr(texture_baker_cpp, 'rasterize'))
except Exception as e:
    print('Failed to import texture_baker_cpp:', e)
    raise
"
        """,
    )
)

# Lightweight image for the HTTP layer only
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]"
)


# ============================================================
# PYTORCH AMP COMPATIBILITY PATCH
# Must run before importing SF3D.
# ============================================================

def patch_legacy_cuda_amp(torch):
    import torch.amp as torch_amp

    cuda_amp = torch.cuda.amp

    def make_wrapper(original_decorator):
        @functools.wraps(original_decorator)
        def wrapper(*args, **kwargs):
            kwargs.pop("device_type", None)
            return original_decorator(*args, **kwargs)

        wrapper._sf3d_compat_patched = True
        return wrapper

    for name in ("custom_fwd", "custom_bwd"):
        original = getattr(cuda_amp, name)
        if not getattr(original, "_sf3d_compat_patched", False):
            original = make_wrapper(original)
            setattr(cuda_amp, name, original)

        # PyTorch 2.3 only exposes these on torch.cuda.amp.
        # This image's SF3D does: from torch.amp import custom_fwd, custom_bwd
        if getattr(torch_amp, name, None) is None:
            setattr(torch_amp, name, getattr(cuda_amp, name))

    print("[SF3D] AMP compatibility patch applied.")
    print(f"[SF3D] PyTorch: {torch.__version__}")


# ============================================================
# IMAGE PREPROCESSING
# Matches Stability-AI/stable-fast-3d sf3d/utils.py.
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
        image = ImageOps.exif_transpose(source).convert("RGBA")

    longest = max(image.width, image.height)
    if longest > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / longest
        image = image.resize(
            (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    return image


def remove_background(image, session, force=None):
    """Official SF3D behavior: skip rembg when the image is already cut out."""
    from rembg import remove

    if force is False:
        return image

    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    if force is True:
        do_remove = True

    if do_remove:
        image = remove(image, session=session).convert("RGBA")

    return image


def resize_foreground(image, ratio: float):
    """Center the object so it occupies `ratio` of a square transparent frame.

    PIL crop pads with transparent pixels when the window extends outside
    the source, which is what torchvision's crop does in the official repo.
    """
    import numpy as np
    from PIL import Image

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    mask = np.array(image)[:, :, -1]
    foreground = mask > 0.5
    if not foreground.any():
        raise ValueError("No foreground object detected.")

    rows = foreground.any(axis=1)
    cols = foreground.any(axis=0)
    y1 = int(rows.argmax())
    y2 = int(len(rows) - 1 - rows[::-1].argmax())
    x1 = int(cols.argmax())
    x2 = int(len(cols) - 1 - cols[::-1].argmax())

    height = y2 - y1
    width = x2 - x1
    if height <= 0 or width <= 0:
        raise ValueError("No foreground object detected.")

    y_center = (y1 + y2) / 2
    x_center = (x1 + x2) / 2
    side = max(height, width) / ratio

    top = int(y_center - side / 2)
    left = int(x_center - side / 2)
    extent = int(side)

    framed = image.crop((left, top, left + extent, top + extent))

    # Keep the bake source sharp without handing SF3D a tiny or huge frame.
    longest = max(framed.width, framed.height)
    if longest > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / longest
        framed = framed.resize(
            (
                max(1, int(framed.width * scale)),
                max(1, int(framed.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    elif longest < 512:
        scale = 512 / longest
        framed = framed.resize(
            (
                max(1, int(framed.width * scale)),
                max(1, int(framed.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    return framed


def prepare_input(item: Any, session):
    image = decode_image(item)

    if isinstance(item, dict) and "foreground_ratio" in item:
        ratio = float(item["foreground_ratio"])
        if not 0.5 <= ratio <= 0.95:
            raise ValueError("foreground_ratio must be between 0.5 and 0.95.")
    else:
        ratio = DEFAULT_FOREGROUND_RATIO

    force = item.get("remove_background") if isinstance(item, dict) else None
    if force is not None and not isinstance(force, bool):
        raise ValueError("remove_background must be true or false.")

    image = remove_background(image, session, force=force)
    image = resize_foreground(image, ratio)
    return image, ratio


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
    # One mesh, then the container is destroyed. No warm A10G.
    single_use_containers=True,
    min_containers=0,
    scaledown_window=2,
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
        from rembg import new_session

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

        # Runtime check that the CUDA texture baker is present
        try:
            from texture_baker import texture_baker_cpp
            if not hasattr(texture_baker_cpp, "rasterize"):
                raise RuntimeError(
                    "texture_baker_cpp.rasterize not found – CUDA extension missing"
                )
            print("[SF3D] texture_baker CUDA extension OK")
        except Exception as e:
            raise RuntimeError(
                f"texture_baker CUDA extension failed to load: {e}"
            ) from e

        # Import only after applying the AMP compatibility patch.
        from stable_fast_3d.sf3d.system import SF3D

        # Captured in the snapshot so later cold starts skip this load.
        print("[SF3D] Loading rembg session...")
        self.rembg_session = new_session("u2net")

        print("[SF3D] Loading model on A10G...")

        self.pipeline = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        self.pipeline.to("cuda:0")
        self.pipeline.eval()
        torch.cuda.synchronize()

        print("[SF3D] Model loaded; ready for snapshot.")

    @modal.enter()
    def wake(self):
        # Runs after every snapshot restore, not during capture.
        import torch

        torch.cuda.synchronize()
        print("[SF3D] Restored from snapshot.")

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

            img, ratio = prepare_input(item, self.rembg_session)

            texture_res = (
                int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION))
                if isinstance(item, dict)
                else DEFAULT_TEXTURE_RESOLUTION
            )

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

            if remesh_val not in ("none", "triangle", "quad"):
                raise ValueError(
                    "remesh must be none, triangle, or quad."
                )

            vertex_count = (
                int(item.get("vertex_count", -1))
                if isinstance(item, dict)
                else -1
            )
            if vertex_count == 0 or vertex_count < -1:
                raise ValueError(
                    "vertex_count must be -1 or a positive integer."
                )

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
                    vertex_count=vertex_count,
                )

            if mesh.vertices.shape[0] == 0:
                raise ValueError("No subject detected in the image.")

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

            print(
                f"[SF3D] Generation completed in {elapsed}s "
                f"(texture={texture_res}, ratio={ratio})"
            )

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
    image=web_image,
    scaledown_window=2,
    min_containers=0,
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
