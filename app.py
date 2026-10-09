
import base64
import binascii
import io
import os
import re
import sys
import traceback

import modal


# ---------------------------------------------------------
# 1. MODAL CONFIGURATION
# ---------------------------------------------------------

IMAGE_NAME = "rhydam12/sf3d-gpu-worker:latest"
REPO_DIR = "/app/stable-fast-3d"
CACHE_DIR = "/root/.cache"

image = modal.Image.from_registry(
    IMAGE_NAME,
    add_python="3.10",
)

app = modal.App("sf3d-backend")

models_volume = modal.Volume.from_name(
    "sf3d-models-volume",
    create_if_missing=True,
)


# ---------------------------------------------------------
# 2. PATCH THE SF3D DEVICE MISMATCH
# ---------------------------------------------------------

def patch_sf3d_source():
    """
    Patch query_triplane() before importing SF3D.

    Positions must be on the same device as triplanes before
    grid_sample() receives the constructed sampling grid.
    """
    system_path = os.path.join(
        REPO_DIR,
        "sf3d",
        "system.py",
    )

    if not os.path.isfile(system_path):
        raise FileNotFoundError(
            f"SF3D system.py was not found: {system_path}"
        )

    with open(system_path, "r", encoding="utf-8") as file:
        source = file.read()

    # Avoid applying the patch repeatedly.
    marker = "# Modal device-sync fix"
    if marker in source:
        print("SF3D device-sync patch already present.")
        return

    # Restrict the patch to query_triplane().
    function_match = re.search(
        r"(?m)^([ \t]*)def query_triplane\(",
        source,
    )

    if not function_match:
        raise RuntimeError(
            "Could not find query_triplane() in system.py. "
            "Inspect the installed SF3D source before deploying."
        )

    start = function_match.start()
    next_function = re.search(
        r"(?m)^([ \t]*)def ",
        source[function_match.end():],
    )

    end = (
        function_match.end() + next_function.start()
        if next_function
        else len(source)
    )

    function_source = source[start:end]

    # Handle both assertion-based and direct scale_tensor layouts.
    pattern = re.compile(
        r"(?m)^([ \t]*)positions\s*=\s*scale_tensor\("
    )

    if not pattern.search(function_source):
        raise RuntimeError(
            "The query_triplane() source layout differs from "
            "the expected version. No patch was applied."
        )

    function_source = pattern.sub(
        lambda match: (
            match.group(1)
            + marker
            + "\n"
            + match.group(1)
            + "positions = positions.to(device=triplanes.device)\n"
            + match.group(1)
            + "positions = scale_tensor("
        ),
        function_source,
        count=1,
    )

    source = (
        source[:start]
        + function_source
        + source[end:]
    )

    with open(system_path, "w", encoding="utf-8") as file:
        file.write(source)

    print("Applied SF3D query_triplane device-sync patch.")


# ---------------------------------------------------------
# 3. SF3D MODEL WORKER
# ---------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=900,
    startup_timeout=900,
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={
        "enable_gpu_snapshot": True,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={
        CACHE_DIR: models_volume,
    },
)
class SF3DModel:

    @modal.enter(snap=True)
    def load_model(self):
        import torch
        from huggingface_hub import login
        from rembg import new_session

        print("Initializing SF3D worker.")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check the Modal GPU configuration."
            )

        # Apply the patch before importing SF3D.
        sys.path.insert(0, REPO_DIR)
        patch_sf3d_source()

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

        if not token:
            raise RuntimeError(
                "HF_TOKEN is missing. Add it to the "
                "'huggingface-secret' Modal secret."
            )

        login(token=token)

        # Load the background-removal model once per worker.
        # This also populates its cache under /root/.cache.
        print("Initializing background-removal session.")
        self.rembg_session = new_session("u2net")

        from sf3d.system import SF3D

        print("Loading Stability AI SF3D model.")
        self.device = torch.device("cuda:0")

        self.model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        self.model.to(self.device)
        self.model.eval()

        torch.cuda.synchronize()

        # Commit cached downloads to persistent storage.
        models_volume.commit()

        print("SF3D model initialization complete.")

    # -----------------------------------------------------
    # 4. REVALIDATE DEVICE PLACEMENT BEFORE EACH REQUEST
    # -----------------------------------------------------

    def _enforce_cuda(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable during inference."
            )

        self.device = torch.device("cuda:0")

        # Move all registered parameters and buffers.
        self.model.to(self.device)

        # Explicitly repair misplaced parameter tensors.
        for name, parameter in self.model.named_parameters():
            if parameter.device != self.device:
                parameter.data = parameter.data.to(self.device)

            if parameter.device != self.device:
                raise RuntimeError(
                    f"Parameter {name} is still on "
                    f"{parameter.device}."
                )

        # Explicitly repair misplaced buffers.
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                buffer.data = buffer.data.to(self.device)

            if buffer.device != self.device:
                raise RuntimeError(
                    f"Buffer {name} is still on {buffer.device}."
                )

        torch.cuda.synchronize()

    # -----------------------------------------------------
    # 5. IMAGE DECODING AND VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _decode_image(item):
        from PIL import Image, ImageOps

        if not isinstance(item, dict):
            raise ValueError("The request body must be a JSON object.")

        encoded = item.get("image")

        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError(
                "The 'image' field must contain a base64 image."
            )

        encoded = encoded.strip()

        # Support data:image/png;base64,... and similar headers.
        encoded = re.sub(
            r"^data:image/[^;]+;base64,",
            "",
            encoded,
            flags=re.IGNORECASE,
        )

        encoded = re.sub(r"\s+", "", encoded)

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "The image field contains invalid base64."
            ) from exc

        if not raw:
            raise ValueError("The decoded image is empty.")

        try:
            with Image.open(io.BytesIO(raw)) as source:
                source = ImageOps.exif_transpose(source)
                source.load()
                result = source.convert("RGBA")
        except Exception as exc:
            raise ValueError(
                "The decoded content is not a valid image."
            ) from exc

        # Basic request-size safeguard.
        if result.width * result.height > 40_000_000:
            raise ValueError(
                "The image is too large. Maximum: 40 megapixels."
            )

        return result

    # -----------------------------------------------------
    # 6. IMAGE-TO-3D INFERENCE
    # -----------------------------------------------------

    @modal.method()
    def process_image(self, item: dict):
        import torch
        from sf3d.utils import (
            remove_background,
            resize_foreground,
        )

        # Validate request and image.
        image = self._decode_image(item)

        resolution = item.get("texture_resolution", 2048)

        if (
            isinstance(resolution, bool)
            or not isinstance(resolution, int)
            or resolution not in (512, 1024, 2048)
        ):
            raise ValueError(
                "'texture_resolution' must be 512, 1024, or 2048."
            )

        remesh_option = item.get("remesh", "quad")

        if remesh_option not in ("none", "triangle", "quad"):
            raise ValueError(
                "'remesh' must be 'none', 'triangle', or 'quad'."
            )

        remesh = (
            None if remesh_option == "none"
            else remesh_option
        )

        # Important after a snapshot restore.
        self._enforce_cuda()

        # Remove background and center the foreground.
        image = remove_background(
            image,
            self.rembg_session,
        )

        bbox = image.getbbox()

        if bbox is None:
            raise ValueError(
                "Background removal returned an empty image."
            )

        image = image.crop(bbox)
        image = resize_foreground(image, 0.85)

        print(
            f"Starting SF3D inference: "
            f"resolution={resolution}, remesh={remesh_option}"
        )

        try:
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    mesh, glob_dict = self.model.run_image(
                        image,
                        bake_resolution=resolution,
                        remesh=remesh,
                    )

            torch.cuda.synchronize()

        except Exception:
            print("SF3D inference traceback:")
            traceback.print_exc()
            raise

        if mesh is None:
            raise RuntimeError("SF3D returned no mesh.")

        if len(mesh.vertices) == 0:
            raise RuntimeError("SF3D returned an empty mesh.")

        # Export the mesh with textures to GLB.
        output_path = "/tmp/output.glb"

        try:
            mesh.export(
                output_path,
                file_type="glb",
                include_normals=True,
            )

            with open(output_path, "rb") as file:
                glb_bytes = file.read()

            if not glb_bytes:
                raise RuntimeError("The exported GLB file is empty.")

            return {
                "model": base64.b64encode(
                    glb_bytes
                ).decode("ascii")
            }

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# ---------------------------------------------------------
# 7. FASTAPI WEBHOOK
# ---------------------------------------------------------

@app.function(
    image=image,
    timeout=900,
)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(
        title="SF3D Backend",
        version="1.0.0",
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.get("/")
    async def health():
        return {
            "status": "ok",
            "service": "sf3d-backend",
        }

    @web_app.post("/")
    async def run_generate(request: Request):
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Request body must be valid JSON.",
            ) from exc

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Request JSON must be an object.",
            )

        try:
            model_instance = SF3DModel()

            result = await model_instance.process_image.remote.aio(
                data
            )

            return result

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            print("SF3D request failed:")
            traceback.print_exc()

            raise HTTPException(
                status_code=500,
                detail=(
                    "3D generation failed. Open the Modal "
                    "SF3DModel function logs for the traceback."
                ),
            ) from exc

    return web_app
