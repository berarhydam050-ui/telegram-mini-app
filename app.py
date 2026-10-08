import base64
import io
import os

import modal


# ---------------------------------------------------------
# Modal app
# ---------------------------------------------------------

app = modal.App("sf3d-backend")


# ---------------------------------------------------------
# Persistent Hugging Face cache
# ---------------------------------------------------------

hf_cache = modal.Volume.from_name(
    "sf3d-weights-cache",
    create_if_missing=True,
)

HF_CACHE_DIR = "/root/.cache/huggingface"


# ---------------------------------------------------------
# SF3D container image
#
# We use a CUDA DEVELOPMENT image because SF3D's
# texture_baker is a CUDA extension and must be compiled.
# ---------------------------------------------------------

sf3d_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "wget",
        "unzip",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "setuptools==69.5.1",
        "wheel",
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone https://github.com/Stability-AI/stable-fast-3d.git /app/stable-fast-3d",
        "cd /app/stable-fast-3d && pip install -r requirements.txt --no-build-isolation",
    )
    .pip_install(
        "fastapi",
        "uvicorn",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{HF_CACHE_DIR}/transformers",
            "TORCH_HOME": "/root/.cache/torch",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


# ---------------------------------------------------------
# SF3D GPU model
# ---------------------------------------------------------

@app.cls(
    image=sf3d_image,
    gpu="A10G",
    volumes={
        HF_CACHE_DIR: hf_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=300,
    scaledown_window=60,
)
class SF3DModel:

    @modal.enter()
    def load_model(self):
        import torch

        # Make sure the repository is importable.
        import sys

        repo_path = "/app/stable-fast-3d"

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        print("======================================")
        print("Starting SF3D")
        print("======================================")

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available inside the Modal container."
            )

        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

        # -------------------------------------------------
        # IMPORTANT:
        # Official SF3D API
        # -------------------------------------------------

        from sf3d.system import SF3D

        print("Loading Stable Fast 3D model...")

        self.model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        self.model.to("cuda")
        self.model.eval()

        print("======================================")
        print("SF3D MODEL LOADED SUCCESSFULLY")
        print("======================================")

        # Save downloaded Hugging Face files to the Modal Volume.
        hf_cache.commit()


    @modal.method()
    def generate_mesh(
        self,
        image_base64: str,
        texture_resolution: int = 1024,
        remesh_option: str = "none",
    ):

        import torch
        from PIL import Image
        import rembg

        print("Received image")

        # -------------------------------------------------
        # Remove data URL prefix
        # -------------------------------------------------

        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        # -------------------------------------------------
        # Decode image
        # -------------------------------------------------

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            raise ValueError(
                f"Invalid base64 image: {e}"
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGBA")

        print(
            "Input image:",
            image.size,
            image.mode,
        )

        # -------------------------------------------------
        # Background removal
        # -------------------------------------------------

        print("Removing background...")

        rembg_session = rembg.new_session()

        image = rembg.remove(
            image,
            session=rembg_session,
        )

        image = image.convert("RGBA")

        print("Background removed")

        # -------------------------------------------------
        # SF3D inference
        # -------------------------------------------------

        print("Running SF3D inference...")

        texture_resolution = int(texture_resolution)

        if texture_resolution not in (
            512,
            1024,
            2048,
        ):
            texture_resolution = 1024

        if remesh_option not in (
            "none",
            "triangle",
            "quad",
        ):
            remesh_option = "none"

        with torch.inference_mode():

            mesh, global_dict = self.model.run_image(
                image,
                bake_resolution=texture_resolution,
                remesh=remesh_option,
                vertex_count=-1,
            )

        print("SF3D inference completed")

        # -------------------------------------------------
        # Validate mesh
        # -------------------------------------------------

        if mesh is None:
            raise RuntimeError(
                "SF3D returned no mesh."
            )

        if len(mesh.vertices) == 0:
            raise RuntimeError(
                "SF3D generated an empty mesh."
            )

        print(
            "Vertices:",
            len(mesh.vertices),
        )

        print(
            "Faces:",
            len(mesh.faces),
        )

        # -------------------------------------------------
        # Export GLB
        # -------------------------------------------------

        output_buffer = io.BytesIO()

        mesh.export(
            output_buffer,
            file_type="glb",
            include_normals=True,
        )

        output_buffer.seek(0)

        glb_bytes = output_buffer.read()

        print(
            "GLB size:",
            len(glb_bytes),
            "bytes",
        )

        # -------------------------------------------------
        # Return base64 GLB
        # -------------------------------------------------

        return base64.b64encode(
            glb_bytes
        ).decode("utf-8")


# ---------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------

@app.function(
    image=sf3d_image,
    timeout=300,
)
@modal.asgi_app()
def api():

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(
        title="SF3D Backend",
        version="1.0",
    )

    # -----------------------------------------------------
    # CORS
    # -----------------------------------------------------

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------
    # Health endpoint
    # -----------------------------------------------------

    @web_app.get("/")
    async def health():

        return {
            "status": "ok",
            "service": "sf3d-backend",
        }

    # -----------------------------------------------------
    # Generate endpoint
    # -----------------------------------------------------

    @web_app.post("/generate")
    async def generate_endpoint(data: dict):

        try:

            image = data.get("image")

            if not image:
                return {
                    "status": "error",
                    "message": "Missing image",
                }

            texture_resolution = data.get(
                "texture_resolution",
                1024,
            )

            remesh_option = data.get(
                "remesh_option",
                "none",
            )

            print("Creating SF3D model instance...")

            model = SF3DModel()

            print("Sending generation request to GPU...")

            result = await model.generate_mesh.remote.aio(
                image,
                int(texture_resolution),
                str(remesh_option),
            )

            print("Generation completed")

            return {
                "status": "success",
                "model": result,
            }

        except Exception as e:

            import traceback

            traceback.print_exc()

            return {
                "status": "error",
                "message": str(e),
        }
