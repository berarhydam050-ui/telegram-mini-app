import base64
import io
import os
import sys

import modal


# =========================================================
# MODAL APP
# =========================================================

app = modal.App("sf3d-backend")


# =========================================================
# HUGGING FACE CACHE VOLUME
# =========================================================

hf_cache = modal.Volume.from_name(
    "sf3d-weights-cache",
    create_if_missing=True,
)

HF_CACHE_DIR = "/root/.cache/huggingface"


# =========================================================
# SF3D IMAGE
# =========================================================

sf3d_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )

    # -----------------------------------------------------
    # System packages
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Install setuptools/wheel from normal PyPI
    # IMPORTANT:
    # Do NOT put these in the CUDA PyTorch pip install.
    # -----------------------------------------------------

    .pip_install(
        "setuptools==69.5.1",
        "wheel",
    )

    # -----------------------------------------------------
    # Install CUDA PyTorch separately
    # -----------------------------------------------------

    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )

    # -----------------------------------------------------
    # Clone official Stable Fast 3D repository
    # -----------------------------------------------------

    .run_commands(
        "git clone https://github.com/Stability-AI/stable-fast-3d.git /app/stable-fast-3d",

        "cd /app/stable-fast-3d && pip install -r requirements.txt --no-build-isolation",
    )

    # -----------------------------------------------------
    # Web API dependencies
    # -----------------------------------------------------

    .pip_install(
        "fastapi",
        "uvicorn",
    )

    # -----------------------------------------------------
    # Environment variables
    # -----------------------------------------------------

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


# =========================================================
# SF3D MODEL
# =========================================================

@app.cls(
    image=sf3d_image,

    # NVIDIA A10G GPU
    gpu="A10G",

    # Persistent Hugging Face cache
    volumes={
        HF_CACHE_DIR: hf_cache,
    },

    # Hugging Face token
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],

    # Maximum request/container time
    timeout=300,

    # Shut down idle GPU after 60 seconds
    scaledown_window=60,
)
class SF3DModel:

    # =====================================================
    # LOAD MODEL WHEN CONTAINER STARTS
    # =====================================================

    @modal.enter()
    def load_model(self):

        import torch

        # -------------------------------------------------
        # Add official SF3D repository to Python path
        # -------------------------------------------------

        repo_path = "/app/stable-fast-3d"

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        print("==========================================")
        print("Starting Stable Fast 3D")
        print("==========================================")

        print(
            "PyTorch:",
            torch.__version__,
        )

        print(
            "CUDA available:",
            torch.cuda.is_available(),
        )

        # -------------------------------------------------
        # Make sure GPU is available
        # -------------------------------------------------

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is not available inside the Modal container."
            )

        print(
            "CUDA:",
            torch.version.cuda,
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Official SF3D import
        # -------------------------------------------------

        from sf3d.system import SF3D

        print("Importing SF3D successfully.")

        # -------------------------------------------------
        # Load official Stable Fast 3D model
        # -------------------------------------------------

        print("Loading Stable Fast 3D model...")

        self.model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        # -------------------------------------------------
        # Move model to GPU
        # -------------------------------------------------

        self.model.to("cuda")

        self.model.eval()

        print("==========================================")
        print("SF3D MODEL LOADED SUCCESSFULLY")
        print("==========================================")

        # -------------------------------------------------
        # Save Hugging Face cache
        # -------------------------------------------------

        hf_cache.commit()


    # =====================================================
    # IMAGE → 3D
    # =====================================================

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

        print("==========================================")
        print("Received image")
        print("==========================================")

        # -------------------------------------------------
        # Remove data URL prefix
        #
        # Example:
        # data:image/png;base64,AAAA...
        # -------------------------------------------------

        if "," in image_base64:

            image_base64 = image_base64.split(
                ",",
                1
            )[1]

        # -------------------------------------------------
        # Decode Base64
        # -------------------------------------------------

        try:

            image_bytes = base64.b64decode(
                image_base64
            )

        except Exception as e:

            raise ValueError(
                f"Invalid base64 image: {e}"
            )

        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGBA")

        print(
            "Image size:",
            image.size,
        )

        # -------------------------------------------------
        # Remove background
        # -------------------------------------------------

        print("Removing background...")

        rembg_session = rembg.new_session()

        image = rembg.remove(
            image,
            session=rembg_session,
        )

        image = image.convert("RGBA")

        print("Background removed.")

        # -------------------------------------------------
        # Validate texture resolution
        # -------------------------------------------------

        try:

            texture_resolution = int(
                texture_resolution
            )

        except Exception:

            texture_resolution = 1024

        if texture_resolution not in (
            512,
            1024,
            2048,
        ):

            texture_resolution = 1024

        # -------------------------------------------------
        # Validate remesh option
        # -------------------------------------------------

        if remesh_option not in (
            "none",
            "triangle",
            "quad",
        ):

            remesh_option = "none"

        # -------------------------------------------------
        # Run SF3D
        # -------------------------------------------------

        print("==========================================")
        print("Running SF3D inference...")
        print("Texture:", texture_resolution)
        print("Remesh:", remesh_option)
        print("==========================================")

        with torch.inference_mode():

            mesh, global_dict = self.model.run_image(
                image,
                bake_resolution=texture_resolution,
                remesh=remesh_option,
                vertex_count=-1,
            )

        print("SF3D inference finished.")

        # -------------------------------------------------
        # Check mesh
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

        print("Exporting GLB...")

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
        # Convert GLB → Base64
        # -------------------------------------------------

        result = base64.b64encode(
            glb_bytes
        ).decode("utf-8")

        print("GLB encoded successfully.")

        return result


# =========================================================
# FASTAPI WEB API
# =========================================================

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

    # =====================================================
    # CORS
    # =====================================================

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    @web_app.get("/")
    async def health():

        return {
            "status": "ok",
            "service": "sf3d-backend",
        }

    # =====================================================
    # GENERATE 3D
    # =====================================================

    @web_app.post("/generate")
    async def generate_endpoint(data: dict):

        try:

            # -------------------------------------------------
            # Get image
            # -------------------------------------------------

            image = data.get("image")

            if not image:

                return {
                    "status": "error",
                    "message": "Missing image",
                }

            # -------------------------------------------------
            # Optional settings
            # -------------------------------------------------

            texture_resolution = data.get(
                "texture_resolution",
                1024,
            )

            remesh_option = data.get(
                "remesh_option",
                "none",
            )

            # -------------------------------------------------
            # Create model container
            # -------------------------------------------------

            print(
                "Creating SF3D model instance..."
            )

            model = SF3DModel()

            # -------------------------------------------------
            # Send image to GPU
            # -------------------------------------------------

            print(
                "Sending generation request to GPU..."
            )

            result = await model.generate_mesh.remote.aio(
                image,
                int(texture_resolution),
                str(remesh_option),
            )

            # -------------------------------------------------
            # Success
            # -------------------------------------------------

            print(
                "Generation completed successfully."
            )

            return {
                "status": "success",
                "model": result,
            }

        except Exception as e:

            import traceback

            print(
                "=========================================="
            )

            print(
                "GENERATION ERROR"
            )

            print(
                "=========================================="
            )

            traceback.print_exc()

            return {
                "status": "error",
                "message": str(e),
            }

    # =====================================================
    # Return FastAPI application
    # =====================================================

    return web_app
