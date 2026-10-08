import base64
import io
import os
import sys

import modal


# ============================================================
# MODAL APP
# ============================================================

app = modal.App("sf3d-backend")


# ============================================================
# HUGGING FACE CACHE
# ============================================================

hf_cache = modal.Volume.from_name(
    "sf3d-weights-cache",
    create_if_missing=True,
)

HF_CACHE_DIR = "/root/.cache/huggingface"


# ============================================================
# BUILD SF3D IMAGE
# ============================================================

sf3d_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )

    # --------------------------------------------------------
    # Linux build tools
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Python build tools
    # --------------------------------------------------------

    .pip_install(
        "setuptools==69.5.1",
        "wheel",
    )

    # --------------------------------------------------------
    # PyTorch CUDA 12.1
    # --------------------------------------------------------

    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )

    # --------------------------------------------------------
    # Clone SF3D
    # --------------------------------------------------------

    .run_commands(
        "git clone https://github.com/Stability-AI/stable-fast-3d.git /app/stable-fast-3d"
    )

    # --------------------------------------------------------
    # Install SF3D's Python dependencies EXCEPT the two
    # local native extensions.
    # --------------------------------------------------------

    .run_commands(
        """
        cd /app/stable-fast-3d && \
        grep -v '^\\./texture_baker/' requirements.txt | \
        grep -v '^\\./uv_unwrapper/' > /tmp/sf3d_requirements.txt && \
        pip install -r /tmp/sf3d_requirements.txt
        """
    )

    # --------------------------------------------------------
    # Build texture_baker and uv_unwrapper AFTER Torch
    # is already installed.
    #
    # A10G = NVIDIA compute capability 8.6
    # --------------------------------------------------------

    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "MAX_JOBS": "2",
            "USE_CUDA": "1",
            "USE_NATIVE_ARCH": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )

    # --------------------------------------------------------
    # Build texture_baker
    # --------------------------------------------------------

    .run_commands(
        """
        cd /app/stable-fast-3d && \
        CUDA_HOME=/usr/local/cuda \
        TORCH_CUDA_ARCH_LIST=8.6 \
        MAX_JOBS=2 \
        USE_CUDA=1 \
        USE_NATIVE_ARCH=0 \
        pip install ./texture_baker/ --no-build-isolation
        """
    )

    # --------------------------------------------------------
    # Build uv_unwrapper
    # --------------------------------------------------------

    .run_commands(
        """
        cd /app/stable-fast-3d && \
        CUDA_HOME=/usr/local/cuda \
        TORCH_CUDA_ARCH_LIST=8.6 \
        MAX_JOBS=2 \
        USE_CUDA=1 \
        USE_NATIVE_ARCH=0 \
        pip install ./uv_unwrapper/ --no-build-isolation
        """
    )

    # --------------------------------------------------------
    # FastAPI
    # --------------------------------------------------------

    .pip_install(
        "fastapi",
        "uvicorn",
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{HF_CACHE_DIR}/transformers",
            "TORCH_HOME": "/root/.cache/torch",
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "MAX_JOBS": "2",
            "USE_CUDA": "1",
            "USE_NATIVE_ARCH": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


# ============================================================
# SF3D MODEL
# ============================================================

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

    # ========================================================
    # LOAD MODEL
    # ========================================================

    @modal.enter()
    def load_model(self):

        import torch

        # ----------------------------------------------------
        # Add SF3D repository
        # ----------------------------------------------------

        repo_path = "/app/stable-fast-3d"

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        print("==========================================")
        print("SF3D STARTING")
        print("==========================================")

        print(
            "PyTorch:",
            torch.__version__,
        )

        print(
            "CUDA available:",
            torch.cuda.is_available(),
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available."
            )

        print(
            "CUDA:",
            torch.version.cuda,
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        # ----------------------------------------------------
        # Test native modules BEFORE loading SF3D
        # ----------------------------------------------------

        print("Testing texture_baker...")

        import texture_baker

        print(
            "texture_baker loaded:",
            texture_baker,
        )

        print("Testing uv_unwrapper...")

        import uv_unwrapper

        print(
            "uv_unwrapper loaded:",
            uv_unwrapper,
        )

        print("Native SF3D extensions loaded successfully.")

        # ----------------------------------------------------
        # Official SF3D API
        # ----------------------------------------------------

        from sf3d.system import SF3D

        print("Loading Stable Fast 3D model...")

        self.model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )

        self.model.to("cuda")

        self.model.eval()

        print("==========================================")
        print("SF3D MODEL READY")
        print("==========================================")

        hf_cache.commit()


    # ========================================================
    # GENERATE MESH
    # ========================================================

    @modal.method()
    def generate_mesh(
        self,
        image_base64: str,
        texture_resolution: int = 1024,
        remesh_option: str = "none",
    ):

        import torch
        import rembg

        from PIL import Image

        from sf3d.utils import (
            remove_background,
            resize_foreground,
        )

        print("==========================================")
        print("SF3D GENERATION")
        print("==========================================")

        # ----------------------------------------------------
        # Remove data URL header
        # ----------------------------------------------------

        if "," in image_base64:
            image_base64 = image_base64.split(
                ",",
                1,
            )[1]

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        try:

            image_bytes = base64.b64decode(
                image_base64
            )

        except Exception as e:

            raise ValueError(
                f"Invalid base64 image: {e}"
            )

        # ----------------------------------------------------
        # Open
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGBA")

        print(
            "Input image:",
            image.size,
        )

        # ----------------------------------------------------
        # Background removal
        # ----------------------------------------------------

        print("Removing background...")

        rembg_session = rembg.new_session()

        image = remove_background(
            image,
            rembg_session,
        )

        # ----------------------------------------------------
        # Resize foreground
        # ----------------------------------------------------

        image = resize_foreground(
            image,
            0.85,
        )

        # ----------------------------------------------------
        # Settings
        # ----------------------------------------------------

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

        if remesh_option not in (
            "none",
            "triangle",
            "quad",
        ):
            remesh_option = "none"

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        print("Running SF3D...")

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                mesh, global_dict = self.model.run_image(
                    image,
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                    vertex_count=-1,
                )

        print("SF3D finished.")

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if mesh is None:
            raise RuntimeError(
                "SF3D returned no mesh."
            )

        if len(mesh.vertices) == 0:
            raise RuntimeError(
                "SF3D returned an empty mesh."
            )

        print(
            "Vertices:",
            len(mesh.vertices),
        )

        print(
            "Faces:",
            len(mesh.faces),
        )

        # ----------------------------------------------------
        # Export GLB
        # ----------------------------------------------------

        output = io.BytesIO()

        mesh.export(
            output,
            file_type="glb",
            include_normals=True,
        )

        output.seek(0)

        glb = output.read()

        print(
            "GLB bytes:",
            len(glb),
        )

        # ----------------------------------------------------
        # Return Base64
        # ----------------------------------------------------

        return base64.b64encode(
            glb
        ).decode("utf-8")


# ============================================================
# FASTAPI
# ============================================================

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

    # --------------------------------------------------------
    # CORS
    # --------------------------------------------------------

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    @web_app.get("/")
    async def health():

        return {
            "status": "ok",
            "service": "sf3d-backend",
        }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    @web_app.post("/generate")
    async def generate_endpoint(
        data: dict,
    ):

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

            print(
                "Starting SF3D GPU call..."
            )

            model = SF3DModel()

            result = await model.generate_mesh.remote.aio(
                image,
                int(texture_resolution),
                str(remesh_option),
            )

            return {
                "status": "success",
                "model": result,
            }

        except Exception as e:

            import traceback

            print("================================")
            print("SF3D ERROR")
            print("================================")

            traceback.print_exc()

            return {
                "status": "error",
                "message": str(e),
            }

    return web_app
