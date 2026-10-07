import base64
import io
import os
import modal

cache_volume = modal.Volume.from_name("sf3d-weights-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"

sf3d_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("rembg", "pillow", "trimesh", "accelerate", "transformers", "diffusers", "einops")
    .run_commands(
        "git clone https://github.com/stability-ai/stable-fast-3d /app/stable_fast_3d",
        "pip install -r /app/stable_fast_3d/requirements.txt"
    )
)

app = modal.App("sf3d-backend")

@app.cls(
    image=sf3d_image,
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume},
    scaledown_window=60
)
class SF3DModel:
    @modal.enter()
    def setup(self):
        
