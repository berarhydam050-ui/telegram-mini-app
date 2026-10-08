import base64
import io
import os
import sys

# Inject repository directory into Python path globally before imports
if "/app/stable_fast_3d" not in sys.path:
    sys.path.insert(0, "/app/stable_fast_3d")

import modal

cache_volume = modal.Volume.from_name("sf3d-weights-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"

sf3d_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "rembg",
        "pillow",
        "trimesh",
        "accelerate",
        "transformers",
        "diffusers",
        "einops",
        "huggingface_hub",
        "fastapi[standard]"
    )
    .run_commands(
        "git clone https://github.com/stability-ai/stable-fast-3d /app/stable_fast_3d",
        "cd /app/stable_fast_3d && pip install -r requirements.txt",
        "python3 -c \"import site, shutil; shutil.copytree('/app/stable_fast_3d/sf3d', site.getsitepackages()[0] + '/sf3d', dirs_exist_ok=True)\""
    )
)

app = modal.App("sf3d-backend")

@app.cls(
    image=sf3d_image,
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume},
    scaledown_window=60,
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class SF3DModel:
    @modal.enter()
    def setup(self):
        import torch
        from sf3d.pipeline import StableFast3DPipeline

        os.environ["HF_HOME"] = CACHE_DIR
        self.pipeline = StableFast3DPipeline.from_pretrained(
            "stabilityai/stable-fast-3d",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN")
        ).to("cuda")

        cache_volume.commit()

    @modal.method()
    def generate_mesh(self, img_str: str):
        import torch
        from PIL import Image
        from rembg import remove
        from sf3d.utils import save_glb

        if "," in img_str:
            img_str = img_str.split(",")[1]

        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        clean_img = remove(img)

        with torch.inference_mode():
            out = self.pipeline(clean_img, input_processing=True)

        buf = io.BytesIO()
        save_glb(buf, out)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")

@app.function(
    image=sf3d_image,
    timeout=120
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/generate")
    async def generate_endpoint(data: dict):
        try:
            img_str = data.get("image")
            if not img_str:
                return {"status": "error", "message": "Missing image"}

            model = SF3DModel()
            b64_out = await model.generate_mesh.remote.aio(img_str)
            return {"status": "success", "model": b64_out}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return web_app
    
