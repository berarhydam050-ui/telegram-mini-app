import base64
import io
import os
import modal

cache_volume = modal.Volume.from_name("sf3d-weights-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"

sf3d_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121"
    )
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
    container_idle_timeout=60
)
class SF3DModel:
    @modal.enter()
    def setup(self):
        import sys
        sys.path.append("/app/stable_fast_3d")
        import torch
        from sf3d.pipeline import StableFast3DPipeline

        os.environ["HF_HOME"] = CACHE_DIR
        
        print("Loading Stable Fast 3D into GPU memory...")
        self.pipeline = StableFast3DPipeline.from_pretrained(
            "stabilityai/stable-fast-3d",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        ).to("cuda")
        print("Model loaded successfully.")

    @modal.web_endpoint(method="POST")
    def generate(self, data: dict):
        import sys
        sys.path.append("/app/stable_fast_3d")
        import torch
        from PIL import Image
        from rembg import remove
        from sf3d.utils import save_glb

        try:
            base64_img = data.get("image")
            if not base64_img:
                return {"status": "error", "message": "Missing 'image' parameter"}

            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            image_bytes = base64.b64decode(base64_img)
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            clean_image = remove(input_image)

            with torch.inference_mode():
                outputs = self.pipeline(clean_image, input_processing=True)

            glb_buffer = io.BytesIO()
            save_glb(glb_buffer, outputs)
            glb_buffer.seek(0)

            encoded_glb = base64.b64encode(glb_buffer.read()).decode("utf-8")
            return {"status": "success", "model": encoded_glb}

        except Exception as e:
            return {"status": "error", "message": str(e)}
          
