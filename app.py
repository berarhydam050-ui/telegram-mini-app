import base64
import io
import modal

def download_hf_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="stabilityai/stable-fast-3d")

sf3d_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("rembg", "pillow", "trimesh", "accelerate", "transformers", "diffusers", "einops", "huggingface_hub")
    .run_commands(
        "git clone https://github.com/stability-ai/stable-fast-3d /app/stable_fast_3d",
        "cd /app/stable_fast_3d && pip install -r requirements.txt"
    )
    .run_function(download_hf_weights)
)

app = modal.App("sf3d-backend")

@app.cls(
    image=sf3d_image,
    gpu="A10G",
    scaledown_window=60
)
class SF3DModel:
    @modal.enter()
    def setup(self):
        import sys
        sys.path.append("/app/stable_fast_3d")
        import torch
        from sf3d.pipeline import StableFast3DPipeline

        self.pipeline = StableFast3DPipeline.from_pretrained(
            "stabilityai/stable-fast-3d",
            torch_dtype=torch.float16
        ).to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, data: dict):
        import sys
        sys.path.append("/app/stable_fast_3d")
        import torch
        from PIL import Image
        from rembg import remove
        from sf3d.utils import save_glb

        try:
            img_str = data.get("image")
            if not img_str:
                return {"status": "error", "message": "Missing image"}

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

            b64_out = base64.b64encode(buf.read()).decode("utf-8")
            return {"status": "success", "model": b64_out}

        except Exception as e:
            return {"status": "error", "message": str(e)}
            
