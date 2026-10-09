import base64
import io
import os
import sys
import types
import modal

# Build step: Bake weights directly into image layer for faster container startup
def download_weights_at_build():
    from huggingface_hub import snapshot_download
    from rembg import new_session

    print("Pre-caching rembg model weights...")
    new_session()

    print("Pre-caching Stable Fast 3D model weights...")
    snapshot_download(
        repo_id="stabilityai/stable-fast-3d",
        allow_patterns=["*.txt", "*.json", "*.safetensors"]
    )

image = (
    modal.Image.from_registry(
        "rhydam12/sf3d-gpu-worker:latest",
        add_python="3.10"
    )
    .run_function(download_weights_at_build)
)

app = modal.App("sf3d-backend")

@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=0,  # Zero idle cost: terminates instantly after processing
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SF3DModel:

  @modal.enter()
  def load_model(self):
    import torch
    import torch.cuda.amp

    # Patch PyTorch AMP device_type kwargs
    def safe_custom_fwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return torch.cuda.amp.custom_fwd(*args, **kwargs)

    def safe_custom_bwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return torch.cuda.amp.custom_bwd(*args, **kwargs)

    if not hasattr(torch, "amp"):
        torch.amp = types.ModuleType("amp")

    torch.amp.custom_fwd = safe_custom_fwd
    torch.amp.custom_bwd = safe_custom_bwd

    # Patch texture_baker C++ operations (interpolate and rasterize) for safe execution
    baker_path = "/opt/conda/lib/python3.10/site-packages/texture_baker/baker.py"
    if os.path.exists(baker_path):
      with open(baker_path, "r") as f:
        baker_code = f.read()
      if "cpu_safe_wrapper" not in baker_code:
        patch_header = """import torch

def cpu_safe_wrapper(fn):
    def wrapper(*args, **kwargs):
        args_cpu = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        kwargs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        res = fn(*args_cpu, **kwargs_cpu)
        if isinstance(res, torch.Tensor):
            return res.cuda()
        if isinstance(res, (tuple, list)):
            return type(res)(x.cuda() if isinstance(x, torch.Tensor) else x for x in res)
        return res
    return wrapper

"""
        baker_code = patch_header + baker_code.replace(
            "torch.ops.texture_baker_cpp.rasterize",
            "cpu_safe_wrapper(torch.ops.texture_baker_cpp.rasterize)"
        ).replace(
            "torch.ops.texture_baker_cpp.interpolate",
            "cpu_safe_wrapper(torch.ops.texture_baker_cpp.interpolate)"
        )
        with open(baker_path, "w") as f:
          f.write(baker_code)

    sys.path.append("/app/stable-fast-3d")
    from sf3d.system import SF3D

    # Load pretrained SF3D pipeline onto GPU
    self.model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(device)
    self.model.eval()

  @modal.method()
  def process_image(self, item: dict):
    import torch
    import torch.cuda.amp
    from PIL import Image
    from rembg import new_session

    sys.path.append("/app/stable-fast-3d")
    from sf3d.utils import (
        remove_background,
        resize_foreground,
    )

    image_base64 = item.get("image", "")
    texture_resolution = item.get("texture_resolution", 1024)
    remesh_option = item.get("remesh", "triangle")

    if "," in image_base64:
      image_base64 = image_base64.split(",", 1)[1]

    try:
      image_bytes = base64.b64decode(image_base64)
      image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
      return {"error": f"Invalid image payload: {e}"}

    # 1. Clean Background Stripping
    session = new_session()
    image = remove_background(image, session)

    # 2. Autocrop transparent borders to improve object centering
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # 3. Apply exact 0.85 Foreground Ratio matching Hugging Face Space
    image = resize_foreground(image, 0.85)

    remesh = None
    if str(remesh_option) in ["triangle", "quad"]:
      remesh = str(remesh_option)

    # 4. Fast Tensor Forward Pass
    with torch.inference_mode():
      if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          mesh, glob_dict = self.model.run_image(
              image, 
              bake_resolution=int(texture_resolution), 
              remesh=remesh
          )
      else:
        mesh, glob_dict = self.model.run_image(
            image, 
            bake_resolution=int(texture_resolution), 
            remesh=remesh
        )

    output_path = "/tmp/output.glb"
    mesh.export(output_path, file_type="glb", include_normals=True)

    with open(output_path, "rb") as f:
      glb_bytes = f.read()

    return {"model": base64.b64encode(glb_bytes).decode("utf-8")}


@app.function()
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/")
    async def run_generate(request: Request):
        data = await request.json()
        model_instance = SF3DModel()
        return await model_instance.process_image.remote.aio(data)

    return web_app
    
