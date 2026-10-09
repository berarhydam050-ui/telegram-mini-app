import base64
import io
import os
import sys
import types
import modal

image = modal.Image.from_registry(
    "rhydam12/sf3d-gpu-worker:latest",
    add_python="3.10"
)

app = modal.App("sf3d-backend")
models_volume = modal.Volume.from_name("sf3d-models-volume", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=2,  # Shuts down instantly after task completion ($0.00 idle cost)
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache": models_volume},
)
class SF3DModel:

  @modal.enter(snap=True)
  def load_model(self):
    import torch
    import torch.cuda.amp
    from huggingface_hub import login

    print("STARTING INITIALIZATION FOR ULTRA-REALISM SNAPSHOT")

    # 1. Patch PyTorch AMP custom_fwd / custom_bwd kwargs
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

    network_path = "/app/stable-fast-3d/sf3d/models/network.py"
    if os.path.exists(network_path):
      with open(network_path, "r") as f:
        code = f.read()
      code = code.replace("from torch.amp import custom_bwd, custom_fwd", "from torch.cuda.amp import custom_bwd, custom_fwd")
      code = code.replace('device_type="cuda"', '')
      code = code.replace("device_type='cuda'", '')
      with open(network_path, "w") as f:
        f.write(code)

    # 2. Patch texture_baker C++ ops to run safely on CPU
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

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
      raise ValueError("HF_TOKEN variable is missing from runtime context container.")
    print("Authenticating with Hugging Face Hub...")
    login(token=hf_token)

    u2net_path = "/root/.cache/rembg/u2net.onnx"
    if not os.path.exists(u2net_path):
      from rembg import new_session
      new_session() 
      models_volume.commit() 

    sf3d_path = "/root/.cache/huggingface/hub/models--stabilityai--stable-fast-3d"
    if not os.path.exists(sf3d_path):
      from huggingface_hub import snapshot_download
      snapshot_download(
          repo_id="stabilityai/stable-fast-3d",
          allow_patterns=["*.txt", "*.json", "*.safetensors"]
      )
      models_volume.commit() 

    sys.path.append("/app/stable-fast-3d")
    from sf3d.system import SF3D

    print("Loading network weights into memory for fast snapshot restore...")
    self.model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(device)
    self.model.eval()
    print("SNAPSHOT ENGINE READY")

  @modal.method()
  def process_image(self, item: dict):
    import base64
    import io
    import torch
    import torch.cuda.amp
    from PIL import Image
    from rembg import new_session

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

    sys.path.append("/app/stable-fast-3d")
    from sf3d.utils import (
        remove_background,
        resize_foreground,
    )

    image_base64 = item.get("image", "")
    texture_resolution = int(item.get("texture_resolution", 2048))
    remesh_option = item.get("remesh", "quad")

    if "," in image_base64:
      image_base64 = image_base64.split(",", 1)[1]

    try:
      image_bytes = base64.b64decode(image_base64)
      image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
      return {"error": f"Invalid image payload: {e}"}

    if texture_resolution not in [512, 1024, 2048]:
      texture_resolution = 2048
      
    if str(remesh_option) not in ["none", "triangle", "quad"]:
      remesh_option = "quad"

    session = new_session()
    image = remove_background(image, session)

    bbox = image.getbbox()
    if bbox:
      image = image.crop(bbox)

    image = resize_foreground(image, 0.85)

    remesh = None
    if remesh_option == "triangle":
      remesh = "triangle"
    elif remesh_option == "quad":
      remesh = "quad"

    # FORCE DEVICE CONSISTENCY TO PREVENT CPU/CUDA MISMATCH ON SNAPSHOT RESTORE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(device)

    print("Running tensor inference forward pass cycle...")
    with torch.inference_mode():
      if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          mesh, glob_dict = self.model.run_image(image, bake_resolution=texture_resolution, remesh=remesh)
      else:
        mesh, glob_dict = self.model.run_image(image, bake_resolution=texture_resolution, remesh=remesh)

    output_path = "/tmp/output.glb"
    mesh.export(output_path, file_type="glb", include_normals=True)

    with open(output_path, "rb") as f:
      glb_bytes = f.read()

    return {"model": base64.b64encode(glb_bytes).decode("utf-8")}


@app.function(image=image)
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
    
