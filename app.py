import modal

# ============================================================
# SF3D MODAL IMAGE
# ============================================================
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",
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
        "huggingface_hub",  # Explicitly added for authentication
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .workdir("/app")
    .run_commands("git clone https://github.com/Stability-AI/stable-fast-3d.git")
    .workdir("/app/stable-fast-3d")
    .run_commands(
        "grep -v '^./texture_baker/' requirements.txt | grep -v '^./uv_unwrapper/' > /tmp/sf3d_requirements.txt",
        "pip install -r /tmp/sf3d_requirements.txt",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "MAX_JOBS": "2",
            "USE_CUDA": "1",
            "USE_NATIVE_ARCH": "0",
        }
    )
    .run_commands("pip install ./texture_baker/ --no-build-isolation")
    .run_commands("pip install ./uv_unwrapper/ --no-build-isolation")
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-multipart",
    )
)

app = modal.App("sf3d-backend")


@app.cls(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SF3DModel:

  @modal.enter()
  def load_model(self):
    import os
    import torch
    from huggingface_hub import login

    print("STARTING SF3D")
    
    # 1. Explicitly authenticate with Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
      raise ValueError("HF_TOKEN is missing. Check your huggingface-secret in Modal.")
    print("Authenticating with Hugging Face...")
    login(token=hf_token)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
      print("CUDA:", torch.version.cuda)
      print("GPU:", torch.cuda.get_device_name(0))

    import texture_baker
    print("texture_baker OK")
    
    import uv_unwrapper
    print("uv_unwrapper OK")

    from sf3d.system import SF3D

    # 2. Load the model now that we are authenticated
    self.model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(device)
    self.model.eval()
    print("SF3D MODEL LOADED")
    print("Device:", device)

  @modal.method()
  def generate_mesh(
      self,
      image_base64: str,
      texture_resolution: int = 1024,
      remesh_option: str = "triangle",
  ):
    import base64
    import io
    import torch
    from PIL import Image
    from rembg import new_session
    from sf3d.utils import (
        remove_background,
        resize_foreground,
    )

    if "," in image_base64:
      image_base64 = image_base64.split(",", 1)[1]

    try:
      image_bytes = base64.b64decode(image_base64)
    except Exception as e:
      raise ValueError(f"Invalid base64 image: {e}")

    try:
      image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
      raise ValueError(f"Could not open image: {e}")

    if texture_resolution not in [512, 1024, 2048]:
      texture_resolution = 1024
    if remesh_option not in ["none", "triangle", "quad"]:
      remesh_option = "triangle"

    print("Removing background...")
    session = new_session()
    image = remove_background(image, session)

    print("Resizing foreground...")
    image = resize_foreground(image, 0.85)

    remesh = None
    if remesh_option == "triangle":
      remesh = "triangle"
    elif remesh_option == "quad":
      remesh = "quad"

    print("Running SF3D...")
    with torch.inference_mode():
      if torch.cuda.is_available():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
          mesh, glob_dict = self.model.run_image(
              image,
              bake_resolution=texture_resolution,
              remesh=remesh,
          )
      else:
        mesh, glob_dict = self.model.run_image(
            image,
            bake_resolution=texture_resolution,
            remesh=remesh,
        )

    output_path = "/tmp/output.glb"
    mesh.export(
        output_path,
        file_type="glb",
        include_normals=True,
    )

    with open(output_path, "rb") as f:
      glb_bytes = f.read()

    return base64.b64encode(glb_bytes).decode("utf-8")


@app.function(
    image=image,
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def api():
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from pydantic import BaseModel

  web_app = FastAPI()

  web_app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  class GenerateRequest(BaseModel):
    image: str
    texture_resolution: int = 1024
    remesh: str = "triangle"

  @web_app.get("/")
  async def root():
    return {
        "status": "ok",
        "service": "SF3D",
        "gpu": "A10G",
    }

  @web_app.post("/generate")
  async def generate(request: GenerateRequest):
    try:
      model = SF3DModel()
      result = await model.generate_mesh.remote.aio(
          request.image,
          int(request.texture_resolution),
          str(request.remesh),
      )
      return {
          "status": "success",
          "model": result,
      }
    except Exception as e:
      print("GENERATION ERROR:", repr(e))
      return {
          "status": "error",
          "error": str(e),
      }

  return web_app
    
