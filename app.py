import os
import modal

# ============================================================
# SF3D MODAL CONTAINER IMAGE CONFIGURATION
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
        "huggingface_hub",
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://pytorch.org",
    )
    .workdir("/app")
    .run_commands("git clone https://github.com")
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

# ============================================================
# AUTONOMOUS MODULAR CACHE STORAGE VOLUME
# ============================================================
models_volume = modal.Volume.from_name("sf3d-models-volume", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache": models_volume},
)
class SF3DModel:

  @modal.enter()
  def load_model(self):
    import os
    import torch
    from huggingface_hub import login

    print("STARTING BACKEND INFRASTRUCTURE INITIALIZATION")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
      raise ValueError("HF_TOKEN variable is missing from runtime context container.")
    print("Authenticating with Hugging Face Hub...")
    login(token=hf_token)

    # ----------------============================================
    # PERSISTENT CACHE COMPONENT ROUTINES
    # ------------------------------------------------============
    u2net_path = "/root/.cache/rembg/u2net.onnx"
    if not os.path.exists(u2net_path):
      print("Cache Empty: Fetching rembg u2net.onnx asset weights to Volume...")
      from rembg import new_session
      temp_session = new_session() 
      models_volume.commit() 
      print("Rembg library baseline saved successfully!")

    sf3d_path = "/root/.cache/huggingface/hub/models--stabilityai--stable-fast-3d"
    if not os.path.exists(sf3d_path):
      print("Cache Empty: Sourcing SF3D model configurations from Hugging Face...")
      from huggingface_hub import snapshot_download
      snapshot_download(
          repo_id="stabilityai/stable-fast-3d",
          allow_patterns=["*.txt", "*.json", "*.safetensors"]
      )
      models_volume.commit() 
      print("SF3D system components stored inside your persistent cloud drive folder!")
    # --------------------------------============================

    from sf3d.system import SF3D

    print("Loading network weights locally from mounted Volume disk folder...")
    self.model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(device)
    self.model.eval()
    print("PIPELINE ENGINE READY")

  @modal.fastapi_endpoint(method="POST")
  def generate(self, item: dict):
    import base64
    import io
    import torch
    from PIL import Image
    from rembg import new_session
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
    except Exception as e:
      return {"error": f"Invalid base64 payload conversion structure: {e}"}

    try:
      image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
      return {"error": f"Failed to extract bitmap from data stream: {e}"}

    #  FIXED RESOLUTION SYNTAX CHECK
    if texture_resolution not in:
      texture_resolution = 1024
      
    if remesh_option not in ["none", "triangle", "quad"]:
      remesh_option = "triangle"

    print("Executing background stripping...")
    session = new_session()
    image = remove_background(image, session)

    print("Processing boundary resizing coordinates...")
    image = resize_foreground(image, 0.85)

    remesh = None
    if remesh_option == "triangle":
      remesh = "triangle"
    elif remesh_option == "quad":
      remesh = "quad"

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
      
