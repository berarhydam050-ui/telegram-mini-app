import base64
import binascii
import io
import os
import sys
import types
import traceback
from typing import Any
import modal

# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"

DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_IMAGE_PIXELS = 40_000_000
FOREGROUND_SCALE = 0.85
REMESH_MODE = "none"

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Build stage patch: Modifies network.py on disk BEFORE snapshot execution
image = (
    modal.Image.from_registry("rhydam12/sf3d-gpu-worker:latest", add_python="3.10")
    .run_commands(
        "if [ -f /app/stable_fast_3d/sf3d/models/network.py ]; then "
        "  sed -i 's/from torch.amp import custom_bwd, custom_fwd/from torch.cuda.amp import custom_bwd, custom_fwd/g' /app/stable_fast_3d/sf3d/models/network.py; "
        "  sed -i 's/from torch.amp import/from torch.cuda.amp import/g' /app/stable_fast_3d/sf3d/models/network.py; "
        "  sed -i 's/device_type=\"cuda\"//g' /app/stable_fast_3d/sf3d/models/network.py; "
        "  sed -i \"s/device_type='cuda'//g\" /app/stable_fast_3d/sf3d/models/network.py; "
        "  echo 'Successfully patched network.py on disk!'; "
        "else "
        "  echo 'Warning: Target network.py not found at path layout during image build.'; "
        "fi"
    )
)

# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64") or data.get("data")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("No valid base64 image payload provided.")
    if "," in data and "data:" in data[:30]:
        data = data.split(",", 1)[1]
    
    data = data.strip()
    raw = base64.b64decode(data, validate=False)
    if not raw:
        raise ValueError("Decoded image buffer is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Could not parse image format: {exc}") from exc


def remove_background_and_center(image):
    from PIL import Image
    from rembg import new_session, remove

    session = getattr(remove_background_and_center, "_session", None)
    if session is None:
        session = new_session("u2net")
        remove_background_and_center._session = session

    rgba = remove(image, session=session).convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        bbox = (0, 0, rgba.width, rgba.height)

    foreground = rgba.crop(bbox)
    side = max(foreground.width, foreground.height)
    target_side = max(1, int(side * FOREGROUND_SCALE))
    scale = target_side / max(foreground.width, foreground.height)
    new_size = (max(1, int(foreground.width * scale)), max(1, int(foreground.height * scale)))
    foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

    canvas_side = max(new_size)
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    position = ((canvas_side - foreground.width) // 2, (canvas_side - foreground.height) // 2)
    canvas.alpha_composite(foreground, dest=position)
    return canvas

# ============================================================
# 3. CPU-SNAPSHOT SCALE-TO-ZERO WORKER
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=15,          
    enable_memory_snapshot=True,  
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=5,
)
class SF3DModel:

    @modal.enter(snap=True)
    def freeze_to_cpu(self):
        """STEP 1: Runs during deployment. Freezes model into standard RAM."""
        import torch
        
        print("--- STARTING CPU MEMORY SNAPSHOT ---")
        
        # 1. Blind PyTorch to CUDA so SF3D loads safely into CPU RAM during build
        self._orig_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        torch.set_default_device('cpu')

        # 2. Import network safely since it has been patched on disk via build steps
        from stable_fast_3d.sf3d.models.network import SF3D
        
        # 3. Load model structure into CPU memory layout
        print("Loading SF3D pipeline state into frozen RAM snapshot...")
        self.pipeline = SF3D.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.safetensors",
            cache_dir=CACHE_DIR
        )
        self.pipeline.eval()
        print("--- CPU SNAPSHOT COMPLETED SUCCESSFULLY ---")

    @modal.enter()
    def hydrate_to_gpu(self):
        """STEP 2: Restores instantly from snapshot inside 10s and wakes up CUDA."""
        import torch
        
        print("--- RESTORING SNAPSHOT / WAKING INSTANCE ---")
        # Restore real CUDA operational metrics
        torch.cuda.is_available = self._orig_cuda_available
        
        # Shift weight layers from standard RAM snapshot onto actual container VRAM
        self.pipeline.to(GPU_DEVICE)
        print("--- MODEL PIPELINE ACTIVE ON CUDA VRAM ---")

    @modal.method()
    def process_mesh(self, image_data: Any) -> dict:
        """Runs mesh inference and guarantees safe multi-key payload arrays."""
        import torch
        
        # Preprocess input image using local functional wrappers
        pil_img = decode_image(image_data)
        processed_img = remove_background_and_center(pil_img)
        
        # --- (Your pipeline inference logic goes here) ---
        # output_mesh = self.pipeline(processed_img, ...)
        # glb_bytes = export_to_glb(output_mesh)
        
        # Simulating returning the final asset data payload
        mock_glb_data = b"glTF\x02\x00\x00\x00" 
        base64_payload = base64.b64encode(mock_glb_data).decode("utf-8")
        
        # Multi-key mappings to seamlessly prevent any client tracking array error codes
        return {
            "model": base64_payload,
            "model_base64": base64_payload,
            "glb": base64_payload
        }

# ============================================================
# 4. WEB ENDPOINT
# ============================================================

@app.function()
@modal.web_endpoint(method="POST")
def generate(item: dict):
    """Exposes the scale-to-zero model pool as an accessible Web API."""
    try:
        model_worker = SF3DModel()
        return model_worker.process_mesh.remote(item)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
        
