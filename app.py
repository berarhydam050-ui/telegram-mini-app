import runpod
import torch
import base64
import io
import sys
import os
from PIL import Image
from rembg import remove, new_session
import trimesh

# Ensure SF3D and dependencies are discoverable
sys.path.append("/content/stable-fast-3d")
sys.path.append("/content/stable-fast-3d/ComfyUI")

from sf3d.system import SF3D

# Pre-initialize device and rembg session on container spin-up
device = "cuda" if torch.cuda.is_available() else "cpu"
rembg_session = new_session("u2net")

# Pre-load SF3D model into VRAM during cold start
print(f"Loading SF3D model into VRAM on device: {device}...")
model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors"
)
model.to(device)
model.eval()

def handler(job):
    try:
        job_input = job.get("input", {})
        image_base64 = job_input.get("image")

        if not image_base64:
            return {"error": "Missing image input parameter"}

        # Safely strip Data URI prefix (e.g., "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Background Removal
        clean_image = remove(input_image, session=rembg_session)

        # Generate 3D Mesh
        with torch.no_grad():
            mesh = model.run_image(clean_image)

        # Convert output to binary GLB
        if isinstance(mesh, trimesh.Trimesh):
            scene = trimesh.Scene(mesh)
            glb_bytes = scene.export(file_type="glb")
        elif isinstance(mesh, trimesh.Scene):
            glb_bytes = mesh.export(file_type="glb")
        elif hasattr(mesh, "export"):
            glb_bytes = mesh.export(file_type="glb")
        else:
            scene = trimesh.Scene(mesh)
            glb_bytes = scene.export(file_type="glb")

        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")

        # Return explicit 'glb' key for Telegram Mini App frontend
        return {
            "glb": f"data:model/gltf-binary;base64,{glb_base64}"
        }

    except Exception as e:
        print(f"Handler execution error: {str(e)}")
        return {"error": str(e)}

# Start RunPod Serverless worker framework
runpod.serverless.start({"handler": handler})
