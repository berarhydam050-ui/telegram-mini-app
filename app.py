import base64
import binascii
import io
import json
import os
import re
import sys
import types
import traceback
import uuid
from pathlib import Path
from typing import Any
import modal


APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"

MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"

DEFAULT_TEXTURE_RESOLUTION = 2048
MAX_IMAGE_PIXELS = 40_000_000
FOREGROUND_SCALE = 0.85
REMESH_MODE = "none"
STARS_PRICE = 5  # Telegram Stars per generation — redeploy 2026-11-06

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
API_KEY = os.environ.get("MODAL_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN")

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry(
    "rhydam12/sf3d-gpu-worker:latest",
    add_python="3.10",
).pip_install("httpx")


def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("Provide an image as a base64 string.")
    if data.startswith("data:"):
        header, separator, data = data.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Invalid base64 data URI.")
    raw = base64.b64decode(data.split(",")[-1] if "," in data else data)
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGBA")


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
        subject = image.convert("RGBA")
    else:
        subject = rgba.crop(bbox)
    canvas = Image.new("RGBA", (1280, 1280), (0, 0, 0, 0))
    max_dim = int(1280 * FOREGROUND_SCALE)
    subject.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    canvas.paste(subject, ((1280 - subject.width) // 2, (1280 - subject.height) // 2), subject)
    return canvas


def write_job(job_id, payload):
    path = Path(CACHE_DIR) / "jobs" / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    models_volume.commit()


def read_job(job_id):
    path = Path(CACHE_DIR) / "jobs" / f"{job_id}.json"
    try:
        models_volume.reload()
    except Exception:
        pass
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def payment_path(pid):
    return Path(CACHE_DIR) / "payments" / f"{pid}.json"


def write_payment(pid, payload):
    path = payment_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    models_volume.commit()


def read_payment(pid):
    try:
        models_volume.reload()
    except Exception:
        pass
    path = payment_path(pid)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@app.cls(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    max_containers=1,
)
class SF3DModel:
    @modal.enter(snap=True)
    def load_model(self):
        import torch
        from huggingface_hub import login
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.device = torch.device(GPU_DEVICE)
        hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf:
            login(token=hf)
        if "/app/stable-fast-3d" not in sys.path:
            sys.path.append("/app/stable-fast-3d")
        from sf3d.system import SF3D
        self.model = SF3D.from_pretrained(MODEL_ID, config_name="config.yaml", weight_name="model.safetensors")
        self.model.eval()
        self.model.to(self.device)
        from rembg import new_session
        remove_background_and_center._session = new_session("u2net")
        torch.cuda.empty_cache()
        print("Snapshot ready")

    @modal.method()
    def process_image(self, item: dict):
        import torch
        job_id = item.get("job_id")
        if job_id:
            write_job(job_id, {"status": "running", "job_id": job_id})
        try:
            original = decode_image(item)
            processed = remove_background_and_center(original)
            texture_resolution = max(512, min(int(item.get("texture_resolution", DEFAULT_TEXTURE_RESOLUTION)), 2048))
            remesh_option = item.get("remesh", REMESH_MODE)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    mesh, _ = self.model.run_image(
                        processed,
                        bake_resolution=texture_resolution,
                        remesh=remesh_option if remesh_option != "none" else None,
                    )
            glb_bytes = mesh.export(file_type="glb")
            asset_id = str(uuid.uuid4())
            path = Path(CACHE_DIR) / "outputs" / f"{asset_id}.glb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(glb_bytes)
            models_volume.commit()
            result = {
                "success": True, "status": "done", "format": "glb",
                "mime_type": "model/gltf-binary", "asset_id": asset_id,
                "download_url": f"/download/{asset_id}", "job_id": job_id,
            }
            if item.get("return_base64", True):
                result["model"] = base64.b64encode(glb_bytes).decode("utf-8")
            if job_id:
                st = {"status": "done", "job_id": job_id, "asset_id": asset_id, "download_url": f"/download/{asset_id}"}
                if item.get("return_base64", True) and len(result.get("model", "")) < 16_000_000:
                    st["model"] = result["model"]
                write_job(job_id, st)
            return result
        except Exception as exc:
            traceback.print_exc()
            if job_id:
                write_job(job_id, {"status": "error", "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            torch.cuda.empty_cache()


@app.function(
    image=image,
    volumes={CACHE_DIR: models_volume},
    scaledown_window=2,
    timeout=60,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("telegram-bot-secret", required=False)],
)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request, Header
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import time, hmac, hashlib
    from urllib.parse import parse_qsl
    import httpx

    web_app = FastAPI(title="SF3D Image-to-3D API")
    web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                           allow_methods=["POST", "GET", "HEAD", "OPTIONS"], allow_headers=["*"],
                           expose_headers=["Content-Disposition", "Content-Length", "Content-Type"])

    def bot_token():
        return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN") or BOT_TOKEN

    def validate_init_data(init_data: str):
        token = bot_token()
        if not token:
            raise ValueError("Bot token not configured")
        parsed = dict(parse_qsl(init_data, keep_blank_values=True))
        recv = parsed.pop("hash", None)
        if not recv:
            raise ValueError("Invalid initData")
        check = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))
        secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(hmac.new(secret, check.encode(), hashlib.sha256).hexdigest(), recv):
            raise ValueError("Invalid Telegram auth")
        user = json.loads(parsed.get("user") or "{}")
        if not user.get("id"):
            raise ValueError("No Telegram user")
        return user

    def consume_payment(payment_id: str, user_id: int):
        if not UUID_RE.fullmatch(payment_id or ""):
            raise ValueError("Invalid payment id")
        data = read_payment(payment_id)
        if not data:
            raise ValueError("Payment not found — pay 5 Stars first")
        if data.get("status") != "paid":
            raise ValueError("Payment not completed")
        if data.get("used"):
            raise ValueError("This payment was already used")
        if int(data.get("user_id") or 0) != int(user_id):
            raise ValueError("Payment belongs to another user")
        data["used"] = True
        data["used_at"] = time.time()
        write_payment(payment_id, data)
        return data

    @web_app.get("/")
    async def health():
        return {"status": "ok", "service": APP_NAME, "model": MODEL_ID, "stars_price": STARS_PRICE, "payments_enabled": bool(bot_token())}

    @web_app.get("/job/{job_id}")
    async def get_job(job_id: str):
        if not UUID_RE.fullmatch(job_id):
            raise HTTPException(400, "Bad job id")
        for _ in range(3):
            data = read_job(job_id)
            if data:
                return data
            time.sleep(0.4)
        raise HTTPException(404, "Job not found")

    @web_app.get("/pay/{payment_id}")
    async def get_pay(payment_id: str):
        if not UUID_RE.fullmatch(payment_id):
            raise HTTPException(400, "Bad payment id")
        data = read_payment(payment_id)
        if not data:
            raise HTTPException(404, "Payment not found")
        return {"payment_id": payment_id, "status": data.get("status"), "used": bool(data.get("used")), "stars": data.get("stars", STARS_PRICE)}

    @web_app.post("/pay/create")
    async def create_pay(request: Request):
        token = bot_token()
        if not token:
            raise HTTPException(503, "Set TELEGRAM_BOT_TOKEN in Modal secret telegram-bot-secret")
        body = await request.json()
        try:
            user = validate_init_data(body.get("initData") or body.get("init_data") or "")
        except ValueError as e:
            raise HTTPException(401, str(e))
        payment_id = str(uuid.uuid4())
        write_payment(payment_id, {"payment_id": payment_id, "user_id": user["id"], "status": "pending", "stars": STARS_PRICE, "created_at": time.time(), "used": False})
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"https://api.telegram.org/bot{token}/createInvoiceLink", json={
                "title": "Form3D Generation",
                "description": "One high-quality 3D model from your photo",
                "payload": payment_id,
                "provider_token": "",
                "currency": "XTR",
                "prices": [{"label": "1× 3D model", "amount": STARS_PRICE}],
            })
            data = resp.json()
        if not data.get("ok"):
            raise HTTPException(502, f"Telegram invoice error: {data.get('description', data)}")
        return {"success": True, "payment_id": payment_id, "invoice_url": data["result"], "stars": STARS_PRICE}

    @web_app.post("/telegram/webhook")
    async def tg_webhook(request: Request):
        token = bot_token()
        if not token:
            return JSONResponse({"ok": True})
        try:
            update = await request.json()
        except Exception:
            return JSONResponse({"ok": True})
        pre = update.get("pre_checkout_query")
        if pre:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"https://api.telegram.org/bot{token}/answerPreCheckoutQuery",
                                  json={"pre_checkout_query_id": pre["id"], "ok": True})
            return JSONResponse({"ok": True})
        msg = update.get("message") or {}
        sp = msg.get("successful_payment")
        if sp and sp.get("currency") == "XTR":
            payment_id = sp.get("invoice_payload") or ""
            if UUID_RE.fullmatch(payment_id):
                existing = read_payment(payment_id) or {"payment_id": payment_id, "status": "pending", "used": False}
                if existing.get("status") != "paid":
                    existing["status"] = "paid"
                    existing["paid_at"] = time.time()
                    existing["telegram_payment_charge_id"] = sp.get("telegram_payment_charge_id")
                    if (msg.get("from") or {}).get("id"):
                        existing["user_id"] = msg["from"]["id"]
                    write_payment(payment_id, existing)
        return JSONResponse({"ok": True})

    @web_app.api_route("/download/{asset_id}", methods=["GET", "HEAD"])
    async def download_asset(asset_id: str):
        if not UUID_RE.fullmatch(asset_id):
            raise HTTPException(400, "Bad asset id")
        path = Path(CACHE_DIR) / "outputs" / f"{asset_id}.glb"
        for _ in range(5):
            try:
                models_volume.reload()
            except Exception:
                pass
            if path.is_file():
                break
            time.sleep(0.8)
        if not path.is_file():
            raise HTTPException(404, "Asset not found or expired")
        return Response(content=path.read_bytes(), media_type="model/gltf-binary", headers={
            "Content-Disposition": f'attachment; filename="form3d-{asset_id}.glb"',
            "Access-Control-Allow-Origin": "*",
        })

    @web_app.post("/")
    async def run_generate(request: Request, x_api_key: str | None = Header(None)):
        key = os.environ.get("MODAL_API_KEY") or API_KEY
        if key and x_api_key != key:
            raise HTTPException(401, "Unauthorized")
        try:
            data = await request.json()
            data.setdefault("texture_resolution", DEFAULT_TEXTURE_RESOLUTION)
            data.setdefault("remesh", REMESH_MODE)
            data.setdefault("return_base64", True)
            if bot_token():
                try:
                    user = validate_init_data(data.get("initData") or data.get("init_data") or "")
                    consume_payment(data.get("payment_id") or data.get("paymentId") or "", user["id"])
                except ValueError as e:
                    raise HTTPException(402, str(e))
            job_id = str(uuid.uuid4())
            data["job_id"] = job_id
            write_job(job_id, {"status": "queued", "job_id": job_id})
            call = SF3DModel().process_image.spawn(data)
            if not data.get("async", True):
                return call.get()
            return {"success": True, "status": "queued", "job_id": job_id, "poll_url": f"/job/{job_id}"}
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"SF3D generation failed: {type(e).__name__}: {e}")

    return web_app
