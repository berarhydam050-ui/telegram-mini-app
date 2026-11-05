import base64
import io
import json
import os
import re
import sys
import traceback
import uuid
import time
import hmac
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl
import modal

APP_NAME = "sf3d-backend"
VOLUME_NAME = "sf3d-models-volume"
CACHE_DIR = "/root/.cache"
MODEL_ID = "stabilityai/stable-fast-3d"
GPU_DEVICE = "cuda:0"
DEFAULT_TEXTURE_RESOLUTION = 2048
FOREGROUND_SCALE = 0.85
REMESH_MODE = "none"

# product_id -> {stars, credits, title}
PACKS = {
    "single": {"stars": 5, "credits": 1, "title": "1× 3D model", "desc": "One high-quality 3D model"},
    "pack100": {"stars": 100, "credits": 25, "title": "Pack 25 models", "desc": "25 generations (100 Stars)"},
    "pack200": {"stars": 200, "credits": 55, "title": "Pack 55 models", "desc": "55 generations (200 Stars)"},
    "pack500": {"stars": 500, "credits": 150, "title": "Pack 150 models", "desc": "150 generations (500 Stars)"},
}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
API_KEY = os.environ.get("MODAL_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN")

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_registry("rhydam12/sf3d-gpu-worker:latest", add_python="3.10").pip_install("httpx")


def decode_image(data: Any):
    from PIL import Image, ImageOps
    if isinstance(data, dict):
        data = data.get("image") or data.get("image_base64") or data.get("imageBase64")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("Provide an image as a base64 string.")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    raw = base64.b64decode(data)
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
    subject = image.convert("RGBA") if bbox is None else rgba.crop(bbox)
    canvas = Image.new("RGBA", (1280, 1280), (0, 0, 0, 0))
    max_dim = int(1280 * FOREGROUND_SCALE)
    subject.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    canvas.paste(subject, ((1280 - subject.width) // 2, (1280 - subject.height) // 2), subject)
    return canvas


def _vol_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    models_volume.commit()


def _vol_read(path: Path):
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


def write_job(job_id, payload):
    _vol_write(Path(CACHE_DIR) / "jobs" / f"{job_id}.json", payload)


def read_job(job_id):
    return _vol_read(Path(CACHE_DIR) / "jobs" / f"{job_id}.json")


def write_payment(pid, payload):
    _vol_write(Path(CACHE_DIR) / "payments" / f"{pid}.json", payload)


def read_payment(pid):
    return _vol_read(Path(CACHE_DIR) / "payments" / f"{pid}.json")


def user_path(user_id: int) -> Path:
    return Path(CACHE_DIR) / "users" / f"{user_id}.json"


def get_user(user_id: int) -> dict:
    data = _vol_read(user_path(user_id))
    if not data:
        return {"user_id": user_id, "credits": 0}
    return data


def save_user(data: dict):
    _vol_write(user_path(int(data["user_id"])), data)


def add_credits(user_id: int, n: int):
    u = get_user(user_id)
    u["credits"] = int(u.get("credits") or 0) + int(n)
    u["updated_at"] = time.time()
    save_user(u)
    return u


def consume_credit(user_id: int):
    u = get_user(user_id)
    credits = int(u.get("credits") or 0)
    if credits < 1:
        raise ValueError("No credits left — buy Stars pack first")
    u["credits"] = credits - 1
    u["updated_at"] = time.time()
    save_user(u)
    return u


@app.cls(
    image=image, gpu="L4", volumes={CACHE_DIR: models_volume}, scaledown_window=2,
    enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")], timeout=900, max_containers=1,
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
                        processed, bake_resolution=texture_resolution,
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
    image=image, volumes={CACHE_DIR: models_volume}, scaledown_window=2, timeout=60,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False),
             modal.Secret.from_name("telegram-bot-secret", required=False)],
)
@modal.asgi_app()
def generate():
    from fastapi import FastAPI, HTTPException, Request, Header
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import httpx

    web_app = FastAPI(title="SF3D Image-to-3D API")
    web_app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["POST", "GET", "HEAD", "OPTIONS"], allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
    )

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

    @web_app.get("/")
    async def health():
        return {
            "status": "ok", "service": APP_NAME, "model": MODEL_ID,
            "packs": {k: {"stars": v["stars"], "credits": v["credits"]} for k, v in PACKS.items()},
            "payments_enabled": bool(bot_token()),
        }

    @web_app.get("/packs")
    async def list_packs():
        return {"packs": PACKS}

    @web_app.get("/credits")
    async def credits(request: Request):
        init_data = request.query_params.get("initData") or ""
        try:
            user = validate_init_data(init_data)
        except ValueError as e:
            raise HTTPException(401, str(e))
        u = get_user(user["id"])
        return {"user_id": user["id"], "credits": int(u.get("credits") or 0)}

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
        return {
            "payment_id": payment_id, "status": data.get("status"),
            "product_id": data.get("product_id"), "credits": data.get("credits"),
            "used": bool(data.get("used")),
        }

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
        product_id = body.get("product_id") or body.get("product") or "single"
        pack = PACKS.get(product_id)
        if not pack:
            raise HTTPException(400, f"Unknown product_id. Use: {', '.join(PACKS)}")
        payment_id = str(uuid.uuid4())
        write_payment(payment_id, {
            "payment_id": payment_id, "user_id": user["id"], "status": "pending",
            "product_id": product_id, "stars": pack["stars"], "credits": pack["credits"],
            "created_at": time.time(), "used": False,
        })
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"https://api.telegram.org/bot{token}/createInvoiceLink", json={
                "title": pack["title"],
                "description": pack["desc"],
                "payload": payment_id,
                "provider_token": "",
                "currency": "XTR",
                "prices": [{"label": pack["title"], "amount": pack["stars"]}],
            })
            data = resp.json()
        if not data.get("ok"):
            raise HTTPException(502, f"Telegram invoice error: {data.get('description', data)}")
        return {
            "success": True, "payment_id": payment_id, "invoice_url": data["result"],
            "product_id": product_id, "stars": pack["stars"], "credits": pack["credits"],
        }

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
                await client.post(
                    f"https://api.telegram.org/bot{token}/answerPreCheckoutQuery",
                    json={"pre_checkout_query_id": pre["id"], "ok": True},
                )
            return JSONResponse({"ok": True})
        msg = update.get("message") or {}
        sp = msg.get("successful_payment")
        if sp and sp.get("currency") == "XTR":
            payment_id = sp.get("invoice_payload") or ""
            if UUID_RE.fullmatch(payment_id):
                existing = read_payment(payment_id) or {"payment_id": payment_id, "status": "pending"}
                if existing.get("status") != "paid":
                    existing["status"] = "paid"
                    existing["paid_at"] = time.time()
                    existing["telegram_payment_charge_id"] = sp.get("telegram_payment_charge_id")
                    uid = (msg.get("from") or {}).get("id") or existing.get("user_id")
                    if uid:
                        existing["user_id"] = uid
                        credits = int(existing.get("credits") or PACKS.get(existing.get("product_id") or "single", {}).get("credits") or 1)
                        add_credits(int(uid), credits)
                        existing["credits_granted"] = credits
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
        return Response(
            content=path.read_bytes(), media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f'attachment; filename="form3d-{asset_id}.glb"',
                "Access-Control-Allow-Origin": "*",
            },
        )

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
                    # Prefer wallet credits; optional one-shot payment_id still supported
                    payment_id = data.get("payment_id") or data.get("paymentId") or ""
                    if payment_id:
                        pay = read_payment(payment_id)
                        if pay and pay.get("status") == "paid" and not pay.get("used"):
                            if int(pay.get("user_id") or 0) != int(user["id"]):
                                raise ValueError("Payment belongs to another user")
                            pay["used"] = True
                            write_payment(payment_id, pay)
                            # credits already granted on webhook for packs; for safety grant 1 if none
                            u = get_user(user["id"])
                            if int(u.get("credits") or 0) < 1 and int(pay.get("credits") or 0) >= 1:
                                add_credits(user["id"], int(pay["credits"]))
                    consume_credit(user["id"])
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
