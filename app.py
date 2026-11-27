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

# product_id -> {stars, credits, title, desc}
PACKS = {
    "first": {"stars": 80, "credits": 10, "title": "First purchase 10 models", "desc": "Welcome offer: 10 models for 80 Stars"},
    "p3": {"stars": 50, "credits": 3, "title": "3 models", "desc": "Balance top-up: 3 models"},
    "p10": {"stars": 150, "credits": 10, "title": "10 models", "desc": "Balance top-up: 10 models"},
    "p30": {"stars": 390, "credits": 30, "title": "30 models", "desc": "Balance top-up: 30 models"},
    "p100": {"stars": 990, "credits": 100, "title": "100 models", "desc": "Balance top-up: 100 models"},
    # legacy aliases
    "single": {"stars": 50, "credits": 3, "title": "3 models", "desc": "Balance top-up: 3 models"},
}

# Free unlimited users (Telegram user IDs)
FREE_USER_IDS = {
    6187218793,   # unlimited free generations
}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
API_KEY = os.environ.get("MODAL_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN")

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_registry("rhydam12/sf3d-gpu-worker:latest", add_python="3.10").pip_install("httpx")
# Web app only. Keeps the GPU image unchanged so a UI update does not rebuild the GPU layer.
web_image = image.add_local_file("index.html", "/root/form3d-index.html")
UI_URL = "https://berarhydam050--sf3d-backend-generate.modal.run/ui?v=4"


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
    # Free users never pay
    if int(user_id) in FREE_USER_IDS:
        return get_user(user_id)

    u = get_user(user_id)
    credits = int(u.get("credits") or 0)
    if credits < 1:
        raise ValueError("No credits left \u2014 buy Stars pack first")
    u["credits"] = credits - 1
    u["updated_at"] = time.time()
    save_user(u)
    return u



def _patch_texture_baker_cpu_fallback():
    """The worker image ships texture_baker compiled for CPU only.
    Rasterize/interpolate on CUDA then raises NotImplementedError.
    Keep mesh inference on the L4; bake textures on CPU when the CUDA op is missing.
    """
    import torch
    try:
        from texture_baker.baker import TextureBaker
    except Exception as exc:
        print("texture_baker not importable:", exc)
        return
    if getattr(TextureBaker, "_cpu_fallback_patched", False):
        return

    orig_rasterize = TextureBaker.rasterize
    orig_interpolate = TextureBaker.interpolate
    state = {"cpu": None}

    def _to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value

    def rasterize(self, uv, face_indices, bake_resolution):
        if state["cpu"] is False:
            return orig_rasterize(self, uv, face_indices, bake_resolution)
        if state["cpu"] is True:
            dev = uv.device
            out = orig_rasterize(self, _to_cpu(uv), _to_cpu(face_indices), bake_resolution)
            return out.to(dev)
        try:
            out = orig_rasterize(self, uv, face_indices, bake_resolution)
            state["cpu"] = False
            print("texture_baker CUDA rasterize ok")
            return out
        except NotImplementedError:
            state["cpu"] = True
            print("texture_baker CUDA kernel missing; baking on CPU")
            dev = uv.device
            out = orig_rasterize(self, _to_cpu(uv), _to_cpu(face_indices), bake_resolution)
            return out.to(dev)

    def interpolate(self, attr, rast, face_indices, *args, **kwargs):
        if state["cpu"]:
            dev = attr.device
            out = orig_interpolate(
                self, _to_cpu(attr), _to_cpu(rast), _to_cpu(face_indices),
                *[_to_cpu(a) for a in args],
                **{k: _to_cpu(v) for k, v in kwargs.items()},
            )
            return out.to(dev)
        return orig_interpolate(self, attr, rast, face_indices, *args, **kwargs)

    TextureBaker.rasterize = rasterize
    TextureBaker.interpolate = interpolate
    TextureBaker._cpu_fallback_patched = True


def _cuda_is_live() -> bool:
    """Fast check only. Do not load weights on a dead L4 — that is what burns credit."""
    import ctypes
    import time
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        print("libcuda missing:", exc)
        return False
    for attempt in range(6):
        try:
            rc = int(cu.cuInit(0))
        except Exception as exc:
            print("cuInit exception:", exc)
            rc = -1
        if rc == 0:
            print(f"cuInit ok (attempt {attempt + 1})")
            return True
        print(f"cuInit failed attempt {attempt + 1}/6 rc={rc}")
        time.sleep(0.35)
    return False


def _release_bad_gpu():
    try:
        modal.experimental.stop_fetching_inputs()
        print("Released unhealthy L4 so the job can land on another live one")
    except Exception as exc:
        print("stop_fetching_inputs failed:", exc)


def _bump_gpu_attempt(job_id: str) -> int:
    path = Path(CACHE_DIR) / "jobs" / f"{job_id}.attempts"
    try:
        models_volume.reload()
    except Exception:
        pass
    n = 0
    if path.is_file():
        try:
            n = int(path.read_text().strip() or "0")
        except Exception:
            n = 0
    n += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(n))
    try:
        models_volume.commit()
    except Exception:
        pass
    return n


@app.cls(
    image=image,
    # Any L4 Modal can place, not one pinned machine. Dead cards are dropped in enter().
    gpu="L4",
    volumes={CACHE_DIR: models_volume},
    # Idle GPU bills by the second. 15s covers a retry, then it shuts off.
    scaledown_window=15,
    enable_memory_snapshot=False,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    startup_timeout=420,
    max_containers=1,
    # Only the explicit "L4 not live" raise uses these. Inference errors return.
    retries=2,
)

class SF3DModel:
    @modal.enter()
    def load_model(self):
        self.ready = False
        self.fatal_error = None
        # Refuse a dead L4 before touching the model. A 3s CUDA failure must not
        # download weights or crash-loop (deployed apps retry crashes forever).
        if not _cuda_is_live():
            print("This L4 is not live - not loading the model")
            _release_bad_gpu()
            return
        try:
            self._load_weights()
            self.ready = True
            print("L4 live, model ready")
        except Exception as exc:
            traceback.print_exc()
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            _release_bad_gpu()

    def _load_weights(self):
        import torch
        from huggingface_hub import login
        import torch.amp as _amp
        import torch.cuda.amp as _camp

        def _drop_device_type(fn):
            def wrapped(*args, **kwargs):
                kwargs.pop("device_type", None)
                return fn(*args, **kwargs)
            return wrapped

        _camp.custom_fwd = _drop_device_type(_camp.custom_fwd)
        _camp.custom_bwd = _drop_device_type(_camp.custom_bwd)
        _amp.custom_fwd = _camp.custom_fwd
        _amp.custom_bwd = _camp.custom_bwd
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.device = torch.device(GPU_DEVICE)
        hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf:
            login(token=hf)
        if "/app/stable-fast-3d" not in sys.path:
            sys.path.append("/app/stable-fast-3d")
        _patch_texture_baker_cpu_fallback()
        from sf3d.system import SF3D
        self.model = SF3D.from_pretrained(MODEL_ID, config_name="config.yaml", weight_name="model.safetensors")
        self.model.eval()
        self.model.to(self.device)
        from rembg import new_session
        remove_background_and_center._session = new_session("u2net")
        torch.cuda.empty_cache()

    @modal.method()
    def process_image(self, item: dict):
        import torch
        job_id = item.get("job_id")
        if not getattr(self, "ready", False):
            fatal = getattr(self, "fatal_error", None)
            if fatal:
                if job_id:
                    write_job(job_id, {"status": "error", "job_id": job_id, "error": fatal, "user_id": item.get("user_id")})
                return {"success": False, "status": "error", "error": fatal, "job_id": job_id}
            n = _bump_gpu_attempt(job_id) if job_id else 1
            if n >= 3:
                msg = "No live L4 right now. Wait a minute and tap Generate again."
                if job_id:
                    write_job(job_id, {"status": "error", "job_id": job_id, "error": msg, "user_id": item.get("user_id")})
                return {"success": False, "status": "error", "error": msg, "job_id": job_id}
            if job_id:
                write_job(job_id, {"status": "queued", "job_id": job_id, "detail": "Switching to another L4", "user_id": item.get("user_id")})
            # Finite retry (retries=2). Do not crash the container - that loops forever on a deployed app.
            raise RuntimeError("L4 not live")
        if job_id:
            write_job(job_id, {"status": "running", "job_id": job_id, "user_id": item.get("user_id")})
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
                # Keep the job record small. The GLB is downloaded separately.
                # Stuffing the model into the volume commit was stalling the API.
                write_job(job_id, {
                    "status": "done", "job_id": job_id, "asset_id": asset_id,
                    "download_url": f"/download/{asset_id}", "user_id": item.get("user_id"),
                })
            return result
        except Exception as exc:
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            if job_id:
                write_job(job_id, {"status": "error", "job_id": job_id, "error": err, "user_id": item.get("user_id")})
            # Return, don't raise: a raised inference error would run the GPU again.
            return {"success": False, "status": "error", "error": err, "job_id": job_id}
        finally:
            torch.cuda.empty_cache()


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("httpx"),
    secrets=[modal.Secret.from_name("telegram-bot-secret")],
    timeout=60,
)
def publish_webapp():
    """CPU-only. Points the bot button at the uncached page. Does not start a GPU."""
    import httpx
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("telegram bot token missing")
    base = f"https://api.telegram.org/bot{token}"
    menu = httpx.post(base + "/getChatMenuButton", json={}, timeout=20).json()
    print("current menu", menu)
    new_menu = httpx.post(base + "/setChatMenuButton", json={
        "menu_button": {"type": "web_app", "text": "Image to 3D", "web_app": {"url": UI_URL}}
    }, timeout=20).json()
    print("set menu", new_menu)
    note = httpx.post(base + "/sendMessage", json={
        "chat_id": 6187218793,
        "text": "Form3D is updated. Open it from this button so Telegram loads the new page. Finding a live L4 can take a few minutes. Leave it on Waiting, and tap Generate only once.",
        "reply_markup": {"inline_keyboard": [[{"text": "Open Form3D", "web_app": {"url": UI_URL}}]]},
    }, timeout=20).json()
    print("notify", {k: note.get(k) for k in ("ok", "description")})


@app.function(
    image=web_image, volumes={CACHE_DIR: models_volume}, scaledown_window=120, timeout=180,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("telegram-bot-secret"),
    ],
)
@modal.asgi_app()
def generate():
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Header
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import asyncio
    import httpx

    web_app = FastAPI(title="SF3D Image-to-3D API")
    web_app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["POST", "GET", "HEAD", "OPTIONS"], allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
    )
    recent_reqs = {}

    @web_app.get("/ui")
    async def ui_page():
        html = Path("/root/form3d-index.html").read_text()
        return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})

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
            "status": "ok", "service": APP_NAME, "model": MODEL_ID, "gpu": "L4",
            "scheduler": "any-live-l4",
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
        uid = int(user["id"])
        if uid in FREE_USER_IDS:
            return {"user_id": uid, "credits": 999999, "free": True}
        u = get_user(uid)
        return {"user_id": uid, "credits": int(u.get("credits") or 0), "free": False}

    @web_app.get("/job/{job_id}")
    async def get_job(job_id: str):
        if not UUID_RE.fullmatch(job_id):
            raise HTTPException(400, "Bad job id")
        for _ in range(3):
            data = await asyncio.to_thread(read_job, job_id)
            if data:
                return data
            await asyncio.sleep(0.25)
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
        product_id = body.get("product_id") or body.get("product") or "p3"
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
                        credits = int(existing.get("credits") or PACKS.get(existing.get("product_id") or "p3", {}).get("credits") or 1)
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
                await asyncio.to_thread(models_volume.reload)
            except Exception:
                pass
            if path.is_file():
                break
            await asyncio.sleep(0.4)
        if not path.is_file():
            raise HTTPException(404, "Asset not found or expired")
        return Response(
            content=path.read_bytes(), media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f'attachment; filename="form3d-{asset_id}.glb"',
                "Access-Control-Allow-Origin": "*",
            },
        )

    REQ_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")

    def _req_path(client_req: str) -> Path:
        return Path(CACHE_DIR) / "reqs" / f"{client_req}.json"

    def accept_job(data: dict, job_id: str, user) -> None:
        # Runs after the HTTP response is already on the way. Volume commits and
        # GPU queueing can take minutes while an L4 is found; the phone must not wait.
        data.setdefault("texture_resolution", DEFAULT_TEXTURE_RESOLUTION)
        data.setdefault("remesh", REMESH_MODE)
        data.setdefault("return_base64", False)
        client_req = str(data.get("client_req") or "")
        if not REQ_RE.fullmatch(client_req):
            client_req = ""
        if client_req:
            prev = _vol_read(_req_path(client_req))
            if prev and prev.get("job_id") and prev.get("job_id") != job_id:
                return
        if user and bot_token():
            payment_id = data.get("payment_id") or data.get("paymentId") or ""
            if payment_id:
                pay = read_payment(payment_id)
                if pay and pay.get("status") == "paid" and not pay.get("used"):
                    if int(pay.get("user_id") or 0) != int(user["id"]):
                        raise HTTPException(402, "Payment belongs to another user")
                    pay["used"] = True
                    write_payment(payment_id, pay)
                    u = get_user(user["id"])
                    if int(u.get("credits") or 0) < 1 and int(pay.get("credits") or 0) >= 1:
                        add_credits(user["id"], int(pay["credits"]))
            try:
                consume_credit(user["id"])
            except ValueError as e:
                write_job(job_id, {"status": "error", "job_id": job_id, "error": str(e), "user_id": int(user["id"])})
                return
        tg_user_id = int(user["id"]) if user else None
        gpu_item = {
            "image": data.get("image") or data.get("image_base64") or data.get("imageBase64"),
            "texture_resolution": data.get("texture_resolution"),
            "remesh": data.get("remesh"),
            "return_base64": bool(data.get("return_base64", False)),
            "job_id": job_id,
            "user_id": tg_user_id,
        }
        job_meta = {"status": "queued", "job_id": job_id}
        if tg_user_id:
            job_meta["user_id"] = tg_user_id
        write_job(job_id, job_meta)
        if client_req:
            _vol_write(_req_path(client_req), {"job_id": job_id, "status": "queued", "user_id": tg_user_id})
        try:
            SF3DModel().process_image.spawn(gpu_item)
        except Exception as exc:
            if tg_user_id and int(tg_user_id) not in FREE_USER_IDS:
                try:
                    add_credits(int(tg_user_id), 1)
                except Exception:
                    pass
            write_job(job_id, {
                "status": "error", "job_id": job_id,
                "error": f"Could not queue GPU: {type(exc).__name__}: {exc}",
                "user_id": tg_user_id,
            })
            return

    async def _kick(data: dict, job_id: str, user):
        try:
            await asyncio.to_thread(accept_job, data, job_id, user)
        except Exception as exc:
            traceback.print_exc()
            try:
                await asyncio.to_thread(write_job, job_id, {
                    "status": "error", "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "user_id": int(user["id"]) if user else None,
                })
            except Exception:
                pass

    @web_app.post("/")
    async def run_generate(request: Request, background: BackgroundTasks, x_api_key: str | None = Header(None)):
        key = os.environ.get("MODAL_API_KEY") or API_KEY
        if key and x_api_key != key:
            raise HTTPException(401, "Unauthorized")
        try:
            data = json.loads((await request.body()).decode("utf-8") or "{}")
        except Exception:
            raise HTTPException(400, "Invalid JSON")
        if not isinstance(data, dict):
            raise HTTPException(400, "Invalid JSON")
        user = None
        if bot_token():
            try:
                user = validate_init_data(data.get("initData") or data.get("init_data") or "")
            except ValueError as e:
                raise HTTPException(402, str(e))
        client_req = str(data.get("client_req") or "")
        if not REQ_RE.fullmatch(client_req):
            client_req = ""
        if client_req and client_req in recent_reqs:
            job_id = recent_reqs[client_req]
            return {"success": True, "status": "queued", "job_id": job_id, "poll_url": f"/job/{job_id}", "reused": True}
        job_id = str(uuid.uuid4())
        if client_req:
            recent_reqs[client_req] = job_id
            if len(recent_reqs) > 300:
                recent_reqs.pop(next(iter(recent_reqs)))
        background.add_task(_kick, data, job_id, user)
        return {"success": True, "status": "queued", "job_id": job_id, "poll_url": f"/job/{job_id}"}

    return web_app
