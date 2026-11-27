@app.cls(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: models_volume},
    scaledown_window=15,
    enable_memory_snapshot=False,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    startup_timeout=420,
    max_containers=1,
    retries=0,
)
class SF3DModel:
    @modal.enter()
    def load_model(self):
        self.ready = False
        self.fatal_error = None
        print(
            "Placement info:",
            {
                "MODAL_REGION": os.environ.get("MODAL_REGION"),
                "MODAL_CLOUD_PROVIDER": os.environ.get("MODAL_CLOUD_PROVIDER"),
            },
        )
        try:
            self._load_weights()
            self.ready = True
            print("L4 live, model ready")
        except Exception as exc:
            traceback.print_exc()
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.ready = False

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
            fatal = getattr(self, "fatal_error", None) or "GPU model not ready"
            if job_id:
                write_job(job_id, {"status": "error", "job_id": job_id, "error": fatal, "user_id": item.get("user_id")})
            return {"success": False, "status": "error", "error": fatal, "job_id": job_id}
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
            return {"success": False, "status": "error", "error": err, "job_id": job_id}
        finally:
            torch.cuda.empty_cache()
            
