# Bad-words detection service (FastAPI) — ML-only version
# Optimized for low-resource servers (≈0.2 vCPU, 256MB RAM).
# - Lightweight inference using sparse linear scorer (pure Python)
# - Optional subprocess isolation to contain native crashes
# - Tunable startup behavior to avoid heavy runtime installations/smoke tests

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Logging setup
def _get_log_level() -> int:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)

logging.basicConfig(level=_get_log_level(), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("badwords_service_ml")

# Globals (ML)
MODEL_AVAILABLE: bool = False
MODEL_PATH: str = ""
ML_DISABLED: bool = os.getenv("ML_DISABLE", "").strip().lower() in {"1", "true", "yes"}
# Enable crash-isolated prediction via env var.
# Default OFF to avoid heavy fork+reload overhead on tiny servers (0.2 vCPU / 256–512MB RAM).
PREDICT_IN_SUBPROCESS: bool = os.getenv("PREDICT_SUBPROCESS", "0").strip().lower() in {"1", "true", "yes"}
# Reduce startup work on constrained hosts
LOW_RESOURCE_MODE: bool = os.getenv("LOW_RESOURCE_MODE", "1").strip().lower() in {"1", "true", "yes"}
RUNTIME_DEPS_INSTALL: bool = os.getenv("RUNTIME_DEPS_INSTALL", "0").strip().lower() in {"1", "true", "yes"}
_model_bundle = None  # dict with keys: "vec_char", "vec_word", "classifier"

# Subprocess prediction worker globals
_predict_req_q = None
_predict_resp_q = None
_predict_proc = None

def _preview(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[:n] + "...(truncated)"

# --- ML model loader/inference ---

def _prepare_unpickle_namespace() -> None:
    """
    Ensure joblib can unpickle vectorizers that reference normalize_text
    from train_gemma_badwords by providing a stub module in sys.modules.
    """
    try:
        import types, sys as _sys, re as _re
        if "train_gemma_badwords" not in _sys.modules:
            mod = types.ModuleType("train_gemma_badwords")
            # Minimal compatible normalize_text used during training
            def _norm(t: str) -> str:
                try:
                    s = str(t).lower().replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
                except Exception:
                    s = ""
                # collapse >2 repeats to two and remove separators between ASCII letters
                s = _re.sub(r"(.)\\1{2,}", r"\1\1", s)
                s = _re.sub(r"(?:[\\W_])+(?=[A-Za-z])", "", s)
                try:
                    from unidecode import unidecode  # type: ignore
                    s = unidecode(s)
                except Exception:
                    pass
                return s
            mod.normalize_text = _norm  # type: ignore[attr-defined]
            _sys.modules["train_gemma_badwords"] = mod
    except Exception as e:
        logger.warning("Failed to prepare unpickle namespace: %s", e)

def _discover_model_path() -> Optional[str]:
    """
    Try multiple locations to find model.joblib:
    - $MODEL_DIR/model.joblib
    - ./model/model.joblib
    - ./outputs/badwords-ml/model.joblib
    - Path declared in ./model/inference_card.json (model_path/model.joblib)
    """
    import json

    candidates = []
    env_dir = os.getenv("MODEL_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.join(env_dir, "model.joblib"))
    candidates.append(os.path.join("model", "model.joblib"))
    candidates.append(os.path.join("outputs", "badwords-ml", "model.joblib"))

    try:
        card_path = os.path.join("model", "inference_card.json")
        if os.path.exists(card_path):
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
            mdir = str(card.get("model_path", "")).strip()
            if mdir:
                candidates.append(os.path.join(mdir, "model.joblib"))
    except Exception as e:
        logger.warning("Failed to read inference_card.json: %s", e)

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    for p in uniq:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    logger.error("Model discovery failed. Checked: %s", ", ".join(uniq))
    return None

def load_model() -> None:
    global MODEL_AVAILABLE, MODEL_PATH, _model_bundle, ML_DISABLED
    MODEL_AVAILABLE = False
    _model_bundle = None

    if ML_DISABLED:
        logger.info("ML model loading disabled via ML_DISABLE env. Service will return 503 for /check.")
        return

    model_file = _discover_model_path()
    if not model_file:
        logger.error("Model file not found; ML detector disabled.")
        return

    MODEL_PATH = model_file
    try:
        _prepare_unpickle_namespace()
        from joblib import load as joblib_load  # type: ignore
        _model_bundle = joblib_load(model_file)
        for key in ("vec_char", "vec_word", "classifier"):
            if key not in _model_bundle:
                raise ValueError(f"Model bundle missing '{key}'")
        MODEL_AVAILABLE = True
        logger.info("ML model loaded from %s", model_file)
    except Exception as e:
        logger.error("Failed to load ML model from %s: %s", model_file, e)
        MODEL_AVAILABLE = False
        _model_bundle = None

def _linear_score_from_sparse(Xc, Xw, clf) -> float:
    """
    Compute linear decision score for a single sample by iterating CSR non-zeros.
    Avoids dense ops and BLAS to prevent segfaults in minimal containers.
    """
    import numpy as _np  # type: ignore

    coef = _np.asarray(getattr(clf, "coef_", None))
    intercept = _np.asarray(getattr(clf, "intercept_", 0.0))
    if coef is None or coef.size == 0:
        raise ValueError("Classifier has no coef_")
    if coef.ndim == 2 and coef.shape[0] == 1:
        w = coef[0]
    else:
        w = coef.ravel()
    b = float(intercept.ravel()[0] if intercept.size else 0.0)

    score = b

    def _accum_csr(csr_mat, w_offset: int = 0) -> float:
        csr = csr_mat.tocsr()
        start, end = csr.indptr[0], csr.indptr[1]
        idx = csr.indices[start:end]
        data = csr.data[start:end]
        s = 0.0
        for j, v in zip(idx, data):
            jj = int(j) + w_offset
            if 0 <= jj < w.shape[0]:
                s += float(v) * float(w[jj])
        return s

    score += _accum_csr(Xc, 0)
    score += _accum_csr(Xw, Xc.shape[1])

    return float(score)

def _predict_worker_loop(model_path: str, q_in, q_out):
    """
    Persistent child-process worker:
    - Loads model bundle once
    - Processes multiple predict requests from q_in and responds on q_out
    - Input items are dicts: {"id": str, "text": str}
    - Output items are dicts: {"id": str, "ok": bool, "result": Optional[bool], "error": Optional[str]}
    """
    try:
        _prepare_unpickle_namespace()
        from joblib import load as joblib_load  # type: ignore
        bundle = joblib_load(model_path)
        vec_char = bundle["vec_char"]
        vec_word = bundle["vec_word"]
        clf = bundle["classifier"]
    except Exception as e:
        # Fatal error: report once and exit
        try:
            q_out.put({"id": "__init__", "ok": False, "result": None, "error": f"load_error: {e}"})
        except Exception:
            pass
        return

    import time as _time

    while True:
        try:
            item = q_in.get()
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            # small backoff to avoid busy loop on unexpected errors
            _time.sleep(0.05)
            continue

        if not isinstance(item, dict):
            continue

        if item.get("cmd") == "shutdown":
            break

        req_id = item.get("id")
        text = item.get("text", "")

        try:
            Xc = vec_char.transform([text])
            Xw = vec_word.transform([text])
            score = _linear_score_from_sparse(Xc, Xw, clf)
            res = bool(score > 0.0)
            q_out.put({"id": req_id, "ok": True, "result": res, "error": None})
        except Exception as e:
            try:
                q_out.put({"id": req_id, "ok": False, "result": None, "error": str(e)})
            except Exception:
                pass

def _ensure_worker_running() -> bool:
    """
    Ensure the persistent prediction worker is running. Returns True if running.
    """
    global _predict_req_q, _predict_resp_q, _predict_proc
    if not PREDICT_IN_SUBPROCESS or not MODEL_PATH:
        return False
    try:
        import multiprocessing as mp
        if _predict_proc is not None:
            if _predict_proc.is_alive():
                return True
            else:
                try:
                    _predict_proc.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                _predict_proc = None
                _predict_req_q = None
                _predict_resp_q = None

        ctx = mp.get_context("spawn")
        _predict_req_q = ctx.Queue()
        _predict_resp_q = ctx.Queue()
        _predict_proc = ctx.Process(target=_predict_worker_loop, args=(MODEL_PATH, _predict_req_q, _predict_resp_q))
        _predict_proc.daemon = True
        _predict_proc.start()
        return True
    except Exception as e:
        logger.warning("Failed to start prediction worker: %s", e)
        _predict_req_q = None
        _predict_resp_q = None
        _predict_proc = None
        return False


def _predict_via_worker(text: str) -> Optional[bool]:
    """
    Send a prediction request to the persistent worker and wait for response or timeout.
    """
    if not _ensure_worker_running():
        return None

    import time
    import uuid

    req_id = str(uuid.uuid4())
    try:
        _predict_req_q.put({"id": req_id, "text": text})
    except Exception as e:
        logger.warning("Failed to enqueue request to worker: %s", e)
        return None

    timeout_sec = float(os.getenv("PREDICT_TIMEOUT_SEC", "10"))
    deadline = time.monotonic() + timeout_sec
    # Poll the response queue for our id
    while time.monotonic() < deadline:
        try:
            msg = _predict_resp_q.get(timeout=0.1)
        except Exception:
            continue
        if isinstance(msg, dict) and msg.get("id") == req_id:
            if msg.get("ok"):
                return bool(msg.get("result"))
            else:
                logger.warning("Prediction worker error: %s", msg.get("error"))
                return None
        # If it's the init error
        if isinstance(msg, dict) and msg.get("id") == "__init__":
            logger.warning("Prediction worker init error: %s", msg.get("error"))
            return None

    logger.warning("Prediction subprocess timed out")
    return None


def model_predict_is_bad(text: str) -> Optional[bool]:
    """
    Return True if model predicts 'bad', False if 'clean', or None if model unavailable or error.

    Implementation note:
    - Avoid dense ops that may rely on BLAS in minimal containers.
    - Compute a linear decision score manually from sparse features.
    - Prefer a persistent subprocess worker to isolate native crashes and avoid repeated model loads.
    """
    if ML_DISABLED or not MODEL_AVAILABLE or (MODEL_PATH == "" and _model_bundle is None):
        return None

    if PREDICT_IN_SUBPROCESS and MODEL_PATH:
        return _predict_via_worker(text)

    # Fallback: in-process safe path (still pure-Python scorer)
    if _model_bundle is None:
        return None
    try:
        vec_char = _model_bundle["vec_char"]
        vec_word = _model_bundle["vec_word"]
        clf = _model_bundle["classifier"]

        Xc = vec_char.transform([text])
        Xw = vec_word.transform([text])

        score = _linear_score_from_sparse(Xc, Xw, clf)
        return bool(score > 0.0)
    except Exception as e:
        logger.warning("Model inference failed: %s", e)
        return None

def ensure_runtime_dependencies() -> None:
    """
    Optionally check and install runtime libraries.
    Disabled by default for low-resource servers to avoid pip activity at startup.
    Enable with env RUNTIME_DEPS_INSTALL=1 if absolutely necessary.
    """
    if not RUNTIME_DEPS_INSTALL:
        return

    import importlib.util as _ilu
    import subprocess as _sp
    import sys as _sys

    # module_name -> pip spec
    need = {
        "numpy": "numpy==1.26.4",
        "scipy": "scipy==1.10.1",
        "sklearn": "scikit-learn==1.6.1",
        "joblib": "joblib>=1.3.2",
        "unidecode": "Unidecode>=1.3.8",
    }

    missing = [m for m in need.keys() if _ilu.find_spec(m) is None]
    if not missing:
        return

    logger.info("Runtime deps missing (%s). Attempting install...", ", ".join(missing))
    for mod in missing:
        spec = need[mod]
        try:
            _sp.check_call([_sys.executable, "-m", "pip", "install", "--no-cache-dir", spec])
            logger.info("Installed %s", spec)
        except Exception as e:
            logger.warning("Failed to install %s: %s", spec, e)

    # Final report
    still_missing = [m for m in need.keys() if _ilu.find_spec(m) is None]
    if still_missing:
        logger.warning("Some dependencies are still missing after install: %s", ", ".join(still_missing))
    else:
        logger.info("All runtime dependencies present.")


def run_startup_smoke_tests() -> None:
    """
    Run quick bilingual smoke tests (Sinhala + English) to verify end-to-end inference.
    Logs results; does not raise.
    """
    samples = [
        ("si", "කෙසේද ඉතිං?"),  # neutral Sinhala
        ("si", "අපහාසකාරී වචන තියෙනවාද?"),  # Sinhala query (may be neutral)
        ("en", "You are an idiot."),  # likely toxic
        ("en", "Have a wonderful day!"),  # neutral
    ]
    try:
        ok = 0
        for lang, text in samples:
            pred = model_predict_is_bad(text)
            if pred is None:
                logger.info("[smoke] lang=%s text_preview=%s -> result=UNAVAILABLE", lang, _preview(text))
            else:
                logger.info("[smoke] lang=%s text_preview=%s -> found=%s", lang, _preview(text), bool(pred))
                ok += 1
        logger.info("[smoke] Completed %d/%d checks.", ok, len(samples))
    except Exception as e:
        logger.warning("Smoke tests failed to run: %s", e)

# --- FastAPI ---

class CheckRequest(BaseModel):
    text: str
    advanced: bool = False  # kept for backward-compatibility (ignored for now)

class DefaultResponse(BaseModel):
    found: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Minimal startup on low-resource servers
    try:
        ensure_runtime_dependencies()
    except Exception as e:
        logger.warning("Dependency check/install encountered an issue: %s", e)

    # Load ML model
    load_model()

    # Optionally run smoke tests (disabled by default in low-resource mode)
    if not LOW_RESOURCE_MODE:
        try:
            run_startup_smoke_tests()
        except Exception as e:
            logger.warning("Startup smoke tests encountered an issue: %s", e)

    logger.info("Startup complete. MODEL_AVAILABLE=%s", MODEL_AVAILABLE)
    yield
    # No teardown required.

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "bad-words service (ml-only)"}

@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s for browsers requesting a favicon
    return Response(status_code=204)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_AVAILABLE,
        "model_path": MODEL_PATH,
        "predict_subprocess": PREDICT_IN_SUBPROCESS,
        "low_resource_mode": LOW_RESOURCE_MODE,
        "ml_disabled": ML_DISABLED,
    }

def check_api_key(req: Request, expected_key: str) -> None:
    key = req.headers.get("X-API-Key", "")
    if expected_key and key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/check")
async def check_default(req: Request, payload: CheckRequest):
    api_key = os.getenv("API_KEY", "")
    check_api_key(req, api_key)

    text_preview = _preview(payload.text)
    logger.info("Incoming /check (ml-only): text_preview=%s", text_preview)

    res = model_predict_is_bad(payload.text)
    if res is None:
        # ML-only service: if model not available or error during inference, return 503
        raise HTTPException(status_code=503, detail="Model unavailable")

    return DefaultResponse(found=bool(res))

# Backward/compat alias for clients calling /check/check
@app.post("/check/check")
async def check_default_alias(req: Request, payload: CheckRequest):
    return await check_default(req, payload)

# __main__
if __name__ == "__main__":
    import uvicorn
    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env and port_env.isdigit() else 8000
    except Exception:
        port = 8000
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)
