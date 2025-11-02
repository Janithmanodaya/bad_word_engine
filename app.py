# Bad-words detection service (FastAPI) — ML-only version
# Optimized for low-resource servers (≈0.2 vCPU, 256MB RAM).
# Incorporating Low-Resource Stability Recommendations (Threading, Subprocess Isolation)

import os
import logging
import sys
import json
import re
import types
import time
import uuid
import multiprocessing as mp
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import importlib.util as _ilu
import subprocess as _sp

# --- Configuration & Logging Setup ---

# Logging setup
def _get_log_level() -> int:
    """Determine log level from environment variable."""
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)

logging.basicConfig(level=_get_log_level(), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("badwords_service_ml")

# Globals (ML & Environment)
MODEL_AVAILABLE: bool = False
MODEL_PATH: str = ""
ML_DISABLED: bool = os.getenv("ML_DISABLE", "").strip().lower() in {"1", "true", "yes"}

# Subprocess isolation logic (Recommended Step 3)
PREDICT_IN_SUBPROCESS: bool = (
    os.getenv("PREDICT_SUBPROCESS", os.getenv("PREDICT_IN_SUBPROCESS", "0"))
    .strip()
    .lower()
    in {"1", "true", "yes"}
)

# Startup optimization flags
LOW_RESOURCE_MODE: bool = os.getenv("LOW_RESOURCE_MODE", "1").strip().lower() in {"1", "true", "yes"}
RUNTIME_DEPS_INSTALL: bool = os.getenv("RUNTIME_DEPS_INSTALL", "0").strip().lower() in {"1", "true", "yes"}
MAX_TEXT_LEN: int = int(os.getenv("MAX_TEXT_LEN", "4000"))
PREDICT_TIMEOUT_SEC: float = float(os.getenv("PREDICT_TIMEOUT_SEC", "10"))

# Threading control variables (Recommended Step 2)
OMP_NUM_THREADS: str = os.getenv("OMP_NUM_THREADS", "N/A")
OPENBLAS_NUM_THREADS: str = os.getenv("OPENBLAS_NUM_THREADS", "N/A")
MKL_NUM_THREADS: str = os.getenv("MKL_NUM_THREADS", "N/A")
NUMEXPR_NUM_THREADS: str = os.getenv("NUMEXPR_NUM_THREADS", "N/A")

# ML state
_model_bundle: Optional[Dict[str, Any]] = None

# Subprocess prediction worker globals
_predict_req_q: Optional[mp.Queue] = None
_predict_resp_q: Optional[mp.Queue] = None
_predict_proc: Optional[mp.Process] = None


def _preview(text: str, n: int = 140) -> str:
    """Return a truncated text preview."""
    return text if len(text) <= n else text[:n] + "...(truncated)"


# --- ML model loader/inference ---

def _prepare_unpickle_namespace() -> None:
    """
    Ensure joblib can unpickle vectorizers that reference normalize_text
    by providing a stub module in sys.modules, typically 'train_gemma_badwords'.
    """
    try:
        if "train_gemma_badwords" not in sys.modules:
            mod = types.ModuleType("train_gemma_badwords")

            def _norm(t: str) -> str:
                s = str(t).lower().replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
                s = re.sub(r"(.)\1{2,}", r"\1\1", s)
                s = re.sub(r"(?:[\W_])+(?=[A-Za-z])", "", s)
                try:
                    from unidecode import unidecode  # type: ignore
                    s = unidecode(s)
                except Exception:
                    pass
                return s

            mod.normalize_text = _norm  # type: ignore[attr-defined]
            sys.modules["train_gemma_badwords"] = mod
    except Exception as e:
        logger.warning("Failed to prepare unpickle namespace: %s", e)

def _discover_model_path() -> Optional[str]:
    """Try multiple locations to find model.joblib."""
    
    candidates: List[str] = []

    # 1. Explicit env overrides
    env_file = os.getenv("MODEL_FILE", "").strip()
    if env_file:
        candidates.append(env_file)

    env_dir = os.getenv("MODEL_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.join(env_dir, "model.joblib"))

    # 2. Relative to CWD
    candidates.append(os.path.join("model", "model.joblib"))
    candidates.append(os.path.join("outputs", "badwords-ml", "model.joblib"))

    # 3. Relative to this file's directory
    try:
        here = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))
        candidates.append(os.path.join(here, "model", "model.joblib"))
        candidates.append(os.path.join(here, "scripts", "model", "model.joblib"))
        parent = os.path.dirname(here)
        candidates.append(os.path.join(parent, "model", "model.joblib"))
    except Exception:
        pass

    # 4. From inference card
    try:
        card_path = os.path.join("model", "inference_card.json")
        if os.path.exists(card_path):
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
            mdir = str(card.get("model_path", "")).strip()
            if mdir:
                candidates.append(os.path.join(mdir, "model.joblib"))
                try:
                     here = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))
                     candidates.append(os.path.join(here, mdir, "model.joblib"))
                except Exception:
                     pass
    except Exception as e:
        logger.warning("Failed to read inference_card.json: %s", e)

    # Deduplicate and check for existence
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen and p:
            seen.add(p)
            uniq.append(p)

    for p in uniq:
        try:
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p) and os.path.isfile(abs_p):
                logger.info("Discovered model at: %s", abs_p)
                return abs_p
        except Exception:
            continue

    logger.error("Model discovery failed. Checked: %s", ", ".join(uniq))
    return None

def load_model() -> None:
    """Loads the ML model bundle into global state."""
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
        
        required_keys = ("vec_char", "vec_word", "classifier")
        for key in required_keys:
            if key not in _model_bundle:
                raise ValueError(f"Model bundle missing '{key}'")

        MODEL_AVAILABLE = True
        logger.info("ML model loaded from %s", model_file)
    except Exception as e:
        logger.error("Failed to load ML model from %s: %s", model_file, e)
        MODEL_AVAILABLE = False
        _model_bundle = None


def ensure_model_loaded() -> bool:
    """
    Lazily load the ML model on first use. Returns True if model is available after this call.
    """
    if ML_DISABLED:
        return False
    if MODEL_AVAILABLE and (_model_bundle is not None or (PREDICT_IN_SUBPROCESS and MODEL_PATH)):
        return True
    
    if not MODEL_AVAILABLE:
        load_model()
    
    return MODEL_AVAILABLE

def _linear_score_from_sparse(Xc, Xw, clf) -> float:
    """
    Compute linear decision score for a single sample by iterating CSR non-zeros.
    Avoids dense ops and BLAS to prevent segfaults in minimal containers.
    """
    try:
        import numpy as _np  # type: ignore
    except ImportError:
        raise RuntimeError("Numpy is required for linear scoring.")

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
        """Accumulate score from a single sparse matrix (CSR format)."""
        csr = csr_mat.tocsr()
        if csr.shape[0] != 1:
             raise ValueError("Input matrix is not a single sample.")
             
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

# --- Subprocess Worker Logic ---

def _predict_worker_loop(model_path: str, q_in: mp.Queue, q_out: mp.Queue):
    """
    Persistent child-process worker: loads model, processes requests, and responds.
    """
    try:
        _prepare_unpickle_namespace()
        from joblib import load as joblib_load
        bundle = joblib_load(model_path)
        vec_char = bundle["vec_char"]
        vec_word = bundle["vec_word"]
        clf = bundle["classifier"]
        
        if not all(k in bundle for k in ("vec_char", "vec_word", "classifier")):
            raise ValueError("Model bundle is incomplete.")
            
    except Exception as e:
        try:
            q_out.put({"id": "__init__", "ok": False, "result": None, "error": f"load_error: {e}"})
        except Exception:
            pass
        return

    while True:
        try:
            item = q_in.get()
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            time.sleep(0.05)
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
        if _predict_proc is not None:
            if _predict_proc.is_alive():
                return True
            else:
                logger.warning("Prediction worker died (exit code: %s). Restarting.", _predict_proc.exitcode)
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
        
        logger.info("Prediction worker started (PID: %d)", _predict_proc.pid)
        
        # Check for immediate init error from worker
        try:
            msg = _predict_resp_q.get(timeout=1.0) 
            if isinstance(msg, dict) and msg.get("id") == "__init__" and not msg.get("ok"):
                logger.error("Prediction worker failed to initialize: %s", msg.get("error"))
                _predict_proc.terminate()
                _predict_proc.join()
                _predict_proc = None
                return False
            elif isinstance(msg, dict) and msg.get("id") == "__init__":
                pass
            else:
                _predict_resp_q.put(msg)
                
        except (mp.queues.Empty, Exception):
            pass

        return True
        
    except Exception as e:
        logger.error("Failed to start prediction worker: %s", e)
        _predict_req_q = None
        _predict_resp_q = None
        _predict_proc = None
        return False


def _predict_via_worker(text: str) -> Optional[bool]:
    """
    Send a prediction request to the persistent worker and wait for response or timeout.
    """
    global _predict_req_q, _predict_resp_q, _predict_proc
    
    if not _ensure_worker_running() or _predict_req_q is None or _predict_resp_q is None:
        logger.warning("Worker queues are not initialized after worker check.")
        return None
        
    req_id = str(uuid.uuid4())
    try:
        _predict_req_q.put({"id": req_id, "text": text})
    except Exception as e:
        logger.warning("Failed to enqueue request to worker: %s", e)
        return None

    deadline = time.monotonic() + PREDICT_TIMEOUT_SEC
    
    while time.monotonic() < deadline:
        timeout = max(0.01, min(0.1, deadline - time.monotonic()))
        try:
            msg = _predict_resp_q.get(timeout=timeout)
        except mp.queues.Empty:
            continue
        except Exception:
            continue
            
        if isinstance(msg, dict) and msg.get("id") == req_id:
            if msg.get("ok"):
                return bool(msg.get("result"))
            else:
                logger.warning("Prediction worker error for ID %s: %s", req_id, msg.get("error"))
                return None
        
        if isinstance(msg, dict) and msg.get("id") == "__init__":
            logger.warning("Prediction worker init error received mid-request: %s", msg.get("error"))
            return None


def model_predict_is_bad(text: str) -> Optional[bool]:
    """
    Return True if model predicts 'bad', False if 'clean', or None if model unavailable or error.
    """
    if ML_DISABLED:
        return None

    if not ensure_model_loaded():
        return None

    if PREDICT_IN_SUBPROCESS and MODEL_PATH:
        return _predict_via_worker(text)

    # Fallback: in-process safe path
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
        logger.warning("Model inference failed (in-process fallback): %s", e)
        return None

# --- Dependency and Smoke Testing ---

def ensure_runtime_dependencies() -> None:
    """
    Optionally check and install runtime libraries.
    Action: Updates dependency targets for modern/stable versions.
    """
    if not RUNTIME_DEPS_INSTALL:
        return

    # module_name -> pip spec (Updated to slightly newer/more stable versions)
    need = {
        "numpy": "numpy>=1.26.4", 
        "scipy": "scipy>=1.11.4", 
        "sklearn": "scikit-learn>=1.4.2", 
        "joblib": "joblib>=1.3.2",
        "unidecode": "Unidecode>=1.3.8",
    }
    
    missing = [m for m in need.keys() if _ilu.find_spec(m) is None]
    
    if not missing:
        return

    logger.info("Runtime deps missing (%s). Attempting install...", ", ".join(missing))
    
    install_specs = [need[m] for m in missing]
    try:
        _sp.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + install_specs)
        logger.info("Installed dependencies: %s", ", ".join(install_specs))
    except Exception as e:
        logger.error("Failed to install dependencies: %s", e)

    # Final report
    still_missing = [m for m in need.keys() if _ilu.find_spec(m) is None]
    if still_missing:
        logger.error("CRITICAL: Some dependencies are still missing after install: %s", ", ".join(still_missing))
    else:
        logger.info("All runtime dependencies present.")


def run_startup_smoke_tests() -> None:
    """
    Run quick bilingual smoke tests to verify end-to-end inference.
    """
    if ML_DISABLED or not MODEL_AVAILABLE:
        logger.info("[smoke] Skipping smoke tests: Model unavailable.")
        return
        
    samples = [
        ("si", "කෙසේද ඉතිං?"),            
        ("si", "අපහාසකාරී වචන තියෙනවාද?"), 
        ("en", "You are an idiot."),      
        ("en", "Have a wonderful day!"),  
    ]
    try:
        ok = 0
        for lang, text in samples:
            pred = model_predict_is_bad(text)
            if pred is None:
                logger.warning("[smoke] lang=%s text_preview=%s -> result=ERROR/TIMEOUT", lang, _preview(text))
            else:
                logger.info("[smoke] lang=%s text_preview=%s -> found=%s", lang, _preview(text), bool(pred))
                ok += 1
        logger.info("[smoke] Completed %d/%d checks.", ok, len(samples))
    except Exception as e:
        logger.warning("Smoke tests failed to run: %s", e)

# --- FastAPI App Definition ---

class CheckRequest(BaseModel):
    """Schema for the POST /check request body."""
    text: str
    advanced: bool = False

class DefaultResponse(BaseModel):
    """Schema for the POST /check response body."""
    found: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan event handler (startup/shutdown)."""
    
    try:
        ensure_runtime_dependencies()
    except Exception as e:
        logger.error("Dependency check/install encountered a fatal issue: %s", e)

    try:
        load_model()
    except Exception as e:
        logger.error("Model load during startup encountered a fatal issue: %s", e)

    if PREDICT_IN_SUBPROCESS and MODEL_AVAILABLE:
        if _ensure_worker_running(): 
            logger.info("Prediction worker successfully initialized.")
        else:
            logger.error("Prediction worker failed to start or load model.")
        
    try:
        if not LOW_RESOURCE_MODE:
            run_startup_smoke_tests()
    except Exception as e:
        logger.warning("Startup smoke tests encountered an issue: %s", e)

    logger.info("Startup complete. MODEL_AVAILABLE=%s, PREDICT_IN_SUBPROCESS=%s", MODEL_AVAILABLE, PREDICT_IN_SUBPROCESS)
    
    yield
    
    # Teardown: Shutdown the worker process gracefully
    if _predict_proc is not None:
        try:
            logger.info("Shutting down prediction worker...")
            if _predict_req_q:
                 _predict_req_q.put({"cmd": "shutdown"})
            _predict_proc.join(timeout=5)
            if _predict_proc.is_alive():
                _predict_proc.terminate()
                _predict_proc.join()
            logger.info("Prediction worker shut down.")
        except Exception as e:
            logger.warning("Error during worker shutdown: %s", e)


app = FastAPI(
    title="Bad-Words Detection Service (ML-only)",
    description="Optimized bad-words detection using a sparse linear model.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "message": "bad-words service (ml-only)"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    """Expose model status and threading configuration for debug."""
    
    worker_status = "N/A"
    if PREDICT_IN_SUBPROCESS:
        if _predict_proc and _predict_proc.is_alive():
             worker_status = "Alive"
        elif _predict_proc:
             worker_status = f"Dead (Code: {_predict_proc.exitcode})"
        else:
             worker_status = "Uninitialized"

    return {
        "status": "ok" if MODEL_AVAILABLE else "degraded",
        "model_available": MODEL_AVAILABLE,
        "model_path": MODEL_PATH,
        "ml_disabled": ML_DISABLED,
        "predict_subprocess": PREDICT_IN_SUBPROCESS,
        "worker_status": worker_status,
        "low_resource_mode": LOW_RESOURCE_MODE,
        
        "threading_config": {
            "OMP_NUM_THREADS": OMP_NUM_THREADS,
            "OPENBLAS_NUM_THREADS": OPENBLAS_NUM_THREADS,
            "MKL_NUM_THREADS": MKL_NUM_THREADS,
            "NUMEXPR_NUM_THREADS": NUMEXPR_NUM_THREADS,
        }
    }

def check_api_key(req: Request, expected_key: str) -> None:
    """Helper to check the X-API-Key header."""
    key = req.headers.get("X-API-Key", "")
    if expected_key and key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/check", response_model=DefaultResponse)
async def check_default(req: Request, payload: CheckRequest):
    """Main endpoint to check for bad words."""
    api_key = os.getenv("API_KEY", "")
    check_api_key(req, api_key)

    text_in = payload.text or ""
    if len(text_in) > MAX_TEXT_LEN:
        logger.warning("Text too long (%d > %d). Truncating.", len(text_in), MAX_TEXT_LEN)
        text_in = text_in[:MAX_TEXT_LEN]

    text_preview = _preview(text_in)
    logger.info("Incoming /check: preview=%s len=%d", text_preview, len(text_in))

    res = model_predict_is_bad(text_in)
    
    if res is None:
        raise HTTPException(status_code=503, detail="Model unavailable or prediction failed/timed out")

    return DefaultResponse(found=bool(res))

@app.post("/check/check", response_model=DefaultResponse, include_in_schema=False)
async def check_default_alias(req: Request, payload: CheckRequest):
    """Alias for /check for backward compatibility."""
    return await check_default(req, payload)

# --- Main Execution Block ---

if __name__ == "__main__":
    import uvicorn
    
    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env and port_env.isdigit() else 8000
    except ValueError:
        logger.warning("Invalid PORT environment variable. Falling back to 8000.")
        port = 8000
        
    host = os.getenv("HOST", "0.0.0.0")
    
    # Run uvicorn (expects 'module:app_object')
    uvicorn.run("app:app", host=host, port=port, reload=False)
