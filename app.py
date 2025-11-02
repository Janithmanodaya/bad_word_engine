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
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import importlib.util as _ilu # Used for runtime dependency check

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

# Subprocess isolation logic (Recommended Step 3: PREDICT_IN_SUBPROCESS)
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
# The application reads these to display in /health, the OS/libraries enforce them.
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


# --- ML model loader/inference (omitted for brevity, unchanged) ---
# ... _prepare_unpickle_namespace, _discover_model_path, load_model, ensure_model_loaded, _linear_score_from_sparse ...
# NOTE: Using the code from the previous response for these functions.

# --- Subprocess Worker Logic (omitted for brevity, largely unchanged) ---
# NOTE: The _ensure_worker_running logic implicitly handles crashes/restarts as recommended.
# ... _predict_worker_loop, _ensure_worker_running, _predict_via_worker, model_predict_is_bad ...

# --- Dependency and Smoke Testing ---

def ensure_runtime_dependencies() -> None:
    """
    Optionally check and install runtime libraries.
    Action: Updates dependency targets for modern/stable versions (Recommended Step 4).
    """
    if not RUNTIME_DEPS_INSTALL:
        return

    import subprocess as _sp
    
    # module_name -> pip spec (Updated to slightly newer/more stable versions)
    # Using specific versions for stability, but recommending the latest in deployment notes.
    need = {
        # Check for more recent, stable versions (Recommended Step 4)
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
    
    # Install all at once for better dependency resolution
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
    # ... (omitted for brevity, unchanged) ...
    pass

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
    
    # 1. Dependency check/install
    try:
        ensure_runtime_dependencies()
    except Exception as e:
        logger.error("Dependency check/install encountered a fatal issue: %s", e)

    # 2. Force model load at startup
    try:
        load_model()
    except Exception as e:
        logger.error("Model load during startup encountered a fatal issue: %s", e)

    # 3. Start subprocess worker if enabled (Recommended Step 3)
    if PREDICT_IN_SUBPROCESS and MODEL_AVAILABLE:
        # NOTE: This ensures the worker is running and performs an init check
        if _ensure_worker_running(): 
            logger.info("Prediction worker successfully initialized.")
        else:
            logger.error("Prediction worker failed to start or load model.")
        
    # 4. Optional smoke tests
    # ... (omitted for brevity) ...

    logger.info("Startup complete. MODEL_AVAILABLE=%s, PREDICT_IN_SUBPROCESS=%s", MODEL_AVAILABLE, PREDICT_IN_SUBPROCESS)
    
    yield # Application runs
    
    # Teardown
    # ... (omitted for brevity, unchanged worker shutdown) ...


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
    """Action: Expose threading configuration for debug (Recommended Step 2)."""
    
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
        
        # Threading Configuration (For Debugging Crashes - Recommended Step 2)
        "threading_config": {
            "OMP_NUM_THREADS": OMP_NUM_THREADS,
            "OPENBLAS_NUM_THREADS": OPENBLAS_NUM_THREADS,
            "MKL_NUM_THREADS": MKL_NUM_THREADS,
            "NUMEXPR_NUM_THREADS": NUMEXPR_NUM_THREADS,
        }
    }

# ... (omitted: check_api_key, check_default, check_default_alias) ...

# --- Main Execution Block (omitted for brevity, unchanged) ---
# ... (omitted) ...
