# Bad-words detection service (FastAPI) — ML-only version
# This rewrite removes all lexicon/fuzzy/substring systems and relies solely on the bundled ML model.
# The model is loaded from model/model.joblib (or discovered paths), and inference is performed with
# a safe, sparse, Python-level linear scorer to avoid native BLAS/segfaults in minimal containers.

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
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
_model_bundle = None  # dict with keys: "vec_char", "vec_word", "classifier"

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
                s = _re.sub(r"(.)\\1{2,}", r"\\1\\1", s)
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
    global MODEL_AVAILABLE, MODEL_PATH, _model_bundle
    MODEL_AVAILABLE = False
    _model_bundle = None

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

def model_predict_is_bad(text: str) -> Optional[bool]:
    """
    Return True if model predicts 'bad', False if 'clean', or None if model unavailable/error.
    Uses a safe manual scorer for linear models.
    """
    if not MODEL_AVAILABLE or _model_bundle is None:
        return None
    try:
        vec_char = _model_bundle["vec_char"]
        vec_word = _model_bundle["vec_word"]
        clf = _model_bundle["classifier"]

        Xc = vec_char.transform([text])
        Xw = vec_word.transform([text])

        # Prefer linear scorer when coef_/intercept_ are available
        if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
            score = _linear_score_from_sparse(Xc, Xw, clf)
            return bool(score > 0.0)

        # Fallback (may use native code depending on estimator)
        # Concatenate sparse with dimensions by using dense is avoided. Try predict on concatenated features via a minimal dense fallback.
        try:
            # Minimal, last-resort fallback: create dense 1xN and call predict
            import numpy as _np  # type: ignore
            Xc_d = Xc.toarray()
            Xw_d = Xw.toarray()
            X = _np.hstack([Xc_d, Xw_d])
            y_pred = clf.predict(X)
            return bool(int(y_pred[0]) == 1)
        except Exception:
            logger.warning("Estimator lacks linear params and dense fallback failed.")
            return None
    except Exception as e:
        logger.warning("Model inference failed: %s", e)
        return None

# --- FastAPI ---

class CheckRequest(BaseModel):
    text: str
    advanced: bool = False  # kept for backward-compatibility (ignored for now)

class DefaultResponse(BaseModel):
    found: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Always load ML model (ML-only service)
    load_model()
    logger.info("Startup complete. MODEL_AVAILABLE=%s", MODEL_AVAILABLE)
    yield
    # No teardown required.

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok", "message": "bad-words service (ml-only)"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_AVAILABLE,
        "model_path": MODEL_PATH,
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
