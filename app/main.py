"""
app/main.py

FastAPI front-end for the RAG Document Assistant.
Provides:
 - static UI mount at /ui (expects app/ui folder)
 - /files -> list uploaded files
 - /upload -> upload a file (saved to sample_data/uploads)
 - /ask -> ask a question (body: {"question": "...", "target": null})
 - /status -> simple health/status
 - /last_sources -> returns last RAG sources
"""

import os
import shutil
import threading
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the rag pipeline module (must be in app/rag_pipeline.py)
try:
    from app import rag_pipeline
except Exception as e:
    # Import fallback when running from repository root (if `app` isn't a package)
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from app import rag_pipeline  # re-try
    # if still fails, let it raise normally

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "sample_data" / "uploads"
STATUS_DIR = BASE / "sample_data" / "ingest_status"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Document Assistant")

# Mount UI if present
UI_DIR = BASE / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    logger.info("Mounted UI at /ui from %s", str(UI_DIR))
else:
    logger.info("No UI directory found at %s", str(UI_DIR))


# ---------------------------
# Pydantic models for requests
# ---------------------------
class AskPayloadModel(BaseModel):
    question: str
    target: Optional[str] = None  # optional name of specific file to target in RAG


# ---------------------------
# Utility helpers
# ---------------------------
def list_uploaded_files():
    files = []
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.name):
            if f.is_file():
                files.append(f.name)
    return files


def save_upload(filename: str, src: UploadFile):
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as out_f:
        shutil.copyfileobj(src.file, out_f)
    return dest


def start_background_ingest(dest_path: Path):
    """
    Write a status file and call rag_pipeline.ingest_and_update_status in a thread.
    status file name: STATUS_DIR/<filename>.status
    """
    status_file = STATUS_DIR / (dest_path.name + ".status")

    def worker():
        try:
            # mark pending
            open(status_file, "w").write("pending")
            # call rag pipeline ingestion helper (expected to exist)
            # rag_pipeline.ingest_and_update_status(file_path, status_file_path)
            if hasattr(rag_pipeline, "ingest_and_update_status"):
                rag_pipeline.ingest_and_update_status(str(dest_path), str(status_file))
            else:
                # Fallback: call ingest_file and write status
                res = rag_pipeline.ingest_file(str(dest_path))
                open(status_file, "w").write("ok" if res.get("ok") else f"error:{res.get('msg')}")
        except Exception as e:
            logger.exception("Background ingest failed: %s", e)
            open(status_file, "w").write(f"error:{e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return status_file


# ---------------------------
# Routes
# ---------------------------

@app.get("/", response_class=JSONResponse)
def root():
    # redirect to UI path if exists
    if UI_DIR.exists():
        return RedirectResponse(url="/ui/")
    return {"ok": True, "msg": "RAG backend running"}


@app.get("/files", response_class=JSONResponse)
def files():
    files = list_uploaded_files()
    return {"ok": True, "files": files}


@app.post("/upload", response_class=JSONResponse)
async def upload(file: UploadFile = File(...)):
    """
    Accepts a single file upload and starts background ingestion.
    Returns immediate success and starts ingestion in background thread.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    # create a safe filename: keep original name but prefix a short hash if collisions desired
    safe_name = file.filename.replace(" ", "_")
    dest = UPLOAD_DIR / safe_name
    # if file already exists, append numeric suffix
    i = 1
    base = dest.stem
    ext = dest.suffix
    while dest.exists():
        dest = UPLOAD_DIR / f"{base}_{i}{ext}"
        i += 1
    # save file
    try:
        saved = save_upload(dest.name, file)
    except Exception as e:
        logger.exception("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    # start background ingestion
    status_path = start_background_ingest(saved)
    return {"ok": True, "msg": f"File '{saved.name}' uploaded successfully.", "status_file": str(status_path.name)}


@app.post("/ask", response_class=JSONResponse, response_model=None)
def ask(payload: AskPayloadModel):
    """
    Ask endpoint. Accepts JSON body matching AskPayloadModel.
    response_model=None prevents FastAPI from deriving a Pydantic response model (avoids errors).
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        # call into rag_pipeline.answer_query
        # answer_query(query: str, k: int = 3, context_chars: int = 1000) -> str
        answer = rag_pipeline.answer_query(question)
        sources = rag_pipeline.get_last_sources() if hasattr(rag_pipeline, "get_last_sources") else []
        return {"ok": True, "answer": answer, "sources": sources}
    except Exception as e:
        logger.exception("Ask failed: %s", e)
        # keep a stable JSON shape for frontend
        return {"ok": False, "answer": "Internal error while answering. Check server logs for details.", "error": str(e)[:200]}


@app.get("/status", response_class=JSONResponse)
def status():
    """
    Return service status (calls rag_pipeline.status() if provided).
    """
    st = {"backend": True}
    try:
        if hasattr(rag_pipeline, "status"):
            st.update(rag_pipeline.status())
    except Exception as e:
        logger.exception("Status check failed: %s", e)
        st["error"] = str(e)
    return st


@app.get("/last_sources", response_class=JSONResponse)
def last_sources():
    try:
        return {"ok": True, "sources": rag_pipeline.get_last_sources() if hasattr(rag_pipeline, "get_last_sources") else []}
    except Exception as e:
        logger.exception("last_sources failed: %s", e)
        return {"ok": False, "sources": [], "error": str(e)[:200]}


# ---------------------------
# Startup/shutdown events
# ---------------------------

@app.on_event("startup")
def on_startup():
    logger.info("Application startup complete.")
    # ensure rag pipeline client initialized (if function exists)
    try:
        if hasattr(rag_pipeline, "init_chromadb_client"):
            rag_pipeline.init_chromadb_client()
    except Exception:
        logger.exception("Chromadb init on startup failed (continuing).")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Application shutdown complete.")
