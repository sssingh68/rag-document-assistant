"""
app/rag_pipeline.py

Simplified, robust RAG pipeline used by the demo:
 - local sentence-transformers embeddings (all-MiniLM-L6-v2)
 - chromadb PersistentClient vector store (with recovery path)
 - calls Ollama via HTTP or client if available
 - ingestion helpers: ingest_file + ingest_and_update_status
 - answer_query wrapper used by the API
"""

import os
import time
import logging
import json
import threading
from pathlib import Path
from typing import List, Optional
import numpy as np
import requests

# Optional imports
try:
    import chromadb
    CHROMADB_OK = True
except Exception:
    chromadb = None
    CHROMADB_OK = False

try:
    from sentence_transformers import SentenceTransformer
    HF_OK = True
except Exception:
    SentenceTransformer = None
    HF_OK = False

try:
    import ollama as ollama_client
    OLLAMA_CLIENT_OK = True
except Exception:
    ollama_client = None
    OLLAMA_CLIENT_OK = False

# constants & paths
BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "sample_data" / "uploads"
VECTOR_DIR = BASE / "vector_store"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

logger = logging.getLogger("rag_pipeline")
logging.basicConfig(level=logging.INFO)

# global state
_client = None
_collection = None
_COLLECTION_NAME = "rag_docs"
_hf_model = None
_last_sources: List[str] = []
_active_source: Optional[str] = None
_chromadb_lock = threading.Lock()


# -----------------------
# Helpers
# -----------------------
def is_ollama_up(host: str = None) -> bool:
    host = host or OLLAMA_HOST
    for path in ["/api/tags", "/api/models", "/v1/models"]:
        try:
            r = requests.get(f"{host}{path}", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def call_ollama_http(prompt: str, model_name: str = DEFAULT_OLLAMA_MODEL, host: str = None, timeout: int = 60) -> str:
    """Call Ollama via HTTP; compatible with various response shapes."""
    host = host or OLLAMA_HOST
    url = f"{host}/api/generate"
    try:
        r = requests.post(url, json={"model": model_name, "prompt": prompt}, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        try:
            j = r.json()
            # defensive extraction
            if isinstance(j, dict):
                return j.get("response") or j.get("output") or j.get("text") or str(j)
            return str(j)
        except Exception:
            return r.text.strip()
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


# -----------------------
# HF model (embeddings)
# -----------------------
def get_hf_model():
    global _hf_model
    if not HF_OK:
        return None
    if _hf_model is None:
        logger.info("Loading HF embedding model: all-MiniLM-L6-v2")
        _hf_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _hf_model


def _embed_chunks(chunks: List[str]):
    model = get_hf_model()
    if model is None:
        raise RuntimeError("HF model not available")
    embs = model.encode(chunks, show_progress_bar=False)
    return [np.array(v).astype("float32").tolist() for v in embs]


def _embed_query(query: str):
    model = get_hf_model()
    if model is None:
        raise RuntimeError("HF model not available")
    v = model.encode([query], show_progress_bar=False)[0]
    return np.array(v).astype("float32").tolist()


# -----------------------
# Chroma init / resilience
# -----------------------
def init_chromadb_client() -> bool:
    """
    Safe init for ChromaDB. If Chroma raises a known config KeyError (legacy DB),
    reset the vector store folder and recreate.
    """
    global _client, _collection
    if not CHROMADB_OK:
        logger.warning("Chromadb not installed.")
        return False

    with _chromadb_lock:
        if _client is None:
            try:
                _client = chromadb.PersistentClient(path=str(VECTOR_DIR))
                logger.info("Chromadb PersistentClient initialized.")
            except Exception as e:
                logger.exception("Chroma init failed: %s", e)
                return False

        # attempt to get or create collection; handle KeyError:'_type' (schema mismatch)
        try:
            _collection = _client.get_collection(name=_COLLECTION_NAME)
            logger.info("Chromadb collection loaded: %s", _COLLECTION_NAME)
        except Exception as e_get:
            logger.warning("Chromadb get_collection failed: %s", e_get)
            # try create
            try:
                _collection = _client.create_collection(name=_COLLECTION_NAME)
                logger.info("Chroma collection created.")
            except Exception as e_create:
                logger.exception("Chromadb create_collection failed: %s", e_create)
                # attempt destructive reset (only if folder exists) to recover legacy broken state
                try:
                    if VECTOR_DIR.exists():
                        logger.warning("Resetting vector store at %s (deleting folder) to recover from chromadb config errors.", VECTOR_DIR)
                        # careful deletion
                        for child in VECTOR_DIR.glob("*"):
                            if child.is_file():
                                child.unlink()
                            else:
                                import shutil
                                shutil.rmtree(child)
                        # re-init client
                        _client = chromadb.PersistentClient(path=str(VECTOR_DIR))
                        _collection = _client.create_collection(name=_COLLECTION_NAME)
                        logger.info("Chromadb re-initialized after reset.")
                    else:
                        return False
                except Exception as e_reset:
                    logger.exception("Failed to reset vector store: %s", e_reset)
                    return False

    return True


def persist_chromadb():
    global _client
    try:
        if _client:
            _client.persist()
    except Exception:
        pass


# -----------------------
# Ingestion
# -----------------------
def ingest_file(file_path: str) -> dict:
    """
    Extract text (PDF/TXT), split into chunks, embed, add to chromadb.
    """
    from PyPDF2 import PdfReader
    logger.info("Starting ingestion for %s", file_path)
    text = ""
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            logger.exception("PDF extraction failed")
    else:
        try:
            text = open(file_path, "r", encoding="utf-8").read()
        except Exception:
            logger.exception("Text read failed")

    if not text.strip():
        return {"ok": False, "msg": "No text extracted"}

    # simple chunking by characters
    chunks, chunk_size, overlap = [], 800, 50
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap

    if not (HF_OK and CHROMADB_OK):
        return {"ok": False, "msg": "Dependencies missing"}

    if not init_chromadb_client():
        return {"ok": False, "msg": "Chroma init failed"}

    try:
        vectors = _embed_chunks(chunks)
        name = Path(file_path).name
        ids, docs, metas = [], [], []
        for idx, ch in enumerate(chunks):
            ids.append(f"{name}__{idx}")
            docs.append(ch)
            metas.append({"source": name, "chunk": idx})
        _collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        persist_chromadb()
        logger.info("Ingested %d chunks into chromadb for %s", len(chunks), name)
        return {"ok": True, "msg": f"{name} ingested", "chunks": len(chunks)}
    except Exception as e:
        logger.exception("Embedding/Chroma add failed: %s", e)
        return {"ok": False, "msg": str(e)}


def ingest_and_update_status(file_path: str, status_path: str):
    """
    Helper used by main to run ingestion in background and update status file.
    Writes "pending", then "ok" or "error:...".
    """
    try:
        open(status_path, "w").write("pending")
        res = ingest_file(file_path)
        open(status_path, "w").write("ok" if res.get("ok") else f"error:{res.get('msg')}")
    except Exception as e:
        open(status_path, "w").write(f"error:{e}")
        logger.exception("ingest_and_update_status failed: %s", e)


# -----------------------
# Answer/query
# -----------------------
def _mock_answer(query: str) -> str:
    return "Mock answer: no vector DB available."

def answer_query(query: str, k: int = 3, context_chars: int = 1000) -> str:
    """
    Main query function. Returns a string answer.
    Top-level exception handling returns safe message for UI.
    """
    global _last_sources
    try:
        # trivial checks
        if not query or not query.strip():
            return "Empty query."

        # ensure chromadb/hf available
        if not CHROMADB_OK:
            _last_sources = ["(mock) chromadb not installed"]
            return _mock_answer(query)
        if not init_chromadb_client():
            _last_sources = ["(mock) chromadb init failed"]
            return _mock_answer(query)
        if not HF_OK:
            _last_sources = ["(mock) sentence-transformers not installed"]
            return _mock_answer(query)

        # embed query
        try:
            q_vec = _embed_query(query)
        except Exception:
            logger.exception("Query embedding failed")
            _last_sources = ["(mock) Query embedding failed"]
            return _mock_answer(query)

        # query chroma
        try:
            # try modern API, then older fallback
            try:
                res = _collection.query(query_embeddings=[q_vec], n_results=k, include=['documents', 'metadatas', 'distances'])
            except Exception:
                res = _collection.query([q_vec], n_results=k, include=['documents', 'metadatas', 'distances'])
        except Exception:
            logger.exception("Chromadb query failed")
            _last_sources = ["(mock) chromadb query failed"]
            return _mock_answer(query)

        # parse results
        try:
            if isinstance(res, dict):
                docs = res.get('documents', [[]])[0]
                metadatas = res.get('metadatas', [[]])[0]
            else:
                docs = res.documents[0] if hasattr(res, "documents") else []
                metadatas = res.metadatas[0] if hasattr(res, "metadatas") else []
        except Exception:
            logger.exception("Parsing chromadb response failed")
            _last_sources = ["(mock) chromadb parse failed"]
            return _mock_answer(query)

        if not docs:
            _last_sources = ["No docs found in vectorstore."]
            return "No indexed documents found."

        # build prompt
        header = (
            "You are a helpful assistant. Use only the provided context to answer.\n\n"
        )
        prompt = header
        srcs = []
        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            excerpt = doc[:context_chars]
            srcs.append(f"{meta.get('source', '?')} — chunk {meta.get('chunk','?')}")
            prompt += f"Context {i+1} ({meta.get('source','?')} chunk {meta.get('chunk','?')}):\n{excerpt}\n\n"

        prompt += f"Question: {query}\n\nAnswer:"
        _last_sources = srcs

        # prefer ollama: try official client, else HTTP
        if is_ollama_up():
            # official client if available
            if OLLAMA_CLIENT_OK and ollama_client is not None:
                try:
                    model_name = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
                    resp = ollama_client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
                    # defensive parsing
                    if isinstance(resp, dict):
                        msg = resp.get("message", {})
                        if isinstance(msg, dict):
                            content = msg.get("content") or msg.get("text") or ""
                            if content:
                                return str(content)
                        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                            first = resp["choices"][0]
                            if isinstance(first, dict):
                                if "message" in first and isinstance(first["message"], dict):
                                    return str(first["message"].get("content", "") or "")
                                if "text" in first:
                                    return str(first.get("text", ""))
                            return str(first)
                    return str(resp)
                except Exception:
                    logger.exception("Ollama client.chat failed - falling back to HTTP")

            # HTTP fallback
            try:
                model_name = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
                res_text = call_ollama_http(prompt, model_name=model_name, host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST), timeout=60)
                return str(res_text)
            except Exception as e:
                logger.exception("Ollama HTTP call failed: %s", e)
                _last_sources = [f"(mock) Ollama HTTP failed: {e}"]
                return _mock_answer(query)
        else:
            _last_sources = ["(mock) Ollama not running"]
            return _mock_answer(query)

    except Exception as e:
        logger.exception("Unhandled exception in answer_query: %s", e)
        _last_sources = [f"(error) {type(e).__name__}: {str(e)[:200]}"]
        return "Internal error while answering. Check server logs for details."


# -----------------------
# Status helpers
# -----------------------
def get_last_sources() -> List[str]:
    return _last_sources.copy()


def status() -> dict:
    return {
        "ollama": is_ollama_up(),
        "chromadb_installed": CHROMADB_OK,
        "hf_installed": HF_OK,
        "vector_store_exists": VECTOR_DIR.exists()
    }


def get_active_source() -> Optional[str]:
    return _active_source


def set_active_source(name: Optional[str]):
    global _active_source
    _active_source = name
