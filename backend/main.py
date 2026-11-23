"""
CalorAI - FastAPI application.

    uvicorn backend.main:app --reload --port 8000

API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from backend.api import chat, data, rag  # noqa: E402  (load .env before settings are read)
from backend.core.config import VERSION, get_settings  # noqa: E402
from backend.core.llm import llm_mode  # noqa: E402
from backend.core.telemetry import recorder  # noqa: E402
from backend.db.database import init_db  # noqa: E402
from backend.rag.service import get_rag, init_rag  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("calorai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    init_db()
    rag_engine = init_rag(
        s.google_api_key,
        enable_dense=s.enable_dense and s.llm_mode != "offline",
        embedding_model=s.embedding_model,
        embedding_dims=s.embedding_dims,
        cache_dir=s.cache_dir / "dense",
    )
    log.info(
        "CalorAI %s | llm=%s | knowledge base: %d foods | dense=%s",
        VERSION,
        llm_mode(),
        len(rag_engine.corpus),
        rag_engine.dense.status if rag_engine.dense else "disabled",
    )
    yield


app = FastAPI(
    title="CalorAI",
    version=VERSION,
    description="Agentic RAG nutrition copilot: LangGraph agent + hybrid retrieval over a nutrition knowledge base.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def response_time_header(request: Request, call_next):
    import time

    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


app.include_router(chat.router)
app.include_router(data.router)
app.include_router(rag.router)


@app.get("/health", tags=["ops"])
def health():
    engine = get_rag()
    return {
        "status": "ok",
        "version": VERSION,
        "llm": {"mode": llm_mode(), "chat_model": get_settings().chat_model if llm_mode() == "gemini" else None},
        "rag": {"foods": len(engine.corpus), "channels": engine.stats()["channels"]},
    }


@app.get("/metrics", tags=["ops"])
def metrics():
    """Rolling p50/p95/p99 latency per chat path (text / image) and per graph node."""
    return recorder.snapshot()


# Single-container deployments: serve the built React app from the same origin.
_static = get_settings().static_dir
if _static.is_dir() and (_static / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "health", "metrics", "docs", "openapi")):
            from fastapi import HTTPException

            raise HTTPException(404, "Not found")
        candidate = _static / path
        if path and candidate.is_file() and candidate.resolve().is_relative_to(_static.resolve()):
            return FileResponse(candidate)
        return FileResponse(_static / "index.html")
