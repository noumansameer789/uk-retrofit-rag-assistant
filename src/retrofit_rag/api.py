"""FastAPI entry point for the grounded retrofit assistant."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .providers import ConfigurationError, LLMError, build_llm
from .rag import RAGService
from .retrieval import Retriever, load_catalogue


DEFAULT_CATALOGUE = Path(__file__).parents[2] / "data" / "guidance.json"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1_000)
    top_k: int = Field(default=3, ge=1, le=5)


class CitationResponse(BaseModel):
    id: int
    title: str
    url: str
    score: float


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[CitationResponse]
    refusal_reason: str | None = None


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    provider = os.getenv("LLM_PROVIDER", "").strip()
    if not provider:
        raise ConfigurationError("LLM_PROVIDER is required")
    catalogue = Path(os.getenv("GUIDANCE_CATALOGUE", str(DEFAULT_CATALOGUE)))
    return RAGService(Retriever(load_catalogue(catalogue)), build_llm(provider))


app = FastAPI(
    title="UK Retrofit RAG Assistant",
    version="1.0.0",
    description="Citation-first LLM answers over curated official UK guidance.",
)


@app.exception_handler(ConfigurationError)
async def configuration_error(_, exc: ConfigurationError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def provider_error(_, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, object]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    return {
        "status": "ok",
        "provider_configured": provider in {"openai", "openai-compatible", "ollama"},
        "provider": provider or None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: RAGService = Depends(get_service)) -> dict[str, object]:
    return service.ask(payload.question, top_k=payload.top_k).to_dict()
