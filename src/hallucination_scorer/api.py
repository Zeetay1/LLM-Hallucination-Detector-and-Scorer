from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .schemas import GroundingRequest, GroundingResponse
from . import score_claim


app = FastAPI(title="LLM Hallucination Detector & Scorer")


@app.post("/score", response_model=GroundingResponse)
def score_endpoint(request: GroundingRequest) -> GroundingResponse:
    """
    FastAPI endpoint that wraps the core scoring pipeline.
    """
    return score_claim(request)


# Serve minimal HTML/JS frontend (no Streamlit)
_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hallucination_scorer.api:app", host="0.0.0.0", port=8000, reload=False)

