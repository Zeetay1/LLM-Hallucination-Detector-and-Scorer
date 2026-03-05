from __future__ import annotations

from fastapi import FastAPI

from .schemas import GroundingRequest, GroundingResponse
from . import score_claim


app = FastAPI(title="LLM Hallucination Detector & Scorer")


@app.post("/score", response_model=GroundingResponse)
def score_endpoint(request: GroundingRequest) -> GroundingResponse:
    """
    FastAPI endpoint that wraps the core scoring pipeline.
    """
    return score_claim(request)


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hallucination_scorer.api:app", host="0.0.0.0", port=8000, reload=False)

