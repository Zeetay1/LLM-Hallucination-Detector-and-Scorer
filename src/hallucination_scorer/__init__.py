from __future__ import annotations

from typing import List

from .config import CONFIG
from .schemas import (
    Chunk,
    CostMetadata,
    FixtureLabel,
    GroundingRequest,
    GroundingResponse,
    SentenceGrounding,
)
from .retrieval import retrieve_top_k_chunks
from .scoring import score_claim_against_chunks


def _normalize_context_chunks(context_chunks: List[str]) -> List[Chunk]:
    """
    Convert a list of raw context strings into Chunk objects with synthetic IDs.
    """
    return [
        Chunk(document_id="context", chunk_index=i, text=text)
        for i, text in enumerate(context_chunks)
    ]


def score_claim(request: GroundingRequest) -> GroundingResponse:
    """
    Score how well an LLM-generated claim is grounded in provided evidence.

    This is the main public entry point for the hallucination scoring pipeline.
    In Phase 2 we support the direct context mode only; retrieval mode will be
    added in Phase 3.
    """
    if request.context_chunks:
        chunks = _normalize_context_chunks(request.context_chunks)
        response = score_claim_against_chunks(request.claim, chunks)
        return response

    if request.corpus_docs:
        retrieved_chunks = retrieve_top_k_chunks(
            request.claim,
            request.corpus_docs,
            k=request.retrieval_k or CONFIG.scoring.default_retrieval_k,
        )
        base_response = score_claim_against_chunks(request.claim, retrieved_chunks)
        response = base_response.model_copy(
            update={
                "used_retrieval": True,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return response

    raise ValueError("Either context_chunks or corpus_docs must be provided.")


__all__ = [
    "Chunk",
    "CostMetadata",
    "SentenceGrounding",
    "GroundingRequest",
    "GroundingResponse",
    "FixtureLabel",
    "score_claim",
]


