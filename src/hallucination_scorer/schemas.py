from __future__ import annotations

from typing import List, Optional, Tuple

from typing_extensions import Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    document_id: str = Field(..., description="Identifier for the source document")
    chunk_index: int = Field(..., ge=0, description="Index of the chunk within its source document")
    text: str = Field(..., description="Raw text content of the chunk")


class SentenceGrounding(BaseModel):
    sentence: str
    sentence_index: int = Field(..., ge=0)
    best_chunk_index: Optional[int] = Field(
        None, ge=0, description="Index of the best supporting chunk within the context or retrieved set"
    )
    best_chunk_document_id: Optional[str] = Field(
        None, description="Document ID of the best supporting chunk"
    )
    best_chunk_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Entailment/grounding score for the best supporting chunk, in [0, 1]",
    )
    unsupported: bool = Field(
        ...,
        description="True if no chunk provides sufficient support according to the scoring threshold",
    )
    evidence_span_text: Optional[str] = Field(
        None,
        description="Span of text in the supporting chunk that best evidences the sentence",
    )
    evidence_span_start: Optional[int] = Field(
        None,
        ge=0,
        description="Start character offset of the evidence span in the supporting chunk",
    )
    evidence_span_end: Optional[int] = Field(
        None,
        ge=0,
        description="End character offset (exclusive) of the evidence span in the supporting chunk",
    )


class CostMetadata(BaseModel):
    """Token and cost tracking for external model API calls."""

    tokens_in: int = Field(0, ge=0, description="Input tokens for external API calls; 0 if none.")
    tokens_out: int = Field(0, ge=0, description="Output tokens; 0 if none.")
    estimated_cost: float = Field(
        0.0,
        ge=0.0,
        description="Estimated cost from config pricing constants; 0.0 if no external API.",
    )


class GroundingRequest(BaseModel):
    claim: str = Field(..., description="LLM-generated claim to be evaluated")
    context_chunks: Optional[List[str]] = Field(
        None,
        description="Direct mode: list of context chunks provided by the caller. "
        "These will be normalized to Chunk objects internally.",
    )
    corpus_docs: Optional[List[str]] = Field(
        None,
        description="Retrieval mode: list of raw documents to build a retrieval index over (Phase 3+).",
    )
    retrieval_k: Optional[int] = Field(
        None,
        ge=1,
        description="Optional override for top-k retrieved chunks when using retrieval mode.",
    )


class GroundingResponse(BaseModel):
    overall_grounding_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate grounding score for the entire claim, in [0, 1].",
    )
    per_sentence: List[SentenceGrounding]
    unsupported_sentence_count: int = Field(
        ...,
        ge=0,
        description="Number of sentences in the claim that were deemed unsupported.",
    )
    calibrated_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score for the overall grounding decision, in [0, 1].",
    )
    used_retrieval: bool = Field(
        ...,
        description="Indicates whether retrieval over a corpus was used to construct the context.",
    )
    retrieved_chunks: Optional[List[Chunk]] = Field(
        None,
        description="Chunks retrieved from a corpus when retrieval mode is used (Phase 3+).",
    )
    cost_metadata: CostMetadata = Field(
        default_factory=lambda: CostMetadata(),
        description="Token and cost metadata; attached to every response and persisted with results.",
    )


class FixtureLabel(BaseModel):
    id: str
    claim: str
    context_chunks: List[str]
    label_type: Literal["obvious_hallucination", "partial", "fully_grounded"]
    expected_overall_score_range: Tuple[float, float] = Field(
        ...,
        description="Inclusive bounds [min, max] for the expected overall grounding score.",
    )

