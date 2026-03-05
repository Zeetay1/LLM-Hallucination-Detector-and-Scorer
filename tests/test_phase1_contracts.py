import json
from pathlib import Path

import pytest

from hallucination_scorer import (
    FixtureLabel,
    GroundingRequest,
    GroundingResponse,
    SentenceGrounding,
    score_claim,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = PROJECT_ROOT / "data" / "fixtures" / "phase1_examples.json"


def load_phase1_examples() -> list[FixtureLabel]:
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return [FixtureLabel(**item) for item in raw]


def test_phase1_fixtures_are_valid():
    """All Phase 1 fixtures should conform to the FixtureLabel schema."""
    examples = load_phase1_examples()
    assert len(examples) == 20
    for example in examples:
        assert example.id
        assert example.claim
        assert example.context_chunks
        lo, hi = example.expected_overall_score_range
        assert 0.0 <= lo <= hi <= 1.0


def test_request_and_response_schemas_construct():
    """Basic construction of request/response models should succeed."""
    example = load_phase1_examples()[0]

    request = GroundingRequest(
        claim=example.claim,
        context_chunks=example.context_chunks,
    )
    assert request.claim == example.claim
    assert request.context_chunks == example.context_chunks

    dummy_sentence = SentenceGrounding(
        sentence="Dummy sentence.",
        sentence_index=0,
        best_chunk_index=None,
        best_chunk_document_id=None,
        best_chunk_score=None,
        unsupported=True,
        evidence_span_text=None,
        evidence_span_start=None,
        evidence_span_end=None,
    )
    response = GroundingResponse(
        overall_grounding_score=0.5,
        per_sentence=[dummy_sentence],
        unsupported_sentence_count=1,
        calibrated_confidence=0.5,
        used_retrieval=False,
        retrieved_chunks=None,
    )
    assert response.overall_grounding_score == 0.5
    assert len(response.per_sentence) == 1


def test_obvious_hallucination_scores_low():
    """
    An obvious hallucination should receive a low overall grounding score.

    This test is expected to fail until the scoring pipeline is implemented
    in later phases.
    """
    examples = load_phase1_examples()
    obvious = next(e for e in examples if e.label_type == "obvious_hallucination")

    request = GroundingRequest(
        claim=obvious.claim,
        context_chunks=obvious.context_chunks,
    )

    response = score_claim(request)

    assert response.overall_grounding_score < 0.3

