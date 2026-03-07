"""Cost metadata must be present and non-zero after a mocked external API call."""

import json
from unittest.mock import patch

from hallucination_scorer import GroundingResponse, score_claim
from hallucination_scorer.schemas import CostMetadata, SentenceGrounding
from hallucination_scorer.utils.cost_tracking import (
    compute_cost_metadata,
    persist_response,
    zero_cost_metadata,
)


def test_zero_cost_metadata_is_present():
    """Zero cost metadata (local-only) is valid and has expected shape."""
    meta = zero_cost_metadata()
    assert meta.tokens_in == 0
    assert meta.tokens_out == 0
    assert meta.estimated_cost == 0.0


def test_compute_cost_metadata_non_zero_after_mocked_external_api_call():
    """
    After a mocked external API call, cost metadata must be present and non-zero.

    Simulates an external model API returning token counts; we compute
    CostMetadata and attach to response, then assert metadata is present and non-zero.
    """
    tokens_in = 100
    tokens_out = 50
    meta = compute_cost_metadata(tokens_in, tokens_out)
    assert meta.tokens_in == tokens_in
    assert meta.tokens_out == tokens_out
    assert meta.estimated_cost > 0.0

    # Attach to a minimal response and assert
    dummy = SentenceGrounding(
        sentence="Test.",
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
        overall_grounding_score=0.0,
        per_sentence=[dummy],
        unsupported_sentence_count=1,
        calibrated_confidence=0.0,
        used_retrieval=False,
        retrieved_chunks=None,
        cost_metadata=meta,
    )
    assert response.cost_metadata is not None
    assert response.cost_metadata.tokens_in > 0
    assert response.cost_metadata.tokens_out > 0
    assert response.cost_metadata.estimated_cost > 0.0


def test_response_persisted_includes_cost_metadata(tmp_path):
    """Persisting a response includes cost_metadata in the saved JSON."""
    meta = compute_cost_metadata(10, 5)
    dummy = SentenceGrounding(
        sentence="Test.",
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
        per_sentence=[dummy],
        unsupported_sentence_count=0,
        calibrated_confidence=0.5,
        used_retrieval=False,
        retrieved_chunks=None,
        cost_metadata=meta,
    )
    path = tmp_path / "response.json"
    persist_response(path, response)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "cost_metadata" in data
    assert data["cost_metadata"]["tokens_in"] == 10
    assert data["cost_metadata"]["tokens_out"] == 5
    assert data["cost_metadata"]["estimated_cost"] > 0


def test_score_claim_response_includes_cost_metadata():
    """Every score_claim response includes cost_metadata (zero when no external API)."""
    import numpy as np

    from hallucination_scorer import GroundingRequest

    with patch("hallucination_scorer.scoring.get_nli_model") as mock_nli:
        mock_model = mock_nli.return_value
        mock_model.predict_entailment_matrix.return_value = np.array([[0.9]])
        response = score_claim(
            GroundingRequest(claim="The sky is blue.", context_chunks=["The sky is blue."])
        )
    assert hasattr(response, "cost_metadata")
    assert response.cost_metadata is not None
    assert isinstance(response.cost_metadata, CostMetadata)
