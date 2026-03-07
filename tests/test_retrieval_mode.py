from unittest.mock import patch

import numpy as np

from hallucination_scorer import Chunk, GroundingRequest, score_claim


def test_retrieval_mode_sets_flag_and_returns_chunks():
    """Retrieval mode sets used_retrieval and returns chunks with provenance. Mocks used for no live model load."""
    corpus_docs = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Mount Everest is Earth's highest mountain above sea level.",
    ]
    claim = "The Eiffel Tower is in Paris."
    mock_chunks = [
        Chunk(document_id="doc_0", chunk_index=0, text=corpus_docs[0]),
        Chunk(document_id="doc_1", chunk_index=0, text=corpus_docs[1]),
    ]

    with patch("hallucination_scorer.retrieve_top_k_chunks", return_value=mock_chunks):
        with patch("hallucination_scorer.scoring.get_nli_model") as mock_nli:
            mock_model = mock_nli.return_value
            mock_model.predict_entailment_matrix.return_value = np.array([[0.9, 0.2]])
            request = GroundingRequest(claim=claim, corpus_docs=corpus_docs)
            response = score_claim(request)

    assert response.used_retrieval is True
    assert response.retrieved_chunks is not None
    assert len(response.retrieved_chunks) > 0
    for chunk in response.retrieved_chunks:
        assert chunk.document_id.startswith("doc_")
        assert chunk.chunk_index >= 0

