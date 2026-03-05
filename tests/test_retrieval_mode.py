from hallucination_scorer import GroundingRequest, score_claim


def test_retrieval_mode_sets_flag_and_returns_chunks():
    corpus_docs = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Mount Everest is Earth's highest mountain above sea level.",
    ]
    claim = "The Eiffel Tower is in Paris."

    request = GroundingRequest(claim=claim, corpus_docs=corpus_docs)
    response = score_claim(request)

    assert response.used_retrieval is True
    assert response.retrieved_chunks is not None
    assert len(response.retrieved_chunks) > 0
    # All retrieved chunks should have document_id and chunk_index set.
    for chunk in response.retrieved_chunks:
        assert chunk.document_id.startswith("doc_")
        assert chunk.chunk_index >= 0

