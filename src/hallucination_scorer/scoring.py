from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import CONFIG
from .nli_model import get_nli_model
from .schemas import Chunk, GroundingResponse, SentenceGrounding
from .sentence_splitter import split_into_sentences


def _compute_evidence_span(sentence: str, chunk_text: str) -> Tuple[str | None, int | None, int | None]:
    """
    Heuristic evidence span extraction.

    Searches for the longest n-gram from the sentence that appears verbatim
    in the chunk text (case-insensitive). Returns the matched substring and
    its character offsets, or (None, None, None) if nothing reasonable is found.
    """
    sentence_clean = sentence.strip()
    chunk_lower = chunk_text.lower()
    tokens = [t for t in sentence_clean.split() if t]
    if not tokens:
        return None, None, None

    # Try n-grams from longest to shortest
    for n in range(len(tokens), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngram_lower = ngram.lower()
            idx = chunk_lower.find(ngram_lower)
            if idx != -1:
                start = idx
                end = idx + len(ngram)
                return chunk_text[start:end], start, end

    return None, None, None


def score_claim_against_chunks(claim: str, chunks: List[Chunk]) -> GroundingResponse:
    """
    Core scoring routine operating on a claim string and a list of Chunk objects.
    """
    sentences = split_into_sentences(claim)
    if not sentences:
        # Degenerate case: empty claim, treat as fully unsupported.
        return GroundingResponse(
            overall_grounding_score=0.0,
            per_sentence=[],
            unsupported_sentence_count=0,
            calibrated_confidence=0.0,
            used_retrieval=False,
            retrieved_chunks=None,
        )

    chunk_texts = [c.text for c in chunks]
    nli = get_nli_model()
    scores_matrix = nli.predict_entailment_matrix(sentences, chunk_texts)

    sentence_groundings: List[SentenceGrounding] = []
    best_scores: List[float] = []
    unsupported_count = 0

    for idx, sentence in enumerate(sentences):
        row = scores_matrix[idx]
        if row.size == 0:
            best_idx = None
            best_score = 0.0
        else:
            best_idx_int = int(np.argmax(row))
            best_score = float(row[best_idx_int])
            best_idx = best_idx_int

        unsupported = best_score < CONFIG.scoring.entailment_threshold
        if unsupported:
            unsupported_count += 1

        if best_idx is not None:
            best_chunk = chunks[best_idx]
            evidence_span_text, evidence_start, evidence_end = _compute_evidence_span(
                sentence, best_chunk.text
            )
            best_chunk_document_id = best_chunk.document_id
        else:
            best_chunk = None
            best_chunk_document_id = None
            evidence_span_text, evidence_start, evidence_end = None, None, None

        sentence_groundings.append(
            SentenceGrounding(
                sentence=sentence,
                sentence_index=idx,
                best_chunk_index=best_idx,
                best_chunk_document_id=best_chunk_document_id,
                best_chunk_score=best_score if best_idx is not None else None,
                unsupported=unsupported,
                evidence_span_text=evidence_span_text,
                evidence_span_start=evidence_start,
                evidence_span_end=evidence_end,
            )
        )
        best_scores.append(best_score)

    # Aggregate document-level score with strong penalty for unsupported sentences.
    scores_array = np.array(best_scores, dtype=float)
    base_score = float(scores_array.mean()) if scores_array.size > 0 else 0.0
    unsupported_fraction = unsupported_count / len(sentences)
    overall_score = base_score * (1.0 - unsupported_fraction)

    # For now, calibrated_confidence mirrors the overall score; Phase 4 will adjust this.
    response = GroundingResponse(
        overall_grounding_score=overall_score,
        per_sentence=sentence_groundings,
        unsupported_sentence_count=unsupported_count,
        calibrated_confidence=overall_score,
        used_retrieval=False,
        retrieved_chunks=None,
    )
    return response

