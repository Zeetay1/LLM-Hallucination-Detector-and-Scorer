from __future__ import annotations

from typing import List

import spacy

_nlp = None


def _get_nlp():
    """
    Create a lightweight, deterministic spaCy pipeline for sentence splitting.

    Uses a blank English model with the built-in sentencizer component only.
    This avoids heavyweight language model downloads while providing robust
    sentence boundary detection.
    """
    global _nlp
    if _nlp is None:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _nlp = nlp
    return _nlp


def split_into_sentences(text: str) -> List[str]:
    """
    Split a claim into sentences using spaCy's sentencizer.

    Returns a list of non-empty, stripped sentence strings.
    """
    if not text:
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

