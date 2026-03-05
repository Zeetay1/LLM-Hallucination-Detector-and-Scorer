from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CONFIG


class NLIModel:
    """
    Thin wrapper around a HuggingFace NLI model that exposes entailment
    probabilities in a deterministic, batched API.
    """

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(CONFIG.nli.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG.nli.model_name
        )
        self._model.eval()

    @torch.no_grad()
    def predict_entailment_scores(
        self,
        premises: Iterable[str],
        hypotheses: Iterable[str],
    ) -> np.ndarray:
        """
        Compute entailment probabilities for (premise, hypothesis) pairs.

        Returns:
            A NumPy array of shape (N,) with probabilities in [0, 1].
        """
        pair_list: List[tuple[str, str]] = list(zip(premises, hypotheses))
        if not pair_list:
            return np.zeros((0,), dtype=float)

        inputs = self._tokenizer(
            [p for p, _ in pair_list],
            [h for _, h in pair_list],
            padding=True,
            truncation=True,
            max_length=CONFIG.nli.max_length,
            return_tensors="pt",
        )

        outputs = self._model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entailment_probs = probs[:, CONFIG.nli.entailment_label_id]
        return entailment_probs.cpu().numpy()

    def predict_entailment_matrix(
        self,
        sentences: List[str],
        chunk_texts: List[str],
    ) -> np.ndarray:
        """
        Compute an entailment matrix of shape (num_sentences, num_chunks),
        where entry (i, j) is the probability that sentence i is entailed by
        chunk j.
        """
        if not sentences or not chunk_texts:
            return np.zeros((len(sentences), len(chunk_texts)), dtype=float)

        premises: List[str] = []
        hypotheses: List[str] = []
        for sentence in sentences:
            for chunk in chunk_texts:
                premises.append(chunk)
                hypotheses.append(sentence)

        flat_scores = self.predict_entailment_scores(premises, hypotheses)
        matrix = flat_scores.reshape(len(sentences), len(chunk_texts))
        return matrix


_nli_model: NLIModel | None = None


def get_nli_model() -> NLIModel:
    global _nli_model
    if _nli_model is None:
        _nli_model = NLIModel()
    return _nli_model

