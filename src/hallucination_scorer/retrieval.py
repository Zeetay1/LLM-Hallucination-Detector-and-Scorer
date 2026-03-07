from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import CONFIG
from .schemas import Chunk


def chunk_corpus_docs(corpus_docs: List[str], max_chars: int = 500, overlap: int = 100) -> List[Chunk]:
    """
    Deterministically chunk raw documents into Chunk objects.

    Chunks are produced by sliding a fixed-size window with a configurable
    character overlap over each document.
    """
    chunks: List[Chunk] = []
    for doc_id, doc_text in enumerate(corpus_docs):
        if not doc_text:
            continue
        start = 0
        chunk_index = 0
        while start < len(doc_text):
            end = min(start + max_chars, len(doc_text))
            text = doc_text[start:end]
            chunks.append(
                Chunk(
                    document_id=f"doc_{doc_id}",
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            if end == len(doc_text):
                break
            start = end - overlap
            chunk_index += 1
    return chunks


@dataclass
class EmbeddingIndex:
    """
    Simple in-memory embedding index backed by sentence-transformers and
    brute-force cosine similarity. Sufficient for small corpora and unit tests.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        self._embeddings: np.ndarray | None = None
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if self._embeddings is None:
            self._embeddings = embeddings
            self._chunks = list(chunks)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
            self._chunks.extend(chunks)

    def search(self, query: str, k: int | None = None) -> List[Chunk]:
        if self._embeddings is None or not self._chunks:
            return []
        if k is None:
            k = CONFIG.scoring.default_retrieval_k
        query_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, query_emb)
        k = min(k, len(self._chunks))
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self._chunks[int(i)] for i in top_indices]


def retrieve_top_k_chunks(claim: str, corpus_docs: List[str], k: int | None = None) -> List[Chunk]:
    """
    Build an embedding index over the corpus and return the top-k most similar chunks.
    """
    chunks = chunk_corpus_docs(corpus_docs)
    if not chunks:
        return []
    index = EmbeddingIndex()
    index.add(chunks)
    return index.search(claim, k=k)

