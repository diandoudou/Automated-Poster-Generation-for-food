"""
tools/rag_retriever.py
---------------------------------
Lightweight RAG (Retrieval Augmented Generation) retrieval module, used by Layout Planner
and other components to query the "design knowledge base."
To ensure it can run in local/offline environments, this implementation
first attempts to use sentence-transformers; if unavailable, it falls back
to a simple Bag-of-Words + cosine similarity approach.
"""

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]+", text.lower())
    return tokens or [text.lower()]

class _SentenceEncoder:
    """
        Prefer using sentence-transformers to generate semantic vectors;
        if not installed, fall back to Bag-of-Words.
    """
    def __init__(self):
        self._encoder = None
        self._mode = "bow"
        self._vocab = {}

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self._mode = "st"
        except Exception:
            pass

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._mode == "st" and self._encoder is not None:
            embeddings = self._encoder.encode(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )
            return embeddings

        self._build_vocab(texts)
        return self._bow_encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._mode == "st" and self._encoder is not None:
            embeddings = self._encoder.encode(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )
            return embeddings
        return self._bow_encode(texts)

    # Implement Bag-of-Words
    def _build_vocab(self, texts: Sequence[str]) -> None:
        vocab = {}
        for text in texts:
            for token in _tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab

    def _bow_encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._vocab:
            self._build_vocab(texts)
        vocab_size = len(self._vocab)
        vectors = np.zeros((len(texts), vocab_size), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                idx = self._vocab.get(token)
                if idx is not None:
                    vectors[row, idx] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors = vectors / norms
        return vectors

@dataclass
class DesignSnippet:
    text: str
    score: float
    heading: str
    source: str

class DesignKnowledgeBase:
    def __init__(
        self,
        doc_paths: Sequence[str] | None = None,
        chunk_size: int = 600,
        min_chunk_chars: int = 120,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        if doc_paths is None:
            doc_paths = [
                project_root / "rag" / "docs" / "poster-rag-guide.md",
            ]
        self.doc_paths = [Path(p) for p in doc_paths]
        self.chunk_size = chunk_size
        self.min_chunk_chars = min_chunk_chars

        self.entries = self._load_entries()
        self.encoder = _SentenceEncoder()
        self.embeddings = self.encoder.fit_transform([e["text"] for e in self.entries])

    # Public API
    def retrieve(self, query: str, top_k: int = 3) -> List[DesignSnippet]:
        if not query.strip():
            query = "poster layout rules"

        query_emb = self.encoder.encode([query])[0]
        query_emb = self._normalize(query_emb)
        entry_embeddings = self._normalize(self.embeddings)

        scores = entry_embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = self.entries[idx]
            results.append(
                DesignSnippet(
                    text=entry["text"],
                    score=float(scores[idx]),
                    heading=entry["heading"],
                    source=str(entry["source"]),
                )
            )
        return results

    # Internal helpers
    def _load_entries(self) -> List[dict]:
        entries: List[dict] = []
        for path in self.doc_paths:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            entries.extend(self._split_into_chunks(text, path))
        if not entries:
            raise RuntimeError("DesignKnowledgeBase: 未找到任何可用的知识文档。")
        return entries

    def _split_into_chunks(self, text: str, source_path: Path) -> List[dict]:
        entries: List[dict] = []
        current_heading = "General"
        buffer: List[str] = []

        def flush_buffer():
            if not buffer:
                return
            chunk_text = "\n".join(buffer).strip()
            buffer.clear()
            if not chunk_text:
                return
            for sub_chunk in self._chunk_text(chunk_text):
                entries.append(
                    {
                        "text": sub_chunk,
                        "heading": current_heading,
                        "source": source_path,
                    }
                )

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                flush_buffer()
                current_heading = stripped[3:].strip() or "General"
            else:
                buffer.append(line)

        flush_buffer()
        return entries

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < self.chunk_size:
                current += ("\n\n" if current else "") + para
            else:
                if len(current) >= self.min_chunk_chars:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim == 1:
            norm = np.linalg.norm(matrix) + 1e-8
            return matrix / norm
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

__all__ = [
    "DesignKnowledgeBase",
    "DesignSnippet",
]