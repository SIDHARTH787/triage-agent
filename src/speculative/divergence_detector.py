"""
Divergence detector -- lightweight, dependency-free text similarity
(difflib + word overlap), avoiding embedding models to not compete with
ASR/TTS for CPU on an already resource-constrained machine.
"""

import re
import difflib
from dataclasses import dataclass


@dataclass
class DivergenceResult:
    similarity: float
    diverged: bool
    partial_text: str
    final_text: str


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sequence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_overlap_similarity(a: str, b: str) -> float:
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def compute_divergence(partial_text: str, final_text: str, threshold: float = 0.7) -> DivergenceResult:
    norm_partial = _normalize(partial_text)
    norm_final = _normalize(final_text)

    seq_sim = _sequence_similarity(norm_partial, norm_final)
    word_sim = _word_overlap_similarity(norm_partial, norm_final)
    combined_similarity = (seq_sim + word_sim) / 2

    return DivergenceResult(
        similarity=combined_similarity,
        diverged=combined_similarity < threshold,
        partial_text=partial_text,
        final_text=final_text,
    )
