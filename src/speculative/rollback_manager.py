"""
Rollback manager -- coordinates cancelling in-flight speculation and TTS
when divergence is detected, and decides commit vs rollback.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.speculative.divergence_detector import compute_divergence, DivergenceResult

log = logging.getLogger("rollback_manager")


@dataclass
class RollbackDecision:
    committed: bool
    divergence: Optional[DivergenceResult]
    speculative_response: Optional[str]


def resolve_speculation(speculator, tts_engine, final_transcript: str,
                         divergence_threshold: float = 0.7) -> RollbackDecision:
    triggering_partial = speculator.get_triggering_partial()
    if triggering_partial is None:
        return RollbackDecision(committed=False, divergence=None, speculative_response=None)

    divergence = compute_divergence(triggering_partial, final_transcript, threshold=divergence_threshold)

    if divergence.diverged:
        log.warning(
            "DIVERGENCE detected (similarity=%.2f < threshold=%.2f) -- rolling back. "
            "Triggering partial: %r | Final: %r",
            divergence.similarity, divergence_threshold, triggering_partial, final_transcript,
        )
        speculator.cancel()
        tts_engine.cancel()
        return RollbackDecision(committed=False, divergence=divergence, speculative_response=None)

    speculative_response = speculator.resolve(final_transcript)
    if speculative_response is None:
        log.warning(
            "Divergence check passed (similarity=%.2f) but no usable speculative result -- falling back.",
            divergence.similarity,
        )
        return RollbackDecision(committed=False, divergence=divergence, speculative_response=None)

    log.info(
        "COMMIT (similarity=%.2f >= threshold=%.2f) -- using speculative response.",
        divergence.similarity, divergence_threshold,
    )
    return RollbackDecision(committed=True, divergence=divergence, speculative_response=speculative_response)
