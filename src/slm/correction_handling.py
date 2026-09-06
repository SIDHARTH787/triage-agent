"""
Correction detection.

Detects when a patient appears to be correcting a prior (possibly
mistranscribed) statement, and annotates the transcript sent to the SLM
so it knows to treat the correction as authoritative rather than building
on its earlier (possibly wrong) interpretation.
"""

import re
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    is_correction_attempt: bool
    raw_transcript: str


CORRECTION_PATTERNS = [
    r"\bi meant\b",
    r"\bnot\b.{0,15}\bi (meant|said)\b",
    r"\bi (meant|said)\b.{0,15}\bnot\b",
    r"\bthat'?s not what i (meant|said)\b",
    r"\bsorry,? i meant\b",
    r"\bno,? i (meant|said)\b",
    r"\blet me correct\b",
    r"\bi didn'?t say\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CORRECTION_PATTERNS]


def detect_correction(transcript: str) -> CorrectionResult:
    is_correction = any(pattern.search(transcript) for pattern in _COMPILED)
    return CorrectionResult(is_correction_attempt=is_correction, raw_transcript=transcript)


def annotate_transcript(transcript: str) -> str:
    result = detect_correction(transcript)
    if not result.is_correction_attempt:
        return transcript

    return (
        "[Note: the patient appears to be correcting a previous statement, which may "
        "have been mistranscribed. Treat the following as the authoritative, corrected "
        "information and disregard any conflicting earlier assumption in this conversation.]\n"
        f"{transcript}"
    )
