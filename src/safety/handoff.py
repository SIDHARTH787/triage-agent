"""
Human handoff logic.

Decides when to escalate to human staff, independent of what the SLM says.
"""

import re
from dataclasses import dataclass
from src.safety.red_flag_rules import check_red_flags, RedFlagResult


EMERGENCY_OVERRIDE_MESSAGE = (
    "Based on what you've described, this needs immediate attention. "
    "Please proceed to the Emergency Department right away. "
    "I'm alerting hospital staff now."
)

MAX_TURNS_BEFORE_HANDOFF = 6

STUCK_HANDOFF_MESSAGE = (
    "I want to make sure you get the right care. "
    "Let me connect you with a member of our staff who can help directly."
)

_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]{0,60}\]")

FABRICATION_FALLBACK_MESSAGE = (
    "I don't have that specific information on hand, but our front desk "
    "staff can help you with that directly."
)


@dataclass
class HandoffDecision:
    escalate: bool
    reason: str = None
    override_message: str = None
    red_flag_category: str = None


def contains_fabricated_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.search(text))


def evaluate_handoff(transcript: str, turn_count: int) -> HandoffDecision:
    red_flag_result: RedFlagResult = check_red_flags(transcript)

    if red_flag_result.triggered:
        return HandoffDecision(
            escalate=True,
            reason="red_flag",
            override_message=EMERGENCY_OVERRIDE_MESSAGE,
            red_flag_category=red_flag_result.category,
        )

    if turn_count >= MAX_TURNS_BEFORE_HANDOFF:
        return HandoffDecision(
            escalate=True,
            reason="stuck",
            override_message=STUCK_HANDOFF_MESSAGE,
        )

    return HandoffDecision(escalate=False)
