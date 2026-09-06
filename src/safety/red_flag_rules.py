"""
Red-flag rule-based escalation.

Runs OUTSIDE the SLM -- rule-based keyword/phrase matching on the raw
transcript, independent of whatever the model decides to say. This is a
core safety requirement: we cannot rely solely on the SLM's judgment to
catch dangerous symptoms, especially in a small quantized model under 2B params.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RedFlagResult:
    triggered: bool
    matched_phrases: list = field(default_factory=list)
    category: str = None


RED_FLAG_PATTERNS = {
    "cardiac": [
        r"\bchest pain\b",
        r"\bchest (feels|feeling) (tight|heavy|crushing)\b",
        r"\bpain (in|down) my (left )?arm\b",
        r"\bheart (racing|pounding) and (dizzy|faint)\b",
    ],
    "respiratory": [
        r"\b(can'?t|cannot) breathe\b",
        r"\bdifficulty breathing\b",
        r"\bshort(ness)? of breath\b",
        r"\bgasping for air\b",
        r"\bturning blue\b",
        r"\blips\b.{0,25}\bblue\b",
        r"\bhard(ly)? to breathe\b",
    ],
    "neurological": [
        r"\b(sudden|severe) (headache|confusion)\b",
        r"\bslurred speech\b",
        r"\bface\b.{0,20}\bdroop",
        r"\b(can'?t|cannot) move (my|one side)\b",
        r"\blost consciousness\b",
        r"\bpassed out\b",
        r"\bfainted\b",
        r"\bseizure\b",
        r"\bone side (of|weak)",
    ],
    "bleeding": [
        r"\bsevere bleeding\b",
        r"\bwon'?t stop bleeding\b",
        r"\bvomiting blood\b",
        r"\bblood in (my )?(stool|vomit)\b",
        r"\bheavy blood loss\b",
    ],
    "allergic": [
        r"\bthroat (is )?closing\b",
        r"\bswelling (of|in) (my )?(face|throat|lips)\b",
        r"\banaphyla",
    ],
}

_COMPILED = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in RED_FLAG_PATTERNS.items()
}


def check_red_flags(transcript: str) -> RedFlagResult:
    matched = []
    fired_category = None

    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            match = pattern.search(transcript)
            if match:
                matched.append(match.group(0))
                fired_category = fired_category or category

    return RedFlagResult(
        triggered=len(matched) > 0,
        matched_phrases=matched,
        category=fired_category,
    )
