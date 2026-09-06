"""
Structured symptom slot extraction.

Runs as a SEPARATE call from the conversational response the patient hears --
extracts a structured JSON record using Ollama's format="json" mode.
"""

import json
import ollama
from dataclasses import dataclass, asdict
from typing import Optional, List


SLOT_SCHEMA_DESCRIPTION = """Extract structured information from this patient triage conversation.
Return ONLY a JSON object with exactly these fields, no other text:

{
  "chief_complaint": string (the main symptom, in a few words, or null if unclear),
  "duration": string or null (how long the symptom has lasted, e.g. "3 days", or null if not mentioned),
  "severity": "mild" | "moderate" | "severe" | null (based on patient's own description),
  "associated_symptoms": array of strings (other symptoms mentioned, empty array if none),
  "red_flag_present": boolean (true if any emergency-level symptom was mentioned: chest pain, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms),
  "recommended_department": string or null (e.g. "Emergency", "General Medicine", "Orthopedics", "Pediatrics", "Cardiology", or null if not yet determinable),
  "confidence": "low" | "medium" | "high" (how confident you are in this extraction given the conversation so far)
}"""


@dataclass
class SymptomSlots:
    chief_complaint: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    associated_symptoms: List[str] = None
    red_flag_present: bool = False
    recommended_department: Optional[str] = None
    confidence: str = "low"
    _raw_parse_error: Optional[str] = None

    def __post_init__(self):
        if self.associated_symptoms is None:
            self.associated_symptoms = []

    def to_dict(self):
        d = asdict(self)
        d.pop("_raw_parse_error", None)
        return d


def _conversation_to_text(conversation_history: list) -> str:
    lines = []
    for turn in conversation_history:
        speaker = "Patient" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def extract_symptom_slots(conversation_history: list, model_name: str = "qwen2.5:1.5b-instruct") -> SymptomSlots:
    conversation_text = _conversation_to_text(conversation_history)
    prompt = f"{SLOT_SCHEMA_DESCRIPTION}\n\nConversation:\n{conversation_text}"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )
        raw_text = response["message"]["content"]
        parsed = json.loads(raw_text)

        return SymptomSlots(
            chief_complaint=parsed.get("chief_complaint"),
            duration=parsed.get("duration"),
            severity=parsed.get("severity"),
            associated_symptoms=parsed.get("associated_symptoms", []) or [],
            red_flag_present=bool(parsed.get("red_flag_present", False)),
            recommended_department=parsed.get("recommended_department"),
            confidence=parsed.get("confidence", "low"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return SymptomSlots(confidence="low", _raw_parse_error=str(e))
