"""
Speculative pipeline -- Phase 4. Subclasses SequentialTriagePipeline.
"""

import time
import logging

from src.pipeline import SequentialTriagePipeline, log
from src.speculative.speculator import Speculator
from src.speculative.rollback_manager import resolve_speculation
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.safety.slot_extraction import extract_symptom_slots

MIN_WORDS_TO_SPECULATE = 3


class SpeculativeTriagePipeline(SequentialTriagePipeline):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path=config_path)

        self.speculator = Speculator(self.cfg["slm"])
        self.divergence_threshold = self.cfg.get("speculative", {}).get("divergence_threshold", 0.7)
        self._last_speculated_partial = None

        self.asr.on_event = self._on_asr_event

    def _on_asr_event(self, event: dict):
        if event["type"] == "partial":
            self._handle_partial(event)
            return
        self._handle_final(event)

    def _handle_partial(self, event: dict):
        text = event["text"]
        if len(text.split()) < MIN_WORDS_TO_SPECULATE:
            return
        if self._last_speculated_partial is not None:
            return

        self._last_speculated_partial = text
        self.speculator.start_speculation(text, self.conversation_history)

    def _handle_final(self, event: dict):
        t0 = time.time()
        transcript = event["text"]
        if not transcript:
            return

        if self.engagement is not None and not self.engagement.is_present():
            log.info("No patient detected by camera -- ignoring transcript: %r", transcript)
            self.speculator.cancel()
            self._last_speculated_partial = None
            return

        vram_status = self.scheduler.check_budget()
        if vram_status["used_mb"] is not None:
            log.info(
                "VRAM: %d/%d MB (budget: %d MB)%s",
                vram_status["used_mb"], vram_status["total_mb"], vram_status["budget_mb"],
                " -- OVER BUDGET" if vram_status["over_budget"] else "",
            )

        log.info(f"Patient said: {transcript}")
        self.conversation_history.append({"role": "user", "text": transcript})
        self.turn_count += 1

        handoff_decision = evaluate_handoff(transcript, self.turn_count)

        t_slm_start = time.time()
        speculative_used = False

        if handoff_decision.escalate:
            self.speculator.cancel()
            response = handoff_decision.override_message
            log.warning(
                "HANDOFF TRIGGERED | reason=%s | red_flag_category=%s",
                handoff_decision.reason,
                handoff_decision.red_flag_category,
            )
        else:
            rollback_decision = resolve_speculation(
                self.speculator, self.tts, transcript,
                divergence_threshold=self.divergence_threshold,
            )
            if rollback_decision.committed:
                response = rollback_decision.speculative_response
                speculative_used = True
                log.info("Using SPECULATIVE response (no regeneration needed).")
            else:
                response = self.slm.generate(transcript, self.conversation_history)
                log.info("Generated FRESH response (speculation missed or unavailable).")

            if contains_fabricated_placeholder(response):
                log.warning("FABRICATION DETECTED -- replacing with safe fallback. Original: %r", response)
                response = FABRICATION_FALLBACK_MESSAGE

        t_slm_end = time.time()
        self._last_speculated_partial = None

        self.conversation_history.append({"role": "assistant", "text": response})
        log.info(f"Agent response: {response}")

        t_tts_start = time.time()
        self.asr.mute()
        self.tts.speak(response, blocking=True)
        time.sleep(0.3)
        self.asr.unmute()
        t_tts_end = time.time()

        log.info(
            "Latency breakdown | speculative_used=%s | SLM/resolve: %.0fms | TTS: %.0fms | total: %.0fms",
            speculative_used,
            (t_slm_end - t_slm_start) * 1000,
            (t_tts_end - t_tts_start) * 1000,
            (t_tts_end - t0) * 1000,
        )

        t_extract_start = time.time()
        self.latest_slots = extract_symptom_slots(
            self.conversation_history,
            model_name=self.cfg["slm"].get("model_name", "qwen2.5:1.5b-instruct"),
        )
        t_extract_end = time.time()
        log.info(
            "Structured slots (%.0fms): %s",
            (t_extract_end - t_extract_start) * 1000,
            self.latest_slots.to_dict(),
        )
