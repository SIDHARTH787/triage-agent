"""
Main pipeline entry point -- Phase 1 (sequential baseline).

Flow: mic -> ASR (final transcript only, partials logged but unused here) ->
      SLM (full generation) -> TTS (spoken response).

This is deliberately the "slow but correct" version. Its measured latency
becomes the baseline that the Phase 4 speculative pipeline is benchmarked
against in eval/benchmark_latency.py.

Run with: python -m src.pipeline
"""

import time
import yaml
import logging

from src.asr.streaming_asr import StreamingASR
from src.slm.slm_engine import SLMEngine
from src.tts.piper_engine import TTSEngine
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.safety.slot_extraction import extract_symptom_slots
from src.scheduler.async_scheduler import ResourceScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")


class SequentialTriagePipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.slm = SLMEngine(self.cfg["slm"])
        self.tts = TTSEngine(self.cfg["tts"])
        self.conversation_history = []
        self.turn_count = 0
        self.latest_slots = None

        self.scheduler = ResourceScheduler(self.cfg.get("pipeline", {}))

        self.engagement = None
        cv_cfg = self.cfg.get("cv", {})
        if cv_cfg.get("enabled", False):
            from src.cv.engagement_detector import EngagementDetector
            self.engagement = EngagementDetector(
                camera_index=cv_cfg.get("camera_index", 0),
                check_interval_s=cv_cfg.get("check_interval_s", 1.0),
                absence_grace_s=cv_cfg.get("absence_grace_s", 5.0),
                cascade_path=cv_cfg.get("cascade_path"),
            )

        self.asr = StreamingASR(self.cfg["asr"], on_event=self._on_asr_event)

        self.interrupt_listener = None
        if self.cfg.get("pipeline", {}).get("keyboard_interrupt_enabled", True):
            from src.tts.keyboard_interrupt import KeyboardInterruptListener
            try:
                self.interrupt_listener = KeyboardInterruptListener(self.tts, self.asr)
            except RuntimeError as e:
                log.warning("Keyboard interrupt listener unavailable: %s", e)

    def _on_asr_event(self, event: dict):
        if event["type"] == "partial":
            # Phase 1: partials are just logged. Phase 4 will hook the
            # speculator in here instead.
            log.debug(f"partial: {event['text']}")
            return

        # event["type"] == "final" -> run the sequential turn
        t0 = time.time()
        transcript = event["text"]
        if not transcript:
            return

        if self.engagement is not None and not self.engagement.is_present():
            log.info("No patient detected by camera -- ignoring transcript: %r", transcript)
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
        if handoff_decision.escalate:
            response = handoff_decision.override_message
            log.warning(
                "HANDOFF TRIGGERED | reason=%s | red_flag_category=%s",
                handoff_decision.reason,
                handoff_decision.red_flag_category,
            )
        else:
            response = self.slm.generate(transcript, self.conversation_history)
            if contains_fabricated_placeholder(response):
                log.warning("FABRICATION DETECTED in SLM response -- replacing with safe fallback. Original: %r", response)
                response = FABRICATION_FALLBACK_MESSAGE
        t_slm_end = time.time()

        self.conversation_history.append({"role": "assistant", "text": response})
        log.info(f"Agent response: {response}")

        t_tts_start = time.time()
        self.asr.mute()
        self.tts.speak(response, blocking=True)
        time.sleep(0.3)
        self.asr.unmute()
        t_tts_end = time.time()

        log.info(
            "Latency breakdown | SLM: %.0fms | TTS: %.0fms | total (ASR-final -> speech done): %.0fms",
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

    def start(self):
        log.info("Starting sequential triage pipeline (non-blocking). Speak into the mic.")
        if self.engagement is not None:
            self.engagement.start()
        if self.interrupt_listener is not None:
            self.interrupt_listener.start()
        self.asr.start()

    def stop(self):
        log.info("Stopping.")
        self.asr.stop()
        if self.engagement is not None:
            self.engagement.stop()
        if self.interrupt_listener is not None:
            self.interrupt_listener.stop()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    pipeline = SequentialTriagePipeline()
    pipeline.run()
