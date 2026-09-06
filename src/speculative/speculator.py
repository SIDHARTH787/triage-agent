"""
Speculator -- triggers SLM generation on stable partial ASR hypotheses,
buffers the response on a background thread, and resolves against the
final transcript once available.
"""

import time
import threading
import logging

from src.slm.slm_engine import SLMEngine

log = logging.getLogger("speculator")


class Speculator:
    def __init__(self, slm_config: dict):
        self.engine = SLMEngine(slm_config)

        self._lock = threading.Lock()
        self._thread = None
        self._buffer = []
        self._triggering_partial = None
        self._generation_done = threading.Event()
        self._cancelled = False
        self._start_ts = None

    def is_active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def has_result(self) -> bool:
        return self._generation_done.is_set() and not self._cancelled

    def start_speculation(self, partial_text: str, conversation_history: list):
        self.cancel()

        with self._lock:
            self._buffer = []
            self._triggering_partial = partial_text
            self._generation_done.clear()
            self._cancelled = False
            self._start_ts = time.time()

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(partial_text, conversation_history),
            daemon=True,
        )
        self._thread.start()
        log.info("Speculation started on partial: %r", partial_text)

    def _run_generation(self, partial_text: str, conversation_history: list):
        try:
            for chunk in self.engine.generate_stream(partial_text, conversation_history):
                if chunk.get("cancelled"):
                    return
                with self._lock:
                    self._buffer.append(chunk["token"])
                if chunk["done"]:
                    break
        except Exception:
            import traceback
            log.error("Speculative generation thread crashed:")
            traceback.print_exc()
        finally:
            self._generation_done.set()

    def cancel(self):
        self.engine.cancel()
        with self._lock:
            self._cancelled = True

    def resolve(self, final_text: str, wait_timeout_s: float = 3.0):
        if self._thread is None:
            return None

        finished_in_time = self._generation_done.wait(timeout=wait_timeout_s)
        if not finished_in_time:
            log.warning(
                "Speculative generation did not finish within %.1fs of final transcript.",
                wait_timeout_s,
            )
            self.cancel()
            return None

        with self._lock:
            if self._cancelled:
                return None
            return "".join(self._buffer).strip()

    def get_triggering_partial(self):
        with self._lock:
            return self._triggering_partial
