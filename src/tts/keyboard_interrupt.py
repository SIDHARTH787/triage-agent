"""
Keyboard interrupt listener -- press SPACE to interrupt the agent mid-response.
Stand-in for real voice barge-in until a mic setup supporting AEC is available.
"""

import time
import threading
import logging

log = logging.getLogger("keyboard_interrupt")

try:
    import msvcrt
    _MSVCRT_AVAILABLE = True
except ImportError:
    _MSVCRT_AVAILABLE = False


class KeyboardInterruptListener:
    def __init__(self, tts_engine, asr_engine, key: bytes = b" ", poll_interval_s: float = 0.05):
        if not _MSVCRT_AVAILABLE:
            raise RuntimeError("KeyboardInterruptListener currently only supports Windows (uses msvcrt).")

        self.tts_engine = tts_engine
        self.asr_engine = asr_engine
        self.key = key
        self.poll_interval_s = poll_interval_s

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Keyboard interrupt listener started (press SPACE to interrupt the agent while it is speaking).")

    def stop(self):
        self._running = False

    def _run_loop(self):
        try:
            while self._running:
                if msvcrt.kbhit():
                    pressed = msvcrt.getch()
                    if pressed == self.key and self.tts_engine.is_speaking():
                        log.info("INTERRUPT -- spacebar pressed while agent was speaking. Cancelling.")
                        self.tts_engine.cancel()
                        self.asr_engine.unmute()
                time.sleep(self.poll_interval_s)
        except Exception:
            import traceback
            log.error("KeyboardInterruptListener thread crashed:")
            traceback.print_exc()
