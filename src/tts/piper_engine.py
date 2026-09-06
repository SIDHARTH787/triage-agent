"""
TTS engine module wrapping Piper.

Provides incremental (sentence-chunked) synthesis and playback, with a
cancel() method. In Phase 1 (sequential) cancel() is never actually needed --
but the rollback_manager in Phase 4 depends on this existing, so it's built
in from the start rather than retrofitted later.
"""

import threading
import numpy as np
import sounddevice as sd
from piper import PiperVoice


class TTSEngine:
    def __init__(self, config: dict):
        """
        config: the \'tts\' section of config.yaml
        """
        self.cfg = config
        self.voice = PiperVoice.load(config["voice_model"])
        self.sample_rate = self.voice.config.sample_rate

        self._cancel_flag = threading.Event()
        self._playback_thread = None

    def cancel(self):
        """Stop playback ASAP. Safe to call even if nothing is playing."""
        self._cancel_flag.set()
        sd.stop()

    def speak(self, text: str, blocking: bool = True):
        self._cancel_flag.clear()

        def _run():
            for audio_chunk in self._synthesize_chunks(text):
                if self._cancel_flag.is_set():
                    return
                sd.play(audio_chunk, samplerate=self.sample_rate)
                sd.wait()
                if self._cancel_flag.is_set():
                    sd.stop()
                    return

        if blocking:
            _run()
        else:
            self._playback_thread = threading.Thread(target=_run, daemon=True)
            self._playback_thread.start()

    @staticmethod
    def _strip_markdown(text: str) -> str:
        import re
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)
        text = text.replace("*", "")
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    def _synthesize_chunks(self, text: str):
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            for audio_chunk in self.voice.synthesize(sentence):
                yield audio_chunk.audio_float_array

    @staticmethod
    def _split_sentences(text: str):
        import re
        return re.split(r"(?<=[.!?])\s+", text.strip())
