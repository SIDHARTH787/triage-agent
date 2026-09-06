"""
Streaming ASR module.

Wraps faster-whisper to provide:
  - continuous microphone capture
  - periodic "partial hypothesis" emission (re-transcribing the growing buffer)
  - VAD-based end-of-utterance detection -> "final hypothesis"

This is intentionally simple for Phase 1 (sequential baseline). Partial hypotheses
aren't used for speculation yet -- they're just logged/available for Phase 4 later.

Event contract (used by later speculative code too, so keep this stable):
    {
        "type": "partial" | "final",
        "text": str,
        "timestamp_ms": int,
        "is_stable": bool          # heuristic: has this text stopped changing recently?
    }
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


class StreamingASR:
    def __init__(self, config: dict, on_event=None):
        self.cfg = config
        self.on_event = on_event or (lambda e: None)

        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        self.vad = webrtcvad.Vad(2)
        self.silence_ms_threshold = config.get("vad_silence_ms", 500)
        self.partial_interval_ms = config.get("partial_emit_interval_ms", 250)

        self.model = WhisperModel(
            config.get("model_size", "small"),
            device=config.get("device", "cuda"),
            compute_type=config.get("compute_type", "int8_float16"),
        )

        final_model_size = config.get("final_model_size")
        if final_model_size and final_model_size != config.get("model_size", "small"):
            self.final_model = WhisperModel(
                final_model_size,
                device=config.get("final_model_device", config.get("device", "cpu")),
                compute_type=config.get("final_model_compute_type", config.get("compute_type", "int8")),
            )
        else:
            self.final_model = self.model

        self.input_device = config.get("input_device", None)

        self._audio_q = queue.Queue()
        self._running = False
        self._last_partial_text = ""
        self._stable_count = 0
        self._muted = False

    def mute(self):
        """Call this right before TTS playback starts."""
        self._muted = True

    def unmute(self):
        """Call this right after TTS playback finishes."""
        self._muted = False

    def _audio_callback(self, indata, frames, time_info, status):
        if self._muted:
            return
        self._audio_q.put(indata.copy())

    def start(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False
        self._stream.stop()
        self._stream.close()

    def _run_loop(self):
        try:
            self._run_loop_inner()
        except Exception:
            import traceback
            print("StreamingASR worker thread crashed:")
            traceback.print_exc()

    def _run_loop_inner(self):
        buffer = np.zeros((0,), dtype=np.int16)
        silence_ms = 0
        last_partial_emit = 0

        while self._running:
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_flat = frame.flatten()
            buffer = np.concatenate([buffer, frame_flat])

            is_speech = self.vad.is_speech(frame_flat.tobytes(), self.sample_rate)
            if is_speech:
                silence_ms = 0
            else:
                silence_ms += self.frame_ms

            now_ms = int(time.time() * 1000)

            if buffer.size > 0 and (now_ms - last_partial_emit) >= self.partial_interval_ms:
                self._emit_partial(buffer)
                last_partial_emit = now_ms

            min_utterance_samples = int(self.sample_rate * 0.5)
            if silence_ms >= self.silence_ms_threshold and buffer.size > min_utterance_samples:
                self._emit_final(buffer)
                buffer = np.zeros((0,), dtype=np.int16)
                silence_ms = 0
                self._last_partial_text = ""
                self._stable_count = 0

    def _transcribe(self, audio_int16: np.ndarray) -> str:
        audio_float = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float,
            language=self.cfg.get("language", "en"),
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        return " ".join(seg.text for seg in segments).strip()

    def _emit_partial(self, buffer: np.ndarray):
        text = self._transcribe(buffer)
        if not text:
            return

        is_stable = (text == self._last_partial_text)
        self._stable_count = self._stable_count + 1 if is_stable else 0
        self._last_partial_text = text

        self.on_event({
            "type": "partial",
            "text": text,
            "timestamp_ms": int(time.time() * 1000),
            "is_stable": self._stable_count >= 2,
        })

    def _emit_final(self, buffer: np.ndarray):
        audio_float = buffer.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float,
            language=self.cfg.get("language", "en"),
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        text = " ".join(seg.text for seg in segments).strip()

        if not text:
            return

        self.on_event({
            "type": "final",
            "text": text,
            "timestamp_ms": int(time.time() * 1000),
            "is_stable": True,
        })
