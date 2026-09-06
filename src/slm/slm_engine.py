"""
SLM engine module -- Ollama backend.

Swapped from llama-cpp-python to Ollama because llama-cpp-python requires
compiling from source on Windows (needs MSVC Build Tools + CMake + CUDA Toolkit
correctly wired together), which is painful to get right. Ollama ships a
standalone Windows installer with CUDA support prebuilt, and exposes a local
HTTP API -- no compilation needed on our end.

Interface is kept identical to the original llama.cpp version
(generate_stream / generate / cancel) so nothing else in the pipeline changes.

Setup (one-time, outside Python):
  1. Install Ollama: https://ollama.com/download/windows
  2. ollama pull qwen2.5:1.5b-instruct
  3. pip install ollama
"""

import ollama


class SLMEngine:
    def __init__(self, config: dict):
        """
        config: the 'slm' section of config.yaml
        Expects config["model_name"] = "qwen2.5:1.5b-instruct" (Ollama model tag)
        """
        self.cfg = config
        self.model_name = config.get("model_name", "qwen2.5:1.5b-instruct")
        self._cancel_flag = False

        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                "Could not reach Ollama. Is the Ollama service running? "
                "Install from https://ollama.com/download/windows and run "
                f"'ollama pull {self.model_name}' first."
            ) from e

    def cancel(self):
        """Signal the current generation to stop at the next token boundary.
        Used by rollback_manager.py in Phase 4 -- harmless no-op in Phase 1."""
        self._cancel_flag = True

    def _build_messages(self, transcript: str, conversation_history=None):
        from .prompt_templates import SYSTEM_PROMPT
        from .correction_handling import annotate_transcript

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (conversation_history or []):
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["text"]})

        annotated_transcript = annotate_transcript(transcript)
        messages.append({"role": "user", "content": annotated_transcript})
        return messages

    def generate_stream(self, transcript: str, conversation_history=None, grammar=None):
        """
        Streams response tokens for a given (possibly partial) transcript.
        Yields dicts: {"token": str, "done": bool, "cancelled": bool}
        """
        self._cancel_flag = False
        messages = self._build_messages(transcript, conversation_history)

        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            options={
                "temperature": self.cfg.get("temperature", 0.3),
                "num_predict": self.cfg.get("max_new_tokens", 200),
            },
        )

        for chunk in stream:
            if self._cancel_flag:
                yield {"token": "", "done": True, "cancelled": True}
                return
            token_text = chunk["message"]["content"]
            finished = chunk.get("done", False)
            yield {"token": token_text, "done": finished, "cancelled": False}
            if finished:
                return

    def generate(self, transcript: str, conversation_history=None, grammar=None) -> str:
        """Non-streaming convenience wrapper -- collects full response."""
        out = []
        for chunk in self.generate_stream(transcript, conversation_history, grammar):
            out.append(chunk["token"])
            if chunk["done"]:
                break
        return "".join(out).strip()
