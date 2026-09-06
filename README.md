# Triage Agent — Phase 1: Sequential Baseline

Edge-deployed multimodal virtual triage agent. This is the **Phase 1 sequential
baseline**: ASR → SLM → TTS, wired together with no speculative execution yet.
This baseline is what the Phase 4 speculative pipeline gets benchmarked against.

## Setup

Requires a CUDA-capable GPU (4–8GB VRAM target) and Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate

# Install llama-cpp-python with CUDA support (important -- default pip install is CPU-only)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

pip install -r requirements.txt
```

### Download models

1. **SLM** — download a GGUF-quantized instruct model, e.g. Qwen2.5-1.5B-Instruct-Q4_K_M,
   and place it at `models/slm/qwen2.5-1.5b-instruct-q4_k_m.gguf`
   (path is configurable in `config.yaml`).

2. **TTS** — download a Piper voice (e.g. `en_US-lessac-medium`) — both the
   `.onnx` model and its `.onnx.json` config — into `models/tts/`.

3. **ASR** — no manual download needed; faster-whisper pulls the model
   automatically on first run (specify size in `config.yaml`, default `small`).

## Run

```bash
python -m src.pipeline
```

Speak into your microphone. The pipeline will:
1. Transcribe your speech (waits for end-of-utterance via VAD)
2. Generate a triage response with the SLM
3. Speak the response back via TTS
4. Log a latency breakdown (SLM time / TTS time / total) to the console

## Repo structure

```
src/
  asr/            streaming ASR (faster-whisper) + VAD end-of-utterance detection
  slm/            SLM engine (llama.cpp) + prompt templates
  tts/            Piper TTS engine, built cancellable from day one (needed later)
  speculative/    Phase 4 -- speculator, divergence detector, rollback manager
  scheduler/      Phase 3 -- GPU/CPU async task pinning
  safety/         Phase 2 -- GBNF grammar, red-flag rules, human handoff
  cv/             Phase 3 -- MediaPipe engagement detection
  pipeline.py     Main entry point (sequential baseline)
eval/             Vignette test cases + benchmarking scripts
config.yaml       All tunables (model paths, thresholds, VRAM budget)
```

## What's next (not yet implemented)

- [ ] Phase 2: `src/safety/` — GBNF grammar for slot-filling, red-flag rules, handoff logic
- [ ] Phase 3: `src/scheduler/`, `src/cv/` — resource-aware scheduling + engagement detection
- [ ] Phase 4: `src/speculative/` — the actual speculative pipeline execution
- [ ] `eval/` — vignette dataset + sequential-vs-speculative benchmarking

## Notes on latency measurement

`pipeline.py` logs three numbers per turn: SLM generation time, TTS synthesis+playback
time, and total time from ASR-final to speech-done. **This total is your Phase 1
baseline number** — write it down once things are running, since it's the comparison
point for the entire speculative execution contribution.
