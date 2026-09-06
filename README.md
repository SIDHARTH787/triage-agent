# Speculative Multimodal Hospital Triage Agent

A real-time, edge-deployed multimodal virtual triage agent combining streaming ASR, a quantized small language model (SLM), and TTS — designed to run entirely on consumer-grade hardware (tested on an NVIDIA GTX 1650, 4GB VRAM). Speculative pipeline execution reduces the ASR→LLM serialization bottleneck, and a rule-based clinical safety layer operates independently of, and with precedence over, the language model.

Group 14 | Mentor: Dr. Aneesh G. Nath

## Key Features

- **Streaming ASR** (faster-whisper, dual-model: fast model for partials, accurate model for final transcription)
- **Conversational triage reasoning** via a quantized Qwen2.5-1.5B-Instruct model served locally through Ollama
- **TTS synthesis** via Piper, with markdown-stripping and cancellable playback
- **Rule-based clinical safety layer**: red-flag symptom detection, human handoff, anti-fabrication filtering, patient self-correction handling — all operating outside the LLM's control
- **Structured clinical record extraction** (JSON) alongside the spoken conversation
- **CV-based patient presence gating** (OpenCV Haar cascade) to ignore transcripts when no patient is present
- **Resource-aware VRAM scheduling**, measured and logged in real time
- **Speculative pipeline execution**: SLM generation begins on partial ASR hypotheses before the patient finishes speaking, with a divergence-detection mechanism to commit or roll back once the final transcript is available — empirically shown to reduce SLM-stage latency by ~49% with zero false commits on tested cases

## Architecture

```
Mic --> Streaming ASR (faster-whisper, dual-model)
              |
              |--> partial hypotheses --> [Speculative mode only] Speculator --> SLM (background)
              |
              +--> final transcript --> Safety Layer (red-flag / handoff / fabrication check)
                                              |
                                              +--> SLM (Qwen2.5-1.5B via Ollama)
                                              |
                                              +--> TTS (Piper) --> Speaker
                                              |
                                              +--> Structured slot extraction (JSON record)

Camera --> CV Engagement Detector (gates whether transcripts are processed)
GPU/CPU --> Resource Scheduler (VRAM monitoring)
```

Two pipeline variants are provided for direct A/B comparison:
- `src/pipeline.py` — sequential baseline (no speculation)
- `src/pipeline_speculative.py` — speculative execution variant

## Repo Structure

```
src/
  asr/            Streaming ASR (faster-whisper), dual-model partial/final transcription
  slm/             SLM engine (Ollama), prompt templates, correction handling
  tts/             Piper TTS engine, keyboard interrupt listener
  speculative/     Speculator, divergence detector, rollback manager
  scheduler/       GPU/CPU resource scheduler (VRAM budget monitoring)
  safety/          Red-flag rules, human handoff, fabrication detection, structured slot extraction
  cv/              Camera-based patient presence detection
  pipeline.py             Sequential baseline pipeline
  pipeline_speculative.py Speculative execution pipeline
eval/
  benchmark_latency.py    Sequential vs. speculative latency benchmark
config.yaml        All tunables (model paths, thresholds, VRAM budget)
config.example.yaml  Template config with placeholder paths
```

## Setup

### Requirements
- Python 3.9+ (tested on Windows with Conda)
- NVIDIA GPU with CUDA support (4GB+ VRAM recommended)
- A working microphone and (optionally) a webcam for CV presence detection

### 1. Install Ollama

Ollama handles SLM serving with GPU acceleration, sidestepping the need to compile `llama-cpp-python` from source (which requires MSVC Build Tools on Windows).

Download from [ollama.com/download](https://ollama.com/download), then:
```bash
ollama pull qwen2.5:1.5b-instruct
```

If your C: drive is low on space, redirect Ollama's model storage before pulling:
```powershell
setx OLLAMA_MODELS "D:\ollama_models"
```
(requires a fresh terminal window to take effect)

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`webrtcvad` and other packages requiring compilation may need prebuilt-wheel alternatives on Windows — see `requirements.txt` comments for specifics (e.g. `webrtcvad-wheels` instead of `webrtcvad`).

### 3. Download models

- **TTS voice**: download a Piper voice (e.g. `en_US-lessac-medium`) — both the `.onnx` and `.onnx.json` files — into `models/tts/`. Available at the [Piper voices repository](https://huggingface.co/rhasspy/piper-voices).
- **ASR**: no manual download needed — faster-whisper pulls models automatically on first run.
- **CV cascade** (optional, only if `cv.enabled: true`): if your OpenCV install doesn't bundle Haar cascade files, download `haarcascade_frontalface_default.xml` from [OpenCV's GitHub repo](https://github.com/opencv/opencv/tree/master/data/haarcascades) into `models/cv/`.

### 4. Configure

Copy `config.example.yaml` to `config.yaml` and adjust:
- `asr.input_device` — run the device-listing snippet in the docs below to find your microphone's index
- `tts.voice_model` — absolute path to your downloaded Piper voice
- `cv.enabled` / `cv.camera_index` — if using presence detection
- `pipeline.max_vram_gb` — match your actual GPU's VRAM

**Finding your microphone device index:**
```python
import sounddevice as sd
print(sd.query_devices())
```

## Usage

### Run the sequential baseline
```bash
python -m src.pipeline
```

### Run the speculative pipeline
```bash
python -m src.pipeline_speculative
```

Both listen continuously on the configured microphone. Speak a symptom description; the agent will ask follow-up questions, offer general guidance for minor symptoms, and escalate immediately to Emergency (bypassing the LLM entirely) if red-flag symptoms are detected.

**Interrupting the agent:** press SPACE while the agent is speaking to cancel its response (terminal/script mode only — not currently supported when running via Jupyter, since keyboard capture requires an attached console).

### Running from Jupyter

Use the non-blocking `start()`/`stop()` interface instead of `run()`, so the pipeline runs in the background while you use other cells:

```python
from src.pipeline import SequentialTriagePipeline
# or: from src.pipeline_speculative import SpeculativeTriagePipeline

pipeline = SequentialTriagePipeline(config_path="config.yaml")
pipeline.start()   # non-blocking

# ... use other cells, e.g. check camera feed, inspect pipeline.latest_slots ...

pipeline.stop()
```

### Running the latency benchmark

Compares sequential vs. speculative SLM-stage latency using a fixed set of realistic partial/final transcript pairs (no live audio needed):

```bash
python -m eval.benchmark_latency
```

## Configuration Reference

See `config.yaml` for the full annotated list. Key sections:

| Section | Purpose |
|---|---|
| `asr` | Model sizes (partial vs. final), device, mic input index, VAD/silence thresholds |
| `slm` | Ollama model name, generation temperature, max tokens |
| `tts` | Piper voice path, sample rate |
| `pipeline` | Execution mode, VRAM budget, keyboard interrupt toggle |
| `cv` | Enable/disable presence detection, camera index, absence grace period |
| `speculative` | Divergence threshold for commit/rollback decisions |

## Known Limitations

- Domain-boundary enforcement is inconsistent — the SLM occasionally answers clearly off-topic questions despite prompt instructions (a documented limitation of small-model instruction-following reliability, not fixed by design)
- Fabrication detection only catches bracket-placeholder-style output; confidently-stated but unverified specific claims without that pattern can still slip through
- Keyboard interrupt requires a real console (works via `python -m src.pipeline`, not through Jupyter)
- CV presence detection uses coarse Haar-cascade face detection, not gaze/attention tracking
- True voice barge-in (interrupting by speaking, not pressing a key) would require acoustic echo cancellation, not yet implemented — see `src/tts/keyboard_interrupt.py` for details and rationale

## Results Summary

Benchmarked on 10 realistic partial/final transcript pairs (drawn from live testing, including genuine ASR artifacts like truncated words and homophone substitutions):

- **Commit rate**: 50%
- **False commits**: 0
- **Mean latency reduction**: 49.1% (SLM-stage only)
- **Rolled-back cases**: no measurable latency penalty vs. sequential baseline

See `eval/benchmark_latency.py` for the full test set and methodology.
