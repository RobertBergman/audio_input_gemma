# Gemma Microphone Test

This uses Hugging Face Transformers to run an audio-capable Gemma model locally against
microphone input. The recorder captures mono 16 kHz `float32` audio normalized to `[-1, 1]`
and limits clips to 30 seconds.

## Setup

```powershell
uv sync
```

Gemma weights are gated on Hugging Face. Accept the model license for
`google/gemma-4-E2B-it`, then authenticate:

```powershell
Copy-Item .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
```

## Run

```powershell
uv run python gemma_mic.py --seconds 5
```

Keep STT running in repeated 5-second chunks:

```powershell
uv run python gemma_mic.py --continuous --seconds 5
```

For continuous capture without missing microphone audio while Gemma is transcribing,
use buffered mode. This keeps recording into a bounded queue and drops oldest chunks only
if inference falls too far behind:

```powershell
uv run python gemma_mic.py --buffered --seconds 5 --queue-seconds 120
```

If silence is still being transcribed, raise both gates:

```powershell
uv run python gemma_mic.py --buffered --seconds 5 --silence-rms 0.005 --min-peak 0.03
```

To measure your mic's background noise without loading Gemma:

```powershell
uv run python gemma_mic.py --calibrate-silence --seconds 5
```

To see live chunk levels:

```powershell
uv run python gemma_mic.py --buffered --seconds 5 --debug-audio --no-skip-blank
```

Set `--silence-rms 0 --min-peak 0` to disable silence skipping.

The default prompt asks Gemma to output nothing for non-speech, and the runner suppresses
exact `1.7` outputs by default because that is a common prompt-copy hallucination on
non-speech audio. Add more exact suppressions if needed:

```powershell
uv run python gemma_mic.py --buffered --filter-text "1.7" --filter-text "3"
```

Use a different prompt for speech understanding:

```powershell
uv run python gemma_mic.py --seconds 10 --prompt "Summarize what the speaker is asking for."
```

The default Hugging Face model is `google/gemma-4-E2B-it`.
