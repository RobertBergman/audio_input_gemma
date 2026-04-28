import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from scipy.io import wavfile
from transformers import AutoModelForMultimodalLM, AutoProcessor


DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30.0
DEFAULT_BLOCK_SECONDS = 0.5
DEFAULT_QUEUE_SECONDS = 120.0
DEFAULT_SILENCE_RMS = 0.003
DEFAULT_MIN_PEAK = 0.02
DEFAULT_ASR_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* If there is no spoken language, output nothing."
)
DEFAULT_FILTER_TEXTS = ("1.7",)


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32), dtype=np.float32)))


def audio_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def should_skip_audio(audio: np.ndarray, silence_rms: float, min_peak: float) -> bool:
    rms_too_low = silence_rms > 0 and audio_rms(audio) < silence_rms
    peak_too_low = min_peak > 0 and audio_peak(audio) < min_peak
    if silence_rms > 0 and min_peak > 0:
        return rms_too_low and peak_too_low
    return rms_too_low or peak_too_low


def describe_audio(audio: np.ndarray) -> str:
    return f"rms={audio_rms(audio):.5f} peak={audio_peak(audio):.5f}"


def is_filtered_transcript(text: str, filter_texts: tuple[str, ...]) -> bool:
    normalized = " ".join(text.strip().split()).casefold()
    return bool(normalized) and normalized in {item.casefold() for item in filter_texts}


class GemmaTranscriber:
    def __init__(self, model_id: str) -> None:
        token = os.getenv("HF_TOKEN") or None
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            token=token,
        ).eval()

    def transcribe_audio(self, audio: np.ndarray, prompt: str, max_new_tokens: int) -> str:
        audio_path = write_temp_wav(audio)
        try:
            messages = build_messages(audio_path, prompt)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            prompt_length = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_tokens = output[0][prompt_length:]
            return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
        finally:
            audio_path.unlink(missing_ok=True)


def record_microphone(seconds: float) -> np.ndarray:
    if seconds <= 0:
        raise ValueError("--seconds must be greater than 0.")
    if seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"Gemma audio clips are limited to {MAX_AUDIO_SECONDS:g} seconds.")

    print(f"Recording {seconds:g}s from the default microphone...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    # Gemma expects mono 16 kHz float32 samples normalized to [-1, 1].
    return np.clip(audio.reshape(-1), -1.0, 1.0).astype(np.float32)


def write_temp_wav(audio: np.ndarray) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    wavfile.write(path, SAMPLE_RATE, audio)
    return path


def build_messages(audio_path: Path, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(audio_path)},
            ],
        }
    ]


def generate_from_microphone(model_id: str, seconds: float, prompt: str, max_new_tokens: int) -> str:
    transcriber = GemmaTranscriber(model_id)
    return transcriber.transcribe_audio(
        audio=record_microphone(seconds),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def run_continuous_stt(
    transcriber: GemmaTranscriber,
    seconds: float,
    prompt: str,
    max_new_tokens: int,
    skip_blank: bool,
    silence_rms: float,
    min_peak: float,
    debug_audio: bool,
    filter_texts: tuple[str, ...],
) -> None:
    print("Continuous STT started. Press Ctrl+C to stop.")
    chunk_index = 1
    while True:
        audio = record_microphone(seconds)
        audio_description = describe_audio(audio)
        if debug_audio:
            print(f"[{chunk_index:04d}] {audio_description}", file=sys.stderr, flush=True)

        if should_skip_audio(audio, silence_rms, min_peak):
            if not skip_blank:
                print(f"[{chunk_index:04d}] <silence> {audio_description}", flush=True)
            chunk_index += 1
            continue

        result = transcriber.transcribe_audio(
            audio=audio,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        if is_filtered_transcript(result, filter_texts):
            if not skip_blank:
                print(f"[{chunk_index:04d}] <filtered> {result}", flush=True)
            chunk_index += 1
            continue

        if result or not skip_blank:
            print(f"[{chunk_index:04d}] {result}", flush=True)
        chunk_index += 1


class MicrophoneRingBuffer:
    def __init__(self, window_seconds: float, block_seconds: float, queue_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("--seconds must be greater than 0.")
        if window_seconds > MAX_AUDIO_SECONDS:
            raise ValueError(f"Gemma audio clips are limited to {MAX_AUDIO_SECONDS:g} seconds.")
        if block_seconds <= 0:
            raise ValueError("--block-seconds must be greater than 0.")
        if queue_seconds < window_seconds:
            raise ValueError("--queue-seconds must be at least as large as --seconds.")

        self.window_samples = int(window_seconds * SAMPLE_RATE)
        self.block_samples = int(block_seconds * SAMPLE_RATE)
        self.queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, int(queue_seconds / window_seconds))
        )
        self._buffer = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._dropped_windows = 0

    @property
    def dropped_windows(self) -> int:
        with self._lock:
            return self._dropped_windows

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"Microphone status: {status}", file=sys.stderr, flush=True)

        block = np.clip(indata[:, 0], -1.0, 1.0).astype(np.float32, copy=True)
        self._buffer = np.concatenate((self._buffer, block))

        while self._buffer.size >= self.window_samples:
            window = self._buffer[: self.window_samples].copy()
            self._buffer = self._buffer[self.window_samples :]
            self._put_window(window)

    def _put_window(self, window: np.ndarray) -> None:
        try:
            self.queue.put_nowait(window)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            self._dropped_windows += 1

        try:
            self.queue.put_nowait(window)
        except queue.Full:
            with self._lock:
                self._dropped_windows += 1

    def stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.block_samples,
            callback=self._callback,
        )


def run_buffered_stt(
    transcriber: GemmaTranscriber,
    seconds: float,
    block_seconds: float,
    queue_seconds: float,
    prompt: str,
    max_new_tokens: int,
    skip_blank: bool,
    silence_rms: float,
    min_peak: float,
    debug_audio: bool,
    filter_texts: tuple[str, ...],
) -> None:
    mic = MicrophoneRingBuffer(
        window_seconds=seconds,
        block_seconds=block_seconds,
        queue_seconds=queue_seconds,
    )
    print(
        "Buffered continuous STT started. "
        f"Capturing {seconds:g}s windows with up to {queue_seconds:g}s queued. "
        "Press Ctrl+C to stop."
    )

    chunk_index = 1
    last_drop_count = 0
    with mic.stream():
        while True:
            audio = mic.queue.get()
            audio_description = describe_audio(audio)
            if debug_audio:
                print(f"[{chunk_index:04d}] {audio_description}", file=sys.stderr, flush=True)

            if should_skip_audio(audio, silence_rms, min_peak):
                if not skip_blank:
                    print(f"[{chunk_index:04d}] <silence> {audio_description}", flush=True)
                chunk_index += 1
                continue

            started_at = time.monotonic()
            result = transcriber.transcribe_audio(
                audio=audio,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.monotonic() - started_at
            if is_filtered_transcript(result, filter_texts):
                if not skip_blank:
                    print(f"[{chunk_index:04d} | {elapsed:.1f}s] <filtered> {result}", flush=True)
                chunk_index += 1
                continue

            drop_count = mic.dropped_windows
            if drop_count != last_drop_count:
                print(
                    f"Warning: dropped {drop_count - last_drop_count} audio window(s); "
                    "STT is slower than capture.",
                    file=sys.stderr,
                    flush=True,
                )
                last_drop_count = drop_count

            if result or not skip_blank:
                print(f"[{chunk_index:04d} | {elapsed:.1f}s] {result}", flush=True)
            chunk_index += 1


def calibrate_silence(seconds: float) -> None:
    print(f"Stay quiet. Recording {seconds:g}s to measure background noise...")
    audio = record_microphone(seconds)
    rms = audio_rms(audio)
    peak = audio_peak(audio)
    print(f"Measured background: rms={rms:.5f} peak={peak:.5f}")
    print(f"Suggested --silence-rms {max(rms * 3.0, 0.005):.5f}")
    print(f"Suggested --min-peak {max(peak * 1.5, 0.02):.5f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record microphone audio and send it directly to a local Gemma audio-capable model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording length per chunk.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--continuous", action="store_true", help="Keep recording and transcribing chunks.")
    parser.add_argument(
        "--buffered",
        action="store_true",
        help="Continuously capture microphone audio into a bounded queue while STT runs.",
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=DEFAULT_BLOCK_SECONDS,
        help="Microphone callback block size for buffered mode.",
    )
    parser.add_argument(
        "--queue-seconds",
        type=float,
        default=DEFAULT_QUEUE_SECONDS,
        help="Maximum audio backlog to keep in buffered mode before dropping oldest chunks.",
    )
    parser.add_argument(
        "--no-skip-blank",
        action="store_true",
        help="Print chunk markers even when the model returns an empty transcription.",
    )
    parser.add_argument(
        "--silence-rms",
        type=float,
        default=DEFAULT_SILENCE_RMS,
        help="Skip STT for chunks below this RMS amplitude. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-peak",
        type=float,
        default=DEFAULT_MIN_PEAK,
        help="Skip STT for chunks below this peak amplitude. Set 0 to disable.",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Print RMS and peak levels for every audio chunk.",
    )
    parser.add_argument(
        "--calibrate-silence",
        action="store_true",
        help="Measure background noise and print suggested silence gate values, without loading Gemma.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_ASR_PROMPT,
        help="Instruction sent with the microphone audio.",
    )
    parser.add_argument(
        "--filter-text",
        action="append",
        default=list(DEFAULT_FILTER_TEXTS),
        help="Suppress exact generated text. Can be repeated. Defaults to filtering '1.7'.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    args = parse_args()
    if args.calibrate_silence:
        calibrate_silence(args.seconds)
        return

    print(f"Loading {args.model} with Transformers...")
    transcriber = GemmaTranscriber(args.model)

    try:
        if args.buffered:
            run_buffered_stt(
                transcriber=transcriber,
                seconds=args.seconds,
                block_seconds=args.block_seconds,
                queue_seconds=args.queue_seconds,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                skip_blank=not args.no_skip_blank,
                silence_rms=args.silence_rms,
                min_peak=args.min_peak,
                debug_audio=args.debug_audio,
                filter_texts=tuple(args.filter_text),
            )
        elif args.continuous:
            run_continuous_stt(
                transcriber=transcriber,
                seconds=args.seconds,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                skip_blank=not args.no_skip_blank,
                silence_rms=args.silence_rms,
                min_peak=args.min_peak,
                debug_audio=args.debug_audio,
                filter_texts=tuple(args.filter_text),
            )
        else:
            audio = record_microphone(args.seconds)
            if args.debug_audio:
                print(describe_audio(audio), file=sys.stderr, flush=True)

            if should_skip_audio(audio, args.silence_rms, args.min_peak):
                result = ""
            else:
                result = transcriber.transcribe_audio(
                    audio=audio,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                if is_filtered_transcript(result, tuple(args.filter_text)):
                    result = ""
            print("\nGemma output:")
            print(result)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
