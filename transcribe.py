"""video -> audio -> English transcript segments (faster-whisper)."""

import logging
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

LOG = logging.getLogger("transcribe")

WHISPER_MODEL_SIZE = "medium"

_whisper_model = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        LOG.info("Loading faster-whisper model '%s' (cpu, int8)...", WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def extract_audio(video_path: Path) -> Path:
    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
    ]
    LOG.info("Extracting audio from %s", video_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr[-2000:]}")
    return wav_path


def transcribe(video_path: Path) -> list[dict]:
    wav_path = extract_audio(video_path)
    try:
        model = get_whisper_model()
        LOG.info("Transcribing %s", video_path.name)
        segments, info = model.transcribe(str(wav_path), language="en", vad_filter=True)
        result = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            result.append({"start": seg.start, "end": seg.end, "text": text})
        LOG.info("Transcribed %d segments (detected language=%s)", len(result), info.language)
        return result
    finally:
        wav_path.unlink(missing_ok=True)
