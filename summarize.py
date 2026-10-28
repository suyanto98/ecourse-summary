#!/usr/bin/env python3
"""video (English e-course) -> ringkasan materi PDF (Bahasa Indonesia).

Pipeline: ffmpeg+faster-whisper (transkrip) -> Ollama local LLM (ringkasan
terstruktur) -> ffmpeg (screenshot chart per topik) -> Jinja2 + WeasyPrint (PDF).

Usage:
    .venv/bin/python summarize.py --video "/path/video.mp4" --output ./output
"""

import argparse
import base64
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

import frames
import llm_summarize
import transcribe

LOG = logging.getLogger("summarize")

BASE_DIR = Path(__file__).resolve().parent


def get_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip()) if result.returncode == 0 else 0.0


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def attach_frames(video_path: Path, topics: list[dict]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(topics):
            t["timestamp"] = format_timestamp(t["start_seconds"])
            frame_path = Path(tmp) / f"frame_{i}.jpg"
            ok = frames.extract_frame(video_path, t["start_seconds"], frame_path)
            t["image_b64"] = base64.b64encode(frame_path.read_bytes()).decode() if ok else None


def render_pdf(video_path: Path, result: dict, duration: float, out_pdf: Path) -> None:
    env = Environment(loader=FileSystemLoader(BASE_DIR / "templates"))
    template = env.get_template("report.html.jinja")
    html = template.render(
        title=result.get("title") or video_path.stem,
        subtitle=video_path.stem,
        duration=format_timestamp(duration),
        source_name=video_path.name,
        summary=result.get("summary", ""),
        topics=result.get("topics", []),
        glossary=result.get("glossary", []),
        takeaways=result.get("takeaways", []),
        generated_at=datetime.now().strftime("%d %b %Y %H:%M"),
    )
    HTML(string=html, base_url=str(BASE_DIR)).write_pdf(str(out_pdf))
    LOG.info("Wrote %s", out_pdf)


def process_video(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Step 1/4: transcribing...")
    segments = transcribe.transcribe(video_path)

    LOG.info("Step 2/4: summarizing with local LLM...")
    result = llm_summarize.summarize(segments, video_title=video_path.stem)

    LOG.info("Step 3/4: extracting %d screenshot(s)...", len(result.get("topics", [])))
    attach_frames(video_path, result.get("topics", []))

    LOG.info("Step 4/4: rendering PDF...")
    duration = get_duration_seconds(video_path)
    out_pdf = output_dir / f"{video_path.stem}_ringkasan.pdf"
    render_pdf(video_path, result, duration, out_pdf)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path ke file video sumber")
    parser.add_argument("--output", default="./output", help="Folder output PDF")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video tidak ditemukan: {video_path}")

    out_pdf = process_video(video_path, Path(args.output).expanduser())
    print(f"\nSelesai: {out_pdf}")


if __name__ == "__main__":
    main()
