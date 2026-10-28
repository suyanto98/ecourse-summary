"""Extract a still frame from a video at a given timestamp (ffmpeg)."""

import logging
import subprocess
from pathlib import Path

LOG = logging.getLogger("frames")


def extract_frame(video_path: Path, at_seconds: float, out_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(at_seconds, 0):.2f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        LOG.warning("Failed to extract frame at %.1fs: %s", at_seconds, result.stderr[-500:])
        return False
    return True
