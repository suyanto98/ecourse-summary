"""transcript segments -> structured Indonesian summary, via a local Ollama model.

Map-reduce so it works regardless of the model's context window: transcript is
split into chunks, each chunk is turned into a list of topics (heading + body +
a timestamp), then all topics are reduced into one title/summary/glossary/
takeaways pass.
"""

import json
import logging
import urllib.request

LOG = logging.getLogger("llm_summarize")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "aya-expanse:8b"
CHUNK_CHAR_BUDGET = 6000


def _call_ollama(messages: list[dict], model: str = MODEL) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _chunk_segments(segments: list[dict], max_chars: int = CHUNK_CHAR_BUDGET) -> list[list[dict]]:
    chunks, current, current_len = [], [], 0
    for seg in segments:
        line_len = len(seg["text"]) + 12
        if current and current_len + line_len > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(seg)
        current_len += line_len
    if current:
        chunks.append(current)
    return chunks


def _nearest_start(all_starts: list[float], guess: float) -> float:
    return min(all_starts, key=lambda s: abs(s - guess))


def _extract_topics(chunk: list[dict], all_starts: list[float]) -> list[dict]:
    transcript_block = "\n".join(f"[{seg['start']:.1f}] {seg['text']}" for seg in chunk)
    system = (
        "Kamu adalah asisten yang meringkas transkrip video edukasi trading berbahasa "
        "Inggris menjadi catatan materi berbahasa Indonesia. Balas HANYA dengan JSON valid, "
        "tanpa teks lain."
    )
    user = (
        "Berikut potongan transkrip video (format: [detik] teks). Identifikasi 1-4 topik "
        "materi berbeda yang dibahas di potongan ini. Untuk tiap topik, tulis heading singkat "
        "(Bahasa Indonesia), body penjelasan 2-4 kalimat (Bahasa Indonesia, terjemahkan & "
        "ringkas maknanya, jangan terjemahan kata-per-kata), dan start_seconds = angka detik "
        "PALING AWAL (dari yang tertera di [ ]) tempat topik itu mulai dibahas.\n\n"
        f"Transkrip:\n{transcript_block}\n\n"
        'Balas dengan JSON: {"topics": [{"start_seconds": <number>, "heading": "...", '
        '"body": "..."}]}'
    )
    raw = _call_ollama([{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        parsed = json.loads(raw)
        topics = parsed.get("topics", [])
    except (json.JSONDecodeError, AttributeError):
        LOG.warning("Failed to parse topic JSON from LLM, skipping chunk: %r", raw[:300])
        return []
    for t in topics:
        try:
            t["start_seconds"] = _nearest_start(all_starts, float(t.get("start_seconds", chunk[0]["start"])))
        except (TypeError, ValueError):
            t["start_seconds"] = chunk[0]["start"]
    return topics


def _synthesize(topics: list[dict], video_title: str) -> dict:
    topics_block = "\n".join(f"- {t['heading']}: {t['body']}" for t in topics)
    system = (
        "Kamu adalah asisten yang membuat ringkasan akhir catatan materi video edukasi "
        "trading dalam Bahasa Indonesia. Balas HANYA dengan JSON valid, tanpa teks lain."
    )
    user = (
        f"Judul video: {video_title}\n\n"
        f"Daftar topik yang sudah diekstrak dari video:\n{topics_block}\n\n"
        "Buatkan:\n"
        "1. title: judul ringkas untuk laporan (Bahasa Indonesia)\n"
        "2. summary: ringkasan eksekutif 2-3 kalimat inti materi\n"
        "3. glossary: 4-8 istilah penting yang disebutkan beserta definisi singkatnya "
        "(list of {term, definition})\n"
        "4. takeaways: 3-5 poin actionable takeaway (list of string)\n\n"
        'Balas dengan JSON: {"title": "...", "summary": "...", '
        '"glossary": [{"term": "...", "definition": "..."}], "takeaways": ["...", "..."]}'
    )
    raw = _call_ollama([{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        LOG.warning("Failed to parse synthesis JSON from LLM: %r", raw[:300])
        return {"title": video_title, "summary": "", "glossary": [], "takeaways": []}


def summarize(segments: list[dict], video_title: str) -> dict:
    if not segments:
        return {"title": video_title, "summary": "", "topics": [], "glossary": [], "takeaways": []}

    all_starts = [seg["start"] for seg in segments]
    chunks = _chunk_segments(segments)
    LOG.info("Summarizing %d segments in %d chunk(s) with model %s", len(segments), len(chunks), MODEL)

    topics: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        LOG.info("Extracting topics from chunk %d/%d...", i, len(chunks))
        topics.extend(_extract_topics(chunk, all_starts))

    LOG.info("Synthesizing final summary from %d topic(s)...", len(topics))
    final = _synthesize(topics, video_title)
    final["topics"] = sorted(topics, key=lambda t: t["start_seconds"])
    return final
