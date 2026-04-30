from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from resume_screener.models import JobDescription, LabeledPair, Resume


def load_job(path: str | Path) -> JobDescription:
    with Path(path).open("r", encoding="utf-8") as file:
        return JobDescription.from_dict(json.load(file))


def load_resumes(path: str | Path) -> list[Resume]:
    records = _load_json_or_jsonl(path)
    return [Resume.from_dict(record) for record in records]


def load_pairs(path: str | Path) -> list[LabeledPair]:
    records = _load_json_or_jsonl(path)
    return [LabeledPair.from_dict(record) for record in records]


def write_rankings(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json_or_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    return [payload]

