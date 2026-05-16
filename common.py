from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def save_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def preview(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n[обрезано]"


def print_header(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")