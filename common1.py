from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_routerai_client() -> OpenAI:
    api_key = os.getenv("ROUTERAI_API_KEY")
    base_url = os.getenv("ROUTERAI_BASE_URL")

    if not api_key:
        raise RuntimeError("Не задан ROUTERAI_API_KEY в .env")

    if not base_url:
        raise RuntimeError("Не задан ROUTERAI_BASE_URL в .env")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def normalize_chunks(raw: Any, source_name: str = "") -> list[dict[str, Any]]:
    """
    Приводит разные возможные форматы чанков к единому виду.

    На выходе:
    {
        "id": "...",
        "text": "...",
        "metadata": {...}
    }
    """

    if isinstance(raw, dict):
        if "chunks" in raw and isinstance(raw["chunks"], list):
            raw_chunks = raw["chunks"]
        elif "data" in raw and isinstance(raw["data"], list):
            raw_chunks = raw["data"]
        else:
            raw_chunks = [raw]
    elif isinstance(raw, list):
        raw_chunks = raw
    else:
        raise ValueError("Неподдерживаемый формат JSON с чанками")

    chunks: list[dict[str, Any]] = []

    for i, item in enumerate(raw_chunks):
        if isinstance(item, str):
            text = item
            metadata = {}
            chunk_id = f"{source_name}_chunk_{i:05d}"
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("page_content")
                or item.get("chunk")
                or ""
            )

            metadata = item.get("metadata", {}).copy()

            for key in [
                "chapter",
                "section",
                "title",
                "level",
                "page",
                "page_start",
                "page_end",
                "source",
                "parent_id",
            ]:
                if key in item and key not in metadata:
                    metadata[key] = item[key]

            chunk_id = (
                str(item.get("id"))
                if item.get("id") is not None
                else f"{source_name}_chunk_{i:05d}"
            )
        else:
            continue

        text = clean_text(text)

        if not text:
            continue

        metadata["source_file"] = source_name
        metadata["chunk_index"] = i

        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
            }
        )

    return chunks


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def call_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Запрашивает у модели JSON.
    Если response_format не поддерживается конкретным роутером/моделью,
    можно убрать response_format ниже.
    """

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            return extract_json_object(content)

        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Ошибка вызова chat model после {max_retries} попыток: {last_error}")


def call_chat_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Ошибка вызова chat model после {max_retries} попыток: {last_error}")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"^```", "", text.strip())
        text = re.sub(r"```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Не удалось извлечь JSON из ответа модели:\n{text}")

    return json.loads(text[start : end + 1])


def batched(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]