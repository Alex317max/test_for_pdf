from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common1 import (
    call_chat_json,
    get_routerai_client,
    load_json,
    normalize_chunks,
    save_jsonl,
)


SYSTEM_GENERATOR = """
Ты создаёшь эталонный датасет для оценки RAG-системы по технической документации.

Нужно сгенерировать один объект JSON строго такого вида:
{
  "question": "...",
  "ground_truth": "...",
  "context": "...",
  "question_type": "simple|multi_context|reasoning"
}

Требования:
1. Вопрос должен быть на русском языке.
2. Ответ должен быть полностью основан на переданном контексте.
3. Нельзя использовать знания вне контекста.
4. ground_truth должен быть эталонным ответом: точным, проверяемым, без лишней воды.
5. Для simple вопрос должен отвечаться по одному фрагменту.
6. Для multi_context вопрос должен требовать объединения нескольких фрагментов.
7. Для reasoning вопрос должен требовать простого логического вывода из контекста.
8. Не добавляй markdown, комментарии и пояснения вне JSON.
"""


SYSTEM_CRITIC = """
Ты проверяешь качество синтетического примера для RAG-оценки.

Верни JSON строго такого вида:
{
  "is_valid": true,
  "fixed_question": "...",
  "fixed_ground_truth": "...",
  "reason": "..."
}

Критерии:
1. Вопрос должен быть понятным и отвечаемым по контексту.
2. Эталонный ответ должен быть полностью подтверждён контекстом.
3. В ответе не должно быть внешних фактов.
4. Если пример хороший, верни is_valid=true и продублируй вопрос/ответ.
5. Если пример можно исправить, исправь вопрос и/или ответ.
6. Если пример плохой и исправить нельзя, верни is_valid=false.
"""


def build_prompt(context: str, question_type: str) -> str:
    return f"""
Тип вопроса: {question_type}

Контекст:
\"\"\"
{context}
\"\"\"

Сгенерируй один эталонный пример для RAG-датасета.
"""


def build_critic_prompt(
    context: str,
    question: str,
    ground_truth: str,
    question_type: str,
) -> str:
    return f"""
Тип вопроса: {question_type}

Контекст:
\"\"\"
{context}
\"\"\"

Вопрос:
{question}

Эталонный ответ:
{ground_truth}

Проверь и при необходимости исправь пример.
"""


def sample_context(
    chunks: list[dict[str, Any]],
    question_type: str,
    max_context_chars: int,
) -> tuple[str, list[str]]:
    if question_type == "simple":
        chunk = random.choice(chunks)
        return chunk["text"][:max_context_chars], [chunk["id"]]

    if question_type in {"multi_context", "reasoning"}:
        start_idx = random.randint(0, max(0, len(chunks) - 2))
        selected = chunks[start_idx : start_idx + 2]

        # Если контекст короткий, можно добавить третий соседний чанк.
        if len(chunks) > start_idx + 2 and random.random() < 0.5:
            selected.append(chunks[start_idx + 2])

        joined = "\n\n--- ФРАГМЕНТ ---\n\n".join(c["text"] for c in selected)
        return joined[:max_context_chars], [c["id"] for c in selected]

    raise ValueError(f"Неизвестный question_type: {question_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="JSON-файл с чанками")
    parser.add_argument("--output", default="outputs/testset.jsonl")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--distribution",
        default="simple:0.5,multi_context:0.4,reasoning:0.1",
        help="Распределение типов вопросов",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    generator_model = os.getenv("GENERATOR_MODEL", "deepseek/deepseek-v4-pro")
    critic_model = os.getenv("CRITIC_MODEL", "anthropic/claude-sonnet-4.6")

    source_name = Path(args.chunks).stem
    raw = load_json(args.chunks)
    chunks = normalize_chunks(raw, source_name=source_name)

    if not chunks:
        raise RuntimeError("Не найдено ни одного чанка с текстом")

    distribution = parse_distribution(args.distribution)
    client = get_routerai_client()

    rows: list[dict[str, Any]] = []

    for i in tqdm(range(args.test_size), desc="Generating testset"):
        question_type = random.choices(
            population=list(distribution.keys()),
            weights=list(distribution.values()),
            k=1,
        )[0]

        context, chunk_ids = sample_context(
            chunks=chunks,
            question_type=question_type,
            max_context_chars=args.max_context_chars,
        )

        generated = call_chat_json(
            client=client,
            model=generator_model,
            system_prompt=SYSTEM_GENERATOR,
            user_prompt=build_prompt(context, question_type),
            temperature=0.3,
        )

        question = str(generated.get("question", "")).strip()
        ground_truth = str(generated.get("ground_truth", "")).strip()

        if not question or not ground_truth:
            continue

        critic = call_chat_json(
            client=client,
            model=critic_model,
            system_prompt=SYSTEM_CRITIC,
            user_prompt=build_critic_prompt(
                context=context,
                question=question,
                ground_truth=ground_truth,
                question_type=question_type,
            ),
            temperature=0.0,
        )

        if not critic.get("is_valid", False):
            continue

        fixed_question = str(critic.get("fixed_question", question)).strip()
        fixed_ground_truth = str(critic.get("fixed_ground_truth", ground_truth)).strip()

        row = {
            "id": f"q_{len(rows):05d}",
            "question_type": question_type,
            "question": fixed_question,
            "contexts": [context],
            "ground_truth": fixed_ground_truth,
            "source_chunk_ids": chunk_ids,
            "source_chunks_file": args.chunks,
            "generator_model": generator_model,
            "critic_model": critic_model,
        }

        rows.append(row)

    save_jsonl(args.output, rows)
    print(f"Готово. Сохранено примеров: {len(rows)}")
    print(f"Файл: {args.output}")


def parse_distribution(raw: str) -> dict[str, float]:
    result = {}

    for part in raw.split(","):
        name, value = part.split(":")
        result[name.strip()] = float(value)

    total = sum(result.values())

    if total <= 0:
        raise ValueError("Сумма весов distribution должна быть больше 0")

    return {k: v / total for k, v in result.items()}


if __name__ == "__main__":
    main()