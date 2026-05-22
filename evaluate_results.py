from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


# ============================================================
# 1. RouterAI / OpenAI-compatible client
# ============================================================

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


# ============================================================
# 2. JSONL utils
# ============================================================

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. Text normalization and deterministic metrics
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w\sа-яА-Яa-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    true_tokens = tokenize(ground_truth)

    if not pred_tokens and not true_tokens:
        return 1.0

    if not pred_tokens or not true_tokens:
        return 0.0

    pred_counts = defaultdict(int)
    true_counts = defaultdict(int)

    for token in pred_tokens:
        pred_counts[token] += 1

    for token in true_tokens:
        true_counts[token] += 1

    common = 0
    for token in pred_counts:
        common += min(pred_counts[token], true_counts[token])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(true_tokens)

    return 2 * precision * recall / (precision + recall)


def char_similarity(prediction: str, ground_truth: str) -> float:
    """
    Простая посимвольная похожесть через SequenceMatcher.
    Это не CER, но удобная быстрая оценка близости ответа к эталону.
    """

    from difflib import SequenceMatcher

    pred = normalize_text(prediction)
    true = normalize_text(ground_truth)

    if not pred and not true:
        return 1.0

    if not pred or not true:
        return 0.0

    return SequenceMatcher(None, pred, true).ratio()


# ============================================================
# 4. Chroma: загрузка retrieved documents по id
# ============================================================

class ChromaContextLoader:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collections_cache = {}

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections_cache:
            self.collections_cache[collection_name] = self.client.get_collection(collection_name)
        return self.collections_cache[collection_name]

    def get_documents_by_ids(
        self,
        collection_name: str,
        ids: list[str],
    ) -> list[dict[str, Any]]:
        if not ids:
            return []

        collection = self.get_collection(collection_name)

        result = collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        docs_by_id = {}

        for i, doc_id in enumerate(result.get("ids", [])):
            docs_by_id[doc_id] = {
                "id": doc_id,
                "text": result["documents"][i],
                "metadata": result["metadatas"][i],
            }

        # Возвращаем в исходном порядке retrieved ids.
        return [docs_by_id[doc_id] for doc_id in ids if doc_id in docs_by_id]


# ============================================================
# 5. Exact context metrics по id
# ============================================================

def context_recall_exact(
    expected_ids: list[str],
    retrieved_ids: list[str],
) -> float | None:
    """
    Показывает, какая доля эталонных source_chunk_ids попала в top-k.

    Важно:
    эта метрика полностью корректна только когда testset и results
    используют одну и ту же систему id чанков.
    Например, paragraph -> paragraph.

    Для chapter/section против paragraph-эталона exact id может быть 0,
    даже если смысловой контекст найден правильно.
    Поэтому дополнительно считаем LLM context recall.
    """

    expected = set(expected_ids)
    retrieved = set(retrieved_ids)

    if not expected:
        return None

    return len(expected & retrieved) / len(expected)


def context_precision_exact(
    expected_ids: list[str],
    retrieved_ids: list[str],
) -> float | None:
    """
    Показывает, какая доля retrieved chunks входит в эталонные source_chunk_ids.
    """

    expected = set(expected_ids)
    retrieved = set(retrieved_ids)

    if not retrieved:
        return None

    if not expected:
        return None

    return len(expected & retrieved) / len(retrieved)


# ============================================================
# 6. LLM-as-a-Judge metrics
# ============================================================

JUDGE_SYSTEM_PROMPT = """
Ты являешься строгим оценщиком RAG-системы.

Тебе будут переданы:
1. Вопрос пользователя.
2. Эталонный ответ.
3. Ответ тестируемой RAG-системы.
4. Эталонный контекст, на основе которого был создан вопрос.
5. Контекст, найденный retrieval-модулем и переданный модели.

Оцени качество по метрикам от 0 до 1.

Верни только JSON строго такого вида:
{
  "semantic_similarity": 0.0,
  "context_recall_llm": 0.0,
  "context_precision_llm": 0.0,
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "judge_comment": "..."
}

Определения метрик:

semantic_similarity:
Насколько ответ RAG-системы смыслово совпадает с эталонным ответом.
1.0 — полностью совпадает по смыслу.
0.0 — не совпадает.

context_recall_llm:
Насколько найденный retrieval-контекст содержит информацию, необходимую для ответа.
1.0 — вся нужная информация найдена.
0.5 — найдена часть информации.
0.0 — нужная информация не найдена.

context_precision_llm:
Насколько найденный retrieval-контекст не содержит лишнего шума.
1.0 — почти весь найденный контекст полезен.
0.5 — есть и полезные, и лишние фрагменты.
0.0 — контекст в основном бесполезен.

faithfulness:
Насколько ответ RAG-системы опирается на найденный контекст и не содержит выдуманных фактов.
1.0 — ответ полностью подтверждается найденным контекстом.
0.5 — есть частично неподтверждённые утверждения.
0.0 — ответ в основном не подтверждается контекстом.

answer_relevancy:
Насколько ответ отвечает именно на заданный вопрос.
1.0 — ответ полностью релевантен вопросу.
0.5 — ответ частично релевантен.
0.0 — ответ не отвечает на вопрос.

Не будь слишком мягким. Если информации в retrieved context нет, faithfulness и context_recall должны быть низкими.
"""


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
        raise ValueError(f"Не удалось извлечь JSON из ответа судьи:\n{text}")

    return json.loads(text[start : end + 1])


def clamp_score(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value


def build_judge_prompt(
    question: str,
    ground_truth: str,
    answer: str,
    reference_contexts: list[str],
    retrieved_contexts: list[dict[str, Any]],
    max_context_chars: int,
) -> str:
    reference_context = "\n\n--- ЭТАЛОННЫЙ ФРАГМЕНТ ---\n\n".join(reference_contexts)
    retrieved_context = "\n\n--- НАЙДЕННЫЙ ФРАГМЕНТ ---\n\n".join(
        f"id={item.get('id')}\n{item.get('text', '')}"
        for item in retrieved_contexts
    )

    reference_context = reference_context[:max_context_chars]
    retrieved_context = retrieved_context[:max_context_chars]

    return f"""
Вопрос:
{question}

Эталонный ответ:
{ground_truth}

Ответ RAG-системы:
{answer}

Эталонный контекст:
\"\"\"
{reference_context}
\"\"\"

Найденный retrieval-контекст:
\"\"\"
{retrieved_context}
\"\"\"

Оцени пример по заданным метрикам.
"""


def judge_with_llm(
    client: OpenAI,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
    reference_contexts: list[str],
    retrieved_contexts: list[dict[str, Any]],
    max_context_chars: int,
    max_retries: int = 3,
) -> dict[str, Any]:
    user_prompt = build_judge_prompt(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        reference_contexts=reference_contexts,
        retrieved_contexts=retrieved_contexts,
        max_context_chars=max_context_chars,
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            raw = extract_json_object(content)

            return {
                "semantic_similarity": clamp_score(raw.get("semantic_similarity")),
                "context_recall_llm": clamp_score(raw.get("context_recall_llm")),
                "context_precision_llm": clamp_score(raw.get("context_precision_llm")),
                "faithfulness": clamp_score(raw.get("faithfulness")),
                "answer_relevancy": clamp_score(raw.get("answer_relevancy")),
                "judge_comment": str(raw.get("judge_comment", "")),
            }

        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)

    return {
        "semantic_similarity": 0.0,
        "context_recall_llm": 0.0,
        "context_precision_llm": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "judge_comment": f"JUDGE_ERROR: {last_error}",
    }


# ============================================================
# 7. Aggregation
# ============================================================

def mean_ignore_none(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]

    if not cleaned:
        return None

    return statistics.mean(cleaned)


def aggregate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["collection"]].append(row)

    summary = []

    metric_names = [
        "exact_match",
        "token_f1",
        "char_similarity",
        "semantic_similarity",
        "context_recall_exact",
        "context_precision_exact",
        "context_recall_llm",
        "context_precision_llm",
        "faithfulness",
        "answer_relevancy",
    ]

    for collection, collection_rows in grouped.items():
        item = {
            "collection": collection,
            "count": len(collection_rows),
        }

        for metric in metric_names:
            item[metric] = mean_ignore_none([row.get(metric) for row in collection_rows])

        summary.append(item)

    summary.sort(key=lambda x: x["collection"])

    return summary


# ============================================================
# 8. Main evaluation
# ============================================================

def evaluate_file(
    testset_by_id: dict[str, dict[str, Any]],
    results_path: str,
    db_loader: ChromaContextLoader,
    judge_client: OpenAI | None,
    judge_model: str,
    use_llm_judge: bool,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    results = load_jsonl(results_path)
    evaluated_rows = []

    for result_row in tqdm(results, desc=f"Evaluating {Path(results_path).name}"):
        sample_id = result_row.get("id")

        if sample_id not in testset_by_id:
            continue

        test_row = testset_by_id[sample_id]

        collection = result_row.get("collection") or infer_collection_from_filename(results_path)

        question = result_row.get("question", test_row.get("question", ""))
        ground_truth = test_row.get("ground_truth", "")
        answer = result_row.get("answer", "")

        expected_ids = test_row.get("source_chunk_ids", []) or []
        retrieved_ids = result_row.get("retrieved_chunk_ids", []) or []

        reference_contexts = test_row.get("contexts", []) or []

        retrieved_contexts = db_loader.get_documents_by_ids(
            collection_name=collection,
            ids=retrieved_ids,
        )

        row = {
            "id": sample_id,
            "question_type": test_row.get("question_type"),
            "collection": collection,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "expected_chunk_ids": "|".join(expected_ids),
            "retrieved_chunk_ids": "|".join(retrieved_ids),
            "exact_match": exact_match(answer, ground_truth),
            "token_f1": token_f1(answer, ground_truth),
            "char_similarity": char_similarity(answer, ground_truth),
            "context_recall_exact": context_recall_exact(expected_ids, retrieved_ids),
            "context_precision_exact": context_precision_exact(expected_ids, retrieved_ids),
        }

        if use_llm_judge:
            if judge_client is None:
                raise RuntimeError("LLM judge включён, но judge_client отсутствует")

            judge_scores = judge_with_llm(
                client=judge_client,
                model=judge_model,
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                reference_contexts=reference_contexts,
                retrieved_contexts=retrieved_contexts,
                max_context_chars=max_context_chars,
            )

            row.update(judge_scores)
        else:
            row.update(
                {
                    "semantic_similarity": None,
                    "context_recall_llm": None,
                    "context_precision_llm": None,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "judge_comment": "",
                }
            )

        evaluated_rows.append(row)

    return evaluated_rows


def infer_collection_from_filename(path: str) -> str:
    name = Path(path).stem

    mapping = {
        "results_chapter": "chunks_chapter",
        "results_section": "chunks_section",
        "results_paragraph_group": "chunks_paragraph_group",
        "results_paragraph": "chunks_paragraph",
    }

    return mapping.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testset",
        default="outputs/testset.jsonl",
        help="Эталонный датасет JSONL",
    )

    parser.add_argument(
        "--results",
        nargs="+",
        default=[
            "outputs/results_chapter.jsonl",
            "outputs/results_section.jsonl",
            "outputs/results_paragraph_group.jsonl",
            "outputs/results_paragraph.jsonl",
        ],
        help="Файлы результатов RAG",
    )

    parser.add_argument(
        "--db-path",
        default="chroma_db",
        help="Путь к ChromaDB",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Папка для результатов оценки",
    )

    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4.6"),
        help="Модель-судья через RouterAI",
    )

    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Отключить LLM-as-a-Judge и считать только deterministic metrics",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=16000,
        help="Максимальная длина эталонного и найденного контекста для судьи",
    )

    args = parser.parse_args()

    testset = load_jsonl(args.testset)
    testset_by_id = {row["id"]: row for row in testset}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_loader = ChromaContextLoader(args.db_path)

    use_llm_judge = not args.no_llm_judge

    judge_client = get_routerai_client() if use_llm_judge else None

    all_evaluated_rows = []

    for results_path in args.results:
        if not Path(results_path).exists():
            print(f"Файл не найден, пропускаю: {results_path}")
            continue

        evaluated_rows = evaluate_file(
            testset_by_id=testset_by_id,
            results_path=results_path,
            db_loader=db_loader,
            judge_client=judge_client,
            judge_model=args.judge_model,
            use_llm_judge=use_llm_judge,
            max_context_chars=args.max_context_chars,
        )

        all_evaluated_rows.extend(evaluated_rows)

    summary_rows = aggregate_scores(all_evaluated_rows)

    detailed_jsonl = output_dir / "evaluation_detailed.jsonl"
    detailed_csv = output_dir / "evaluation_detailed.csv"
    summary_csv = output_dir / "evaluation_summary.csv"
    summary_json = output_dir / "evaluation_summary.json"

    save_jsonl(detailed_jsonl, all_evaluated_rows)
    save_csv(detailed_csv, all_evaluated_rows)
    save_csv(summary_csv, summary_rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\nГОТОВО")
    print(f"Детальные результаты JSONL: {detailed_jsonl}")
    print(f"Детальные результаты CSV:   {detailed_csv}")
    print(f"Сводная таблица CSV:        {summary_csv}")
    print(f"Сводная таблица JSON:       {summary_json}")

    print("\nСводка:")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()