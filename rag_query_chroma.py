from __future__ import annotations

import argparse
import json
import os
from typing import Any

import chromadb
from chromadb.config import Settings

from common1 import call_chat_text, get_routerai_client, load_jsonl


SYSTEM_ANSWER = """
Ты отвечаешь на вопрос пользователя строго по найденному контексту.

Правила:
1. Используй только предоставленный контекст.
2. Если в контексте нет ответа, скажи: "В найденном контексте нет достаточной информации".
3. Не выдумывай факты.
4. Отвечай на русском языке.
5. Ответ должен быть точным и кратким.
"""


def embed_query(question: str, model: str) -> list[float]:
    client = get_routerai_client()

    response = client.embeddings.create(
        model=model,
        input=question,
    )

    return response.data[0].embedding


def retrieve(
    question: str,
    collection_name: str,
    db_path: str,
    top_k: int,
) -> list[dict[str, Any]]:
    embedding_model = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = chroma_client.get_collection(collection_name)

    query_embedding = embed_query(question, embedding_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "rank": i + 1,
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    return retrieved


def build_answer_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[CHUNK {item['rank']} | id={item['id']} | distance={item['distance']}]\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
Вопрос:
{question}

Найденный контекст:
\"\"\"
{context}
\"\"\"

Сформируй ответ.
"""


def answer_question(
    question: str,
    collection_name: str,
    db_path: str,
    top_k: int,
) -> dict[str, Any]:
    answer_model = os.getenv("ANSWER_MODEL", "deepseek/deepseek-v4-pro")
    client = get_routerai_client()

    retrieved = retrieve(
        question=question,
        collection_name=collection_name,
        db_path=db_path,
        top_k=top_k,
    )

    answer = call_chat_text(
        client=client,
        model=answer_model,
        system_prompt=SYSTEM_ANSWER,
        user_prompt=build_answer_prompt(question, retrieved),
        temperature=0.0,
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "collection": collection_name,
        "top_k": top_k,
        "answer_model": answer_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question")
    parser.add_argument("--testset")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.question:
        result = answer_question(
            question=args.question,
            collection_name=args.collection,
            db_path=args.db_path,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.testset:
        rows = load_jsonl(args.testset)
        output_rows = []

        for row in rows:
            result = answer_question(
                question=row["question"],
                collection_name=args.collection,
                db_path=args.db_path,
                top_k=args.top_k,
            )

            output_rows.append(
                {
                    "id": row["id"],
                    "question_type": row.get("question_type"),
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "answer": result["answer"],
                    "retrieved_chunk_ids": [r["id"] for r in result["retrieved"]],
                    "retrieved_distances": [r["distance"] for r in result["retrieved"]],
                    "collection": args.collection,
                    "top_k": args.top_k,
                }
            )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for row in output_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"Сохранено: {args.output}")
        else:
            print(json.dumps(output_rows, ensure_ascii=False, indent=2))

        return

    raise RuntimeError("Укажи либо --question, либо --testset")


if __name__ == "__main__":
    main()