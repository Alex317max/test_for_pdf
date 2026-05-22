from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from common1 import batched, get_routerai_client, load_json, normalize_chunks


def embed_texts(
    texts: list[str],
    model: str,
    batch_size: int = 32,
    max_retries: int = 3,
) -> list[list[float]]:
    client = get_routerai_client()
    all_embeddings: list[list[float]] = []

    for batch in tqdm(list(batched(texts, batch_size)), desc="Embedding batches"):
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)

        else:
            raise RuntimeError(f"Не удалось получить embeddings: {last_error}")

    return all_embeddings


def prepare_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Chroma metadata должна содержать простые типы.
    Списки/словари переводим в строки.
    """

    prepared = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            prepared[key] = value
        else:
            prepared[key] = str(value)

    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="JSON-файл с чанками")
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument(
        "--collection",
        required=True,
        help="Название коллекции, например chunks_section или chunks_paragraph",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Удалить коллекцию перед загрузкой",
    )
    args = parser.parse_args()

    embedding_model = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

    raw = load_json(args.chunks)
    source_name = Path(args.chunks).stem
    chunks = normalize_chunks(raw, source_name=source_name)

    if not chunks:
        raise RuntimeError("Не найдено ни одного чанка с текстом")

    print(f"Файл чанков: {args.chunks}")
    print(f"Количество чанков: {len(chunks)}")
    print(f"Embedding model: {embedding_model}")
    print(f"Chroma collection: {args.collection}")

    client = chromadb.PersistentClient(
        path=args.db_path,
        settings=Settings(anonymized_telemetry=False),
    )

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Коллекция {args.collection} удалена")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "source_chunks": args.chunks,
        },
    )

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [prepare_metadata(c["metadata"]) for c in chunks]

    embeddings = embed_texts(
        texts=texts,
        model=embedding_model,
        batch_size=args.batch_size,
    )

    for batch_start in tqdm(range(0, len(chunks), args.batch_size), desc="Upserting Chroma"):
        batch_end = batch_start + args.batch_size

        collection.upsert(
            ids=ids[batch_start:batch_end],
            documents=texts[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
        )

    print("Готово.")
    print(f"Всего записей в коллекции: {collection.count()}")


if __name__ == "__main__":
    main()