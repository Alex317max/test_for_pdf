from __future__ import annotations

import argparse
import time
from pathlib import Path

from PyPDF2 import PdfReader

from common import ensure_dir, print_header, preview, save_json, save_text


def extract_text_pypdf2(pdf_path: str) -> dict:
    reader = PdfReader(pdf_path)
    pages = []
    full_text_parts = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(
            {
                "page": page_num,
                "characters": len(page_text),
                "text": page_text,
            }
        )
        if page_text.strip():
            full_text_parts.append(page_text.strip())

    full_text = "\n\n".join(full_text_parts)
    return {
        "method": "PyPDF2",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение текста из PDF с PyPDF2")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = extract_text_pypdf2(args.pdf_path)
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_pypdf2.txt"
    json_path = out_dir / f"{stem}_pypdf2.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("PyPDF2")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()