from __future__ import annotations

import argparse
import time
from pathlib import Path

import pdfplumber

from common import ensure_dir, print_header, preview, save_json, save_text


def extract_text_and_tables_pdfplumber(pdf_path: str) -> dict:
    pages = []
    full_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(layout=True) or ""
            tables = page.extract_tables() or []

            pages.append(
                {
                    "page": page_num,
                    "text": page_text,
                    "tables_count": len(tables),
                    "tables": tables,
                }
            )

            if page_text.strip():
                full_text_parts.append(page_text.strip())

    full_text = "\n\n".join(full_text_parts)
    return {
        "method": "pdfplumber",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение из PDF с pdfplumber")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = extract_text_and_tables_pdfplumber(args.pdf_path)
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_pdfplumber.txt"
    json_path = out_dir / f"{stem}_pdfplumber.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("pdfplumber")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()