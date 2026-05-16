from __future__ import annotations

import argparse
import time
from pathlib import Path

import fitz  # PyMuPDF

from common import ensure_dir, print_header, preview, save_json, save_text


def extract_text_pymupdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))

            text_blocks = []
            page_text_parts = []

            for block in blocks:
                # block: (x0, y0, x1, y1, text, block_no, block_type, ...)
                if len(block) >= 7 and block[6] == 0:
                    text = (block[4] or "").strip()
                    if text:
                        text_blocks.append(
                            {
                                "bbox": [block[0], block[1], block[2], block[3]],
                                "text": text,
                            }
                        )
                        page_text_parts.append(text)

            page_text = "\n".join(page_text_parts)
            pages.append(
                {
                    "page": page_num + 1,
                    "blocks_count": len(text_blocks),
                    "text": page_text,
                    "blocks": text_blocks,
                }
            )
            if page_text.strip():
                full_text_parts.append(page_text.strip())
    finally:
        doc.close()

    full_text = "\n\n".join(full_text_parts)
    return {
        "method": "PyMuPDF",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение текста из PDF с PyMuPDF")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = extract_text_pymupdf(args.pdf_path)
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_pymupdf.txt"
    json_path = out_dir / f"{stem}_pymupdf.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("PyMuPDF")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()