from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

from common import ensure_dir, print_header, preview, save_json, save_text


def ocr_tesseract(
    pdf_path: str,
    dpi: int = 300,
    lang: str = "rus+eng",
    config: str = "--oem 3 --psm 6",
    tesseract_cmd: str | None = None,
    poppler_path: str | None = None,
) -> dict:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"C:\poppler\poppler-24.08.0\Library\bin")
    pages = []
    full_text_parts = []

    for page_num, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        pages.append(
            {
                "page": page_num,
                "text": text,
                "characters": len(text),
            }
        )
        if text.strip():
            full_text_parts.append(text.strip())

    full_text = "\n\n".join(full_text_parts)
    return {
        "method": "Tesseract OCR",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR PDF с Tesseract")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение рендеринга PDF")
    parser.add_argument("--lang", default="rus+eng", help="Язык")
    parser.add_argument("--config", default="--oem 3 --psm 6", help="Конфигурация Tesseract")
    parser.add_argument("--tesseract-cmd", default=r"C:\tesseract\tesseract.exe", help="Полный путь к tesseract.exe, если не в PATH")
    parser.add_argument("--poppler-path", default=r"C:\poppler\poppler-24.08.0\Library\bin", help="Путь к папке bin Poppler в Windows")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = ocr_tesseract(
        pdf_path=args.pdf_path,
        dpi=args.dpi,
        lang=args.lang,
        config=args.config,
        tesseract_cmd=args.tesseract_cmd,
        poppler_path=args.poppler_path,
    )
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_tesseract.txt"
    json_path = out_dir / f"{stem}_tesseract.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("Tesseract OCR")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()