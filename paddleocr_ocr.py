from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

from common import ensure_dir, print_header, preview, save_json, save_text


def _set_paddle_device(use_gpu: bool) -> str:
    if use_gpu and paddle.device.is_compiled_with_cuda():
        paddle.set_device("gpu")
        return "cuda"
    paddle.set_device("cpu")
    return "cpu"


def _extract_lines_from_paddle_result(result) -> list[str]:
    lines: list[str] = []
    if not result:
        return lines

    blocks = result[0] if isinstance(result, list) and len(result) == 1 else result
    if not isinstance(blocks, list):
        return lines

    for item in blocks:
        try:
            text = item[1][0]
            if text:
                lines.append(text)
        except Exception:
            continue
    return lines


def ocr_paddleocr(
    pdf_path: str,
    dpi: int = 300,
    lang: str = "ru",
    use_gpu: bool = False,
    poppler_path: str | None = None,
) -> dict:
    device = _set_paddle_device(use_gpu)

    ocr = PaddleOCR(
        lang=lang,
        use_angle_cls=True,
    )

    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

    pages = []
    full_text_parts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for page_num, image in enumerate(images, start=1):
            image_path = tmpdir_path / f"page_{page_num}.png"
            image.save(image_path)

            result = ocr.ocr(str(image_path), cls=True)
            lines = _extract_lines_from_paddle_result(result)
            page_text = "\n".join(lines)

            pages.append(
                {
                    "page": page_num,
                    "lines_count": len(lines),
                    "text": page_text,
                    "lines": lines,
                }
            )
            if page_text.strip():
                full_text_parts.append(page_text.strip())

    full_text = "\n\n".join(full_text_parts)
    return {
        "method": f"PaddleOCR ({device})",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR PDF с PaddleOCR")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение рендеринга PDF")
    parser.add_argument("--lang", default="ru", help="Язык модели")
    parser.add_argument("--gpu", action="store_true", help="Использовать GPU")
    parser.add_argument(
        "--poppler-path",
        default=r"C:\poppler\poppler-24.08.0\Library\bin",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = ocr_paddleocr(
        args.pdf_path,
        dpi=args.dpi,
        lang=args.lang,
        use_gpu=args.gpu,
        poppler_path=args.poppler_path,
    )
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_paddleocr.txt"
    json_path = out_dir / f"{stem}_paddleocr.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("PaddleOCR")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()