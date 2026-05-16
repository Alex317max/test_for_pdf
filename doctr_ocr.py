from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdf2image import convert_from_path

from common import ensure_dir, print_header, preview, save_json, save_text


def _page_to_bgr_numpy(image) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr


def ocr_doctr(
    pdf_path: str,
    dpi: int = 300,
    use_gpu: bool = True,
    poppler_path: str | None = None,
) -> dict:
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ocr_predictor(pretrained=True).to(device)

    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

    pages = []
    all_pages_text = []

    for page_num, image in enumerate(images, start=1):
        page_img = _page_to_bgr_numpy(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            cv2.imwrite(str(tmp_path), page_img)
            doc = DocumentFile.from_images(str(tmp_path))
            result = model(doc)

            page_lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words = [word.value for word in line.words]
                        if words:
                            page_lines.append(" ".join(words))

            page_text = "\n".join(page_lines)
            pages.append(
                {
                    "page": page_num,
                    "lines_count": len(page_lines),
                    "text": page_text,
                    "lines": page_lines,
                }
            )
            if page_text.strip():
                all_pages_text.append(page_text.strip())
        finally:
            tmp_path.unlink(missing_ok=True)

    full_text = "\n\n".join(all_pages_text)
    return {
        "method": f"DocTR ({device})",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR PDF с DocTR")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение рендеринга PDF")
    parser.add_argument("--gpu", action="store_true", help="Использовать GPU")
    parser.add_argument(
        "--poppler-path",
        default=r"C:\poppler\poppler-24.08.0\Library\bin",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()
    result = ocr_doctr(
        args.pdf_path,
        dpi=args.dpi,
        use_gpu=args.gpu,
        poppler_path=args.poppler_path,
    )
    elapsed = time.perf_counter() - start

    stem = Path(args.pdf_path).stem
    txt_path = out_dir / f"{stem}_doctr.txt"
    json_path = out_dir / f"{stem}_doctr.json"

    save_text(txt_path, result["full_text"])
    save_json(json_path, result)

    print_header("DocTR")
    print(f"Файл: {args.pdf_path}")
    print(f"Страниц: {result['pages_count']}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    print("\nПредпросмотр текста:")
    print(preview(result["full_text"]))


if __name__ == "__main__":
    main()