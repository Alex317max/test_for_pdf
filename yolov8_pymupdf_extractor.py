from __future__ import annotations

import argparse
import time
from pathlib import Path

import fitz
import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO

from common import ensure_dir, print_header, preview, save_json, save_text


def extract_layout_yolo_pymupdf(
        pdf_path: str,
        dpi: int = 200,
        poppler_path: str | None = None,
) -> dict:

    model = YOLO("yolov8n-seg.pt")

    label_map = model.names

    print("Конвертация PDF в изображения...")
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    doc = fitz.open(pdf_path)

    # Коэффициент масштабирования
    scale = 72.0 / dpi

    pages = []
    full_text_parts = []

    for page_num, (pil_image, fitz_page) in enumerate(zip(images, doc), start=1):
        print(f"Анализ структуры страницы {page_num} через YOLO...")
        image = np.array(pil_image)
        # Конвертируем из RGB (PIL) в BGR (OpenCV)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Запускаем инференс YOLO
        results = model(image_bgr, verbose=False)
        result = results[0]

        regions = []
        page_text_blocks = []

        # Проверяем, обнаружены ли какие-то сегменты/боксы
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            detected_blocks = []

            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = label_map.get(cls_id, f"object_{cls_id}")

                detected_blocks.append({
                    "bbox": xyxy,
                    "conf": conf,
                    "type": cls_name
                })

            # Сортируем блоки сверху вниз (по координате Y1), чтобы текст шёл в правильном порядке
            detected_blocks.sort(key=lambda b: b["bbox"][1])

            # Извлекаем текст через PyMuPDF по координатам
            for block in detected_blocks:
                x1, y1, x2, y2 = block["bbox"]

                # Переводим пиксели картинки в координаты PDF-документа
                rect = fitz.Rect(x1 * scale, y1 * scale, x2 * scale, y2 * scale)

                # Извлекаем текст, попадающий в эту рамку
                extracted_text = fitz_page.get_text("text", clip=rect).strip()

                if extracted_text:
                    page_text_blocks.append(extracted_text)

                regions.append({
                    "type": block["type"],
                    "confidence": round(block["conf"], 4),
                    "bbox_pixels": [round(c, 2) for c in block["bbox"]],
                    "bbox_pdf": [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)],
                    "extracted_text": extracted_text
                })
        else:
            # Если YOLO ничего не нашел на странице, забираем текст со страницы целиком
            print(f"Пояснение: На странице {page_num} структура не обнаружена, извлекаем текст целиком.")
            page_text = fitz_page.get_text("text").strip()
            if page_text:
                page_text_blocks.append(page_text)

        page_text = "\n\n".join(page_text_blocks)
        pages.append({
            "page": page_num,
            "regions_count": len(regions),
            "text": page_text,
            "regions": regions,
        })
        if page_text.strip():
            full_text_parts.append(page_text.strip())

    doc.close()
    full_text = "\n\n".join(full_text_parts)

    return {
        "method": "YOLOv8 + PyMuPDF",
        "source": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сегментация через YOLOv8 + извлечение PyMuPDF")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--output-dir", default="outputs", help="Папка для результатов")
    parser.add_argument("--dpi", type=int, default=200, help="Разрешение для рендеринга картинок")
    parser.add_argument("--poppler-path", default=r"C:\poppler\poppler-24.08.0\Library\bin", help="Путь к Poppler bin")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    start = time.perf_counter()

    try:
        result = extract_layout_yolo_pymupdf(args.pdf_path, dpi=args.dpi, poppler_path=args.poppler_path)
        elapsed = time.perf_counter() - start

        stem = Path(args.pdf_path).stem
        txt_path = out_dir / f"{stem}_yolo_pymupdf.txt"
        json_path = out_dir / f"{stem}_yolo_pymupdf.json"

        save_text(txt_path, result["full_text"])
        save_json(json_path, result)

        print_header("YOLOv8 + PyMuPDF")
        print(f"Файл: {args.pdf_path} | Время выполнения: {elapsed:.3f} сек")
        print(f"Сохранено в TXT: {txt_path}")
        print(f"Сохранено в JSON: {json_path}")
        print("\nПредпросмотр извлеченного текста:")
        print(preview(result["full_text"]))

    except Exception as e:
        print(f"\n[Ошибка при выполнении программы]: {e}")


if __name__ == "__main__":
    main()