from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# ----------------------------
# Модели данных
# ----------------------------

@dataclass
class Paragraph:
    text: str
    page: int | None = None


@dataclass
class Section:
    id: str
    number: str | None
    title: str
    level: int
    paragraphs: list[Paragraph]
    children: list["Section"]


@dataclass
class Chunk:
    chunk_id: str
    level: str
    section_number: str | None
    section_title: str
    path: list[str]
    text: str
    paragraphs_count: int
    page_start: int | None
    page_end: int | None
    tokens_estimate: int


# ----------------------------
# Загрузка текста
# ----------------------------

def load_parsed_file(path: str | Path) -> tuple[str, list[dict[str, Any]] | None]:
    """
    Загружает txt или json.

    Для json ожидается структура:
    {
        "full_text": "...",
        "pages": [
            {"page": 1, "text": "..."},
            ...
        ]
    }
    """
    path = Path(path)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        full_text = data.get("full_text", "")
        pages = data.get("pages")
        return full_text, pages

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8"), None

    raise ValueError("Поддерживаются только .txt и .json файлы")


# ----------------------------
# Очистка текста
# ----------------------------

def normalize_text(text: str) -> str:
    """
    Нормализует текст после парсинга/OCR:
    - убирает лишние пробелы;
    - выравнивает переносы строк;
    - сохраняет абзацы.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_table_of_contents(text: str) -> str:
    """
    Убирает оглавление, чтобы строки вида:
    1. НАЗНАЧЕНИЕ ..... 4
    не воспринимались как реальные разделы.

    Для технических документов это полезно, но при необходимости
    функцию можно отключить.
    """
    pattern = re.compile(
        r"Содержание\s+.*?(?=\n\s*1[\.\s]+[А-ЯA-ZЁ])",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub("", text)


# ----------------------------
# Поиск заголовков
# ----------------------------

HEADING_PATTERNS = [
    # 1. Назначение
    # 1 Назначение
    # 1.1 Формат PDF
    re.compile(
        r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[А-ЯA-ZЁ][^\n]{2,160})\s*$"
    ),

    # ПРИЛОЖЕНИЕ 1. ...
    re.compile(
        r"^\s*(?P<number>ПРИЛОЖЕНИЕ\s+\d+)\.?\s*(?P<title>[^\n]{0,180})\s*$",
        flags=re.IGNORECASE,
    ),
]


def detect_heading(line: str) -> tuple[str, str, int] | None:
    """
    Возвращает:
    number, title, level

    level:
    - 1 для главы: 1, 2, 3
    - 2 для подпункта: 1.1, 1.2
    - 3 для 1.1.1 и т.д.
    - 1 для приложения
    """
    clean = line.strip()

    # Отсекаем строки оглавления с точками и номером страницы
    if re.search(r"\.{3,}\s*\d+\s*$", clean):
        return None

    for pattern in HEADING_PATTERNS:
        match = pattern.match(clean)
        if not match:
            continue

        number = match.group("number").strip()
        title = match.group("title").strip()

        if number.upper().startswith("ПРИЛОЖЕНИЕ"):
            return number, title, 1

        level = number.count(".") + 1
        return number, title, level

    return None


# ----------------------------
# Разбиение на абзацы
# ----------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """
    Абзац — минимальная единица.
    Здесь абзацем считается блок текста, отделённый пустой строкой.
    Если после парсинга пустые строки потеряны, можно улучшить логику:
    например, склеивать строки до точки.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []

    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Убираем одиночные переносы внутри абзаца
        paragraph = re.sub(r"\n+", " ", paragraph)
        paragraph = re.sub(r"\s{2,}", " ", paragraph)

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def estimate_tokens(text: str) -> int:
    """
    Грубая оценка токенов для русского текста.
    Обычно 1 токен ≈ 3–4 символа.
    Берём консервативно: 1 токен ≈ 3.5 символа.
    """
    return max(1, int(len(text) / 3.5))


# ----------------------------
# Построение иерархии
# ----------------------------

def build_hierarchy(text: str) -> Section:
    """
    Строит дерево разделов.
    """
    root = Section(
        id="root",
        number=None,
        title="Документ",
        level=0,
        paragraphs=[],
        children=[],
    )

    stack: list[Section] = [root]

    current_buffer: list[str] = []

    def flush_buffer_to_current_section() -> None:
        nonlocal current_buffer
        if not current_buffer:
            return

        block = "\n".join(current_buffer).strip()
        paragraphs = split_into_paragraphs(block)

        for paragraph in paragraphs:
            stack[-1].paragraphs.append(Paragraph(text=paragraph))

        current_buffer = []

    lines = text.splitlines()

    for line in lines:
        heading = detect_heading(line)

        if heading:
            flush_buffer_to_current_section()

            number, title, level = heading

            new_section = Section(
                id=f"section_{number}",
                number=number,
                title=title,
                level=level,
                paragraphs=[],
                children=[],
            )

            while stack and stack[-1].level >= level:
                stack.pop()

            stack[-1].children.append(new_section)
            stack.append(new_section)
        else:
            current_buffer.append(line)

    flush_buffer_to_current_section()

    return root


# ----------------------------
# Генерация чанков
# ----------------------------

def collect_section_text(section: Section) -> list[Paragraph]:
    """
    Собирает все абзацы раздела вместе с дочерними подразделами.
    """
    result = list(section.paragraphs)

    for child in section.children:
        result.extend(collect_section_text(child))

    return result


def section_path(section: Section, parents: list[Section]) -> list[str]:
    path = []

    for item in parents + [section]:
        if item.level == 0:
            continue

        if item.number:
            path.append(f"{item.number}. {item.title}".strip())
        else:
            path.append(item.title)

    return path


def make_chunk(
    chunk_id: str,
    level: str,
    section: Section,
    parents: list[Section],
    paragraphs: list[Paragraph],
) -> Chunk:
    text = "\n\n".join(p.text for p in paragraphs)

    pages = [p.page for p in paragraphs if p.page is not None]

    return Chunk(
        chunk_id=chunk_id,
        level=level,
        section_number=section.number,
        section_title=section.title,
        path=section_path(section, parents),
        text=text,
        paragraphs_count=len(paragraphs),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        tokens_estimate=estimate_tokens(text),
    )


def generate_chunks(
    root: Section,
    granularity: str = "section",
    max_tokens: int = 800,
) -> list[Chunk]:
    """
    granularity:
    - chapter
    - section
    - paragraph_group
    - paragraph

    max_tokens используется для paragraph_group.
    """

    chunks: list[Chunk] = []
    counter = 1

    def walk(section: Section, parents: list[Section]) -> None:
        nonlocal counter

        if section.level == 0:
            for child in section.children:
                walk(child, [])
            return

        # Режим: глава
        if granularity == "chapter":
            if section.level == 1:
                paragraphs = collect_section_text(section)
                if paragraphs:
                    chunks.append(
                        make_chunk(
                            chunk_id=f"chunk_{counter:05d}",
                            level="chapter",
                            section=section,
                            parents=parents,
                            paragraphs=paragraphs,
                        )
                    )
                    counter += 1
            else:
                return

        # Режим: раздел / подраздел
        elif granularity == "section":
            paragraphs = section.paragraphs

            if paragraphs:
                chunks.append(
                    make_chunk(
                        chunk_id=f"chunk_{counter:05d}",
                        level="section",
                        section=section,
                        parents=parents,
                        paragraphs=paragraphs,
                    )
                )
                counter += 1

            for child in section.children:
                walk(child, parents + [section])

        # Режим: группы абзацев
        elif granularity == "paragraph_group":
            buffer: list[Paragraph] = []
            buffer_tokens = 0

            for paragraph in section.paragraphs:
                paragraph_tokens = estimate_tokens(paragraph.text)

                if buffer and buffer_tokens + paragraph_tokens > max_tokens:
                    chunks.append(
                        make_chunk(
                            chunk_id=f"chunk_{counter:05d}",
                            level="paragraph_group",
                            section=section,
                            parents=parents,
                            paragraphs=buffer,
                        )
                    )
                    counter += 1
                    buffer = []
                    buffer_tokens = 0

                buffer.append(paragraph)
                buffer_tokens += paragraph_tokens

            if buffer:
                chunks.append(
                    make_chunk(
                        chunk_id=f"chunk_{counter:05d}",
                        level="paragraph_group",
                        section=section,
                        parents=parents,
                        paragraphs=buffer,
                    )
                )
                counter += 1

            for child in section.children:
                walk(child, parents + [section])

        # Режим: абзац
        elif granularity == "paragraph":
            for paragraph in section.paragraphs:
                chunks.append(
                    make_chunk(
                        chunk_id=f"chunk_{counter:05d}",
                        level="paragraph",
                        section=section,
                        parents=parents,
                        paragraphs=[paragraph],
                    )
                )
                counter += 1

            for child in section.children:
                walk(child, parents + [section])

        else:
            raise ValueError(
                "granularity должен быть одним из: chapter, section, paragraph_group, paragraph"
            )

    walk(root, [])
    return chunks


# ----------------------------
# Сохранение результата
# ----------------------------

def save_chunks(chunks: list[Chunk], output_path: str | Path) -> None:
    output_path = Path(output_path)
    data = [asdict(chunk) for chunk in chunks]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Иерархическое разбиение спарсенного текста на чанки"
    )

    parser.add_argument("input_path", help="Путь к .txt или .json после парсинга")
    parser.add_argument("--output", default="chunks.json", help="Куда сохранить чанки")
    parser.add_argument(
        "--granularity",
        default="section",
        choices=["chapter", "section", "paragraph_group", "paragraph"],
        help="Крупность разбиения",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Максимальный размер чанка для paragraph_group",
    )
    parser.add_argument(
        "--keep-toc",
        action="store_true",
        help="Не удалять оглавление",
    )

    args = parser.parse_args()

    start = time.perf_counter()

    text, _pages = load_parsed_file(args.input_path)
    text = normalize_text(text)

    if not args.keep_toc:
        text = remove_table_of_contents(text)

    hierarchy = build_hierarchy(text)

    chunks = generate_chunks(
        root=hierarchy,
        granularity=args.granularity,
        max_tokens=args.max_tokens,
    )

    save_chunks(chunks, args.output)

    elapsed = time.perf_counter() - start

    total_tokens = sum(chunk.tokens_estimate for chunk in chunks)

    print("Готово")
    print(f"Файл: {args.input_path}")
    print(f"Режим чанкинга: {args.granularity}")
    print(f"Количество чанков: {len(chunks)}")
    print(f"Оценка токенов: {total_tokens}")
    print(f"Время: {elapsed:.3f} сек")
    print(f"Результат: {args.output}")


if __name__ == "__main__":
    main()