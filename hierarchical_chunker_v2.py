from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


Granularity = Literal[
    "chapter",
    "section",
    "paragraph_group",
    "paragraph",
]


@dataclass
class Paragraph:
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class Node:
    level: str
    number: str
    title: str
    path: list[str]
    paragraphs: list[Paragraph] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


def estimate_tokens(text: str) -> int:
    """
    Грубая оценка количества токенов.
    Для русского текста обычно 1 токен ≈ 3-4 символа.
    """
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    """
    Базовая нормализация текста после парсинга PDF.
    """
    text = text.replace("\ufeff", "")
    text = text.replace("\x0c", "\n")
    text = text.replace("\u00a0", " ")

    # Исправление частых дефисных переносов:
    # "электро-\nпитания" -> "электропитания"
    text = re.sub(r"([А-Яа-яA-Za-z])-\n([А-Яа-яA-Za-z])", r"\1\2", text)

    # Убираем лишние пробелы в строках
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    return "\n".join(lines)


def is_page_marker_line(line: str) -> bool:
    """
    Определяет мусорные строки с номерами страниц.
    Например:
    2
    15
    - 4 -
    """
    s = line.strip()

    if not s:
        return False

    if re.fullmatch(r"\d{1,3}", s):
        return True

    if re.fullmatch(r"-\s*\d{1,3}\s*-", s):
        return True

    return False


def is_parsed_page_header(line: str) -> bool:
    """
    Определяет служебные заголовки вида:
    <PARSED TEXT FOR PAGE: 1 / 167>
    """
    return bool(re.fullmatch(r"<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>", line.strip()))


def extract_page_number_from_marker(line: str) -> int | None:
    match = re.fullmatch(r"<PARSED TEXT FOR PAGE:\s*(\d+)\s*/\s*\d+>", line.strip())
    if match:
        return int(match.group(1))
    return None


def is_toc_line(line: str) -> bool:
    """
    Определяет строки содержания:
    1.НАЗНАЧЕНИЕ.....................................4
    2.ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ ..................11
    ПРИЛОЖЕНИЕ 1. ... 30
    """
    s = line.strip()

    if "." * 3 in s:
        return True

    if re.search(r"\.{5,}\s*\d{1,3}$", s):
        return True

    if re.match(r"^\d+\.\s*[А-ЯA-ZЁ\s]+\.{3,}\s*\d+$", s):
        return True

    if re.match(r"^ПРИЛОЖЕНИЕ\s+\d+.*\.{3,}\s*\d+$", s, re.IGNORECASE):
        return True

    return False


def is_noise_line(line: str) -> bool:
    """
    Общий фильтр мусорных строк.
    """
    s = line.strip()

    if not s:
        return False

    if is_page_marker_line(s):
        return True

    if is_parsed_page_header(s):
        return True

    if is_toc_line(s):
        return True

    return False


def is_likely_real_heading(line: str) -> bool:
    """
    Дополнительная проверка, чтобы не принять техническую строку из таблицы
    за заголовок.
    """
    s = line.strip()

    if len(s) > 180:
        return False

    # Слишком много чисел — скорее всего таблица или техническая строка
    digit_count = sum(ch.isdigit() for ch in s)
    if len(s) > 0 and digit_count / len(s) > 0.35:
        return False

    # Строки вида "3. X XXXВ X.XА XXВт XoC" лучше не считать заголовками
    technical_patterns = [
        r"X{2,}",
        r"\d+\s*В",
        r"\d+\s*А",
        r"\d+\s*Вт",
        r"\d+\s*кВА",
    ]
    technical_hits = sum(1 for p in technical_patterns if re.search(p, s, re.IGNORECASE))
    if technical_hits >= 2:
        return False

    return True


def detect_heading(line: str) -> tuple[str, str, str] | None:
    """
    Возвращает:
    (level, number, title)

    level:
    - chapter: 1. Назначение
    - section: 1.1 Подраздел
    - subsection: 1.1.1 Подпункт
    - appendix: ПРИЛОЖЕНИЕ 1. ...
    """
    s = line.strip()

    if not s:
        return None

    if not is_likely_real_heading(s):
        return None

    # Приложения
    appendix_match = re.match(
        r"^(ПРИЛОЖЕНИЕ\s+(\d+))\.?\s*(.*)$",
        s,
        re.IGNORECASE,
    )
    if appendix_match:
        number = f"ПРИЛОЖЕНИЕ {appendix_match.group(2)}"
        title = appendix_match.group(3).strip(" .")
        return "chapter", number, title or number

    # Заголовки вида:
    # 1. Назначение
    # 2. Технические характеристики
    chapter_match = re.match(
        r"^(\d{1,2})\.\s+(.+)$",
        s,
    )
    if chapter_match:
        number = chapter_match.group(1)
        title = chapter_match.group(2).strip()

        # Чтобы не ловить случайные пункты внутри текста:
        # настоящие главы обычно начинаются с 1-99 и имеют короткий заголовок
        if len(title) <= 140:
            return "chapter", number, title

    # Заголовки вида:
    # 1.1 Подраздел
    # 5.17 Настройка
    section_match = re.match(
        r"^(\d{1,2}\.\d{1,2})\.?\s+(.+)$",
        s,
    )
    if section_match:
        number = section_match.group(1)
        title = section_match.group(2).strip()
        if len(title) <= 160:
            return "section", number, title

    # Заголовки вида:
    # 1.1.1 Подпункт
    subsection_match = re.match(
        r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\.?\s+(.+)$",
        s,
    )
    if subsection_match:
        number = subsection_match.group(1)
        title = subsection_match.group(2).strip()
        if len(title) <= 160:
            return "subsection", number, title

    return None


def split_to_paragraphs(text: str) -> list[tuple[str, int | None]]:
    """
    Преобразует распарсенный текст в абзацы.
    Возвращает список:
    [(paragraph_text, page_number), ...]

    Важно:
    - заголовки выделяются отдельными абзацами;
    - одиночные номера страниц удаляются;
    - строки внутри одного абзаца склеиваются.
    """
    text = normalize_text(text)
    lines = text.splitlines()

    paragraphs: list[tuple[str, int | None]] = []
    buffer: list[str] = []
    current_page: int | None = None

    def flush():
        nonlocal buffer
        if not buffer:
            return

        paragraph = " ".join(buffer).strip()
        paragraph = re.sub(r"\s+", " ", paragraph)

        if paragraph and not is_noise_line(paragraph):
            paragraphs.append((paragraph, current_page))

        buffer = []

    for raw_line in lines:
        line = raw_line.strip()

        page_num = extract_page_number_from_marker(line)
        if page_num is not None:
            flush()
            current_page = page_num
            continue

        if not line:
            flush()
            continue

        if is_noise_line(line):
            flush()
            continue

        heading = detect_heading(line)
        if heading:
            flush()
            paragraphs.append((line, current_page))
            continue

        buffer.append(line)

    flush()

    return paragraphs


def build_hierarchy(paragraphs: list[tuple[str, int | None]]) -> Node:
    """
    Строит дерево документа.
    Корневой узел: document.
    """
    root = Node(
        level="document",
        number="",
        title="document",
        path=[],
    )

    current_chapter: Node | None = None
    current_section: Node | None = None
    current_subsection: Node | None = None

    for paragraph_text, page in paragraphs:
        heading = detect_heading(paragraph_text)

        if heading:
            level, number, title = heading

            if level == "chapter":
                node = Node(
                    level="chapter",
                    number=number,
                    title=title,
                    path=[f"{number}. {title}"],
                    page_start=page,
                    page_end=page,
                )
                root.children.append(node)
                current_chapter = node
                current_section = None
                current_subsection = None
                continue

            if level == "section":
                if current_chapter is None:
                    current_chapter = Node(
                        level="chapter",
                        number="unknown",
                        title="Без главы",
                        path=["Без главы"],
                    )
                    root.children.append(current_chapter)

                node = Node(
                    level="section",
                    number=number,
                    title=title,
                    path=current_chapter.path + [f"{number}. {title}"],
                    page_start=page,
                    page_end=page,
                )
                current_chapter.children.append(node)
                current_section = node
                current_subsection = None
                continue

            if level == "subsection":
                if current_chapter is None:
                    current_chapter = Node(
                        level="chapter",
                        number="unknown",
                        title="Без главы",
                        path=["Без главы"],
                    )
                    root.children.append(current_chapter)

                if current_section is None:
                    parent_number = ".".join(number.split(".")[:2])
                    current_section = Node(
                        level="section",
                        number=parent_number,
                        title="Без названия",
                        path=current_chapter.path + [parent_number],
                    )
                    current_chapter.children.append(current_section)

                node = Node(
                    level="subsection",
                    number=number,
                    title=title,
                    path=current_section.path + [f"{number}. {title}"],
                    page_start=page,
                    page_end=page,
                )
                current_section.children.append(node)
                current_subsection = node
                continue

        paragraph = Paragraph(
            text=paragraph_text,
            page_start=page,
            page_end=page,
        )

        target: Node

        if current_subsection is not None:
            target = current_subsection
        elif current_section is not None:
            target = current_section
        elif current_chapter is not None:
            target = current_chapter
        else:
            current_chapter = Node(
                level="chapter",
                number="unknown",
                title="Без главы",
                path=["Без главы"],
            )
            root.children.append(current_chapter)
            target = current_chapter

        target.paragraphs.append(paragraph)

        if page is not None:
            if target.page_start is None:
                target.page_start = page
            target.page_end = page

    update_page_ranges(root)

    return root


def update_page_ranges(node: Node) -> None:
    """
    Обновляет page_start/page_end у родительских узлов
    на основе вложенных элементов.
    """
    pages_start = []
    pages_end = []

    for p in node.paragraphs:
        if p.page_start is not None:
            pages_start.append(p.page_start)
        if p.page_end is not None:
            pages_end.append(p.page_end)

    for child in node.children:
        update_page_ranges(child)
        if child.page_start is not None:
            pages_start.append(child.page_start)
        if child.page_end is not None:
            pages_end.append(child.page_end)

    if pages_start:
        node.page_start = min(pages_start)

    if pages_end:
        node.page_end = max(pages_end)


def iter_nodes(node: Node) -> list[Node]:
    """
    Возвращает все узлы дерева, кроме root.
    """
    result = []

    for child in node.children:
        result.append(child)
        result.extend(iter_nodes(child))

    return result


def collect_text_from_node(node: Node) -> str:
    """
    Собирает весь текст из узла и его потомков.
    """
    parts = []

    for p in node.paragraphs:
        parts.append(p.text)

    for child in node.children:
        child_text = collect_text_from_node(child)
        if child_text:
            parts.append(child_text)

    return "\n\n".join(parts).strip()


def collect_paragraphs_from_node(node: Node) -> list[Paragraph]:
    """
    Собирает все абзацы из узла и его потомков.
    """
    result = list(node.paragraphs)

    for child in node.children:
        result.extend(collect_paragraphs_from_node(child))

    return result


def make_chunk(
    chunk_id: int,
    level: str,
    number: str,
    title: str,
    path: list[str],
    text: str,
    paragraphs_count: int,
    page_start: int | None,
    page_end: int | None,
) -> dict:
    return {
        "chunk_id": f"chunk_{chunk_id:05d}",
        "level": level,
        "section_number": number,
        "section_title": title,
        "path": path,
        "text": text,
        "paragraphs_count": paragraphs_count,
        "page_start": page_start,
        "page_end": page_end,
        "tokens_estimate": estimate_tokens(text),
    }


def build_chunks(
    root: Node,
    granularity: Granularity,
    max_tokens: int = 900,
    min_tokens: int = 80,
) -> list[dict]:
    """
    Создает чанки с разной крупностью.

    granularity:
    - chapter: один чанк на главу
    - section: один чанк на подраздел, если подразделов нет — по главам
    - paragraph_group: группы абзацев внутри ближайшего раздела
    - paragraph: каждый абзац отдельно
    """
    chunks: list[dict] = []
    chunk_counter = 1

    nodes = iter_nodes(root)

    if granularity == "chapter":
        chapter_nodes = [n for n in nodes if n.level == "chapter"]

        for node in chapter_nodes:
            text = collect_text_from_node(node)
            paragraphs_count = len(collect_paragraphs_from_node(node))

            if not text:
                continue

            chunks.append(
                make_chunk(
                    chunk_counter,
                    node.level,
                    node.number,
                    node.title,
                    node.path,
                    text,
                    paragraphs_count,
                    node.page_start,
                    node.page_end,
                )
            )
            chunk_counter += 1

        return chunks

    if granularity == "section":
        # Берем section/subsection.
        # Если внутри главы нет section, берем саму главу.
        section_like_nodes = [
            n for n in nodes
            if n.level in {"section", "subsection"}
        ]

        if not section_like_nodes:
            section_like_nodes = [
                n for n in nodes
                if n.level == "chapter"
            ]

        for node in section_like_nodes:
            text = collect_text_from_node(node)
            paragraphs_count = len(collect_paragraphs_from_node(node))

            if not text:
                continue

            chunks.append(
                make_chunk(
                    chunk_counter,
                    node.level,
                    node.number,
                    node.title,
                    node.path,
                    text,
                    paragraphs_count,
                    node.page_start,
                    node.page_end,
                )
            )
            chunk_counter += 1

        return chunks

    if granularity == "paragraph":
        for node in nodes:
            for p in node.paragraphs:
                text = p.text.strip()

                if not text:
                    continue

                # Не сохраняем совсем короткий технический мусор
                if estimate_tokens(text) < 2:
                    continue

                chunks.append(
                    make_chunk(
                        chunk_counter,
                        "paragraph",
                        node.number,
                        node.title,
                        node.path,
                        text,
                        1,
                        p.page_start,
                        p.page_end,
                    )
                )
                chunk_counter += 1

        return chunks

    if granularity == "paragraph_group":
        for node in nodes:
            paragraphs = node.paragraphs

            if not paragraphs:
                continue

            group: list[Paragraph] = []
            group_tokens = 0

            def flush_group():
                nonlocal chunk_counter, group, group_tokens

                if not group:
                    return

                text = "\n\n".join(p.text for p in group).strip()

                if not text:
                    group = []
                    group_tokens = 0
                    return

                if estimate_tokens(text) < 2:
                    group = []
                    group_tokens = 0
                    return

                pages = [
                    p.page_start for p in group
                    if p.page_start is not None
                ]

                page_start = min(pages) if pages else None
                page_end = max(pages) if pages else None

                chunks.append(
                    make_chunk(
                        chunk_counter,
                        "paragraph_group",
                        node.number,
                        node.title,
                        node.path,
                        text,
                        len(group),
                        page_start,
                        page_end,
                    )
                )
                chunk_counter += 1

                group = []
                group_tokens = 0

            for paragraph in paragraphs:
                paragraph_tokens = estimate_tokens(paragraph.text)

                # Если один абзац сам по себе большой,
                # не режем его, потому что абзац — минимальная единица.
                if group and group_tokens + paragraph_tokens > max_tokens:
                    flush_group()

                group.append(paragraph)
                group_tokens += paragraph_tokens

                if group_tokens >= max_tokens:
                    flush_group()

            flush_group()

        return chunks

    raise ValueError(f"Unknown granularity: {granularity}")


def node_to_dict(node: Node) -> dict:
    """
    Преобразует дерево в JSON-совместимый словарь.
    """
    return {
        "level": node.level,
        "number": node.number,
        "title": node.title,
        "path": node.path,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "paragraphs": [asdict(p) for p in node.paragraphs],
        "children": [node_to_dict(child) for child in node.children],
    }


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical text chunker for parsed PDF text"
    )

    parser.add_argument(
        "input_path",
        help="Path to parsed .txt file",
    )

    parser.add_argument(
        "--output",
        default="chunks.json",
        help="Path to output chunks JSON",
    )

    parser.add_argument(
        "--structure-output",
        default="structure.json",
        help="Path to output hierarchy JSON",
    )

    parser.add_argument(
        "--granularity",
        choices=["chapter", "section", "paragraph_group", "paragraph"],
        default="paragraph_group",
        help="Chunk granularity",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max tokens for paragraph_group mode",
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=80,
        help="Min target tokens for paragraph_group mode",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")

    paragraphs = split_to_paragraphs(raw_text)
    root = build_hierarchy(paragraphs)

    chunks = build_chunks(
        root=root,
        granularity=args.granularity,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )

    structure = node_to_dict(root)

    save_json(chunks, args.output)
    save_json(structure, args.structure_output)

    print("Готово.")
    print(f"Входной файл: {input_path}")
    print(f"Чанки: {args.output}")
    print(f"Структура: {args.structure_output}")
    print(f"Режим разбиения: {args.granularity}")
    print(f"Количество чанков: {len(chunks)}")


if __name__ == "__main__":
    main()