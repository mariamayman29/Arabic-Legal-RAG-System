import json
import logging
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import ValidationError

from legal_rag.ingestion.normalize_text import normalize_english
from legal_rag.ingestion.ocr_engine import create_ocr_engine, ocr_arabic_cell
from legal_rag.ingestion.trace_hierarchy import (
    ARABIC_CHAR_RE,
    ARTICLE_START_RE,
    REPEALED_RANGE_RE,
    HierarchyTracker,
)
from legal_rag.schemas import ArticleSchema
from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _extract_en_and_ar_bbox(
    row: list[str | None],
    cells: list[tuple[float, float, float, float] | None],
    page_width: float,
) -> tuple[str, tuple[float, float, float, float] | None]:
    """Uses absolute page geometry (a midpoint split) instead of unreliable table column indexes to separate languages and
    it safely extracts English text from the left half, prevents Arabic character leaks,
    and mathematically merges shattered bounding boxes on the right half into one clean Arabic cell for OCR"""

    midpoint = page_width / 2.0
    en_parts: list[str] = []
    ar_rects: list[pymupdf.Rect] = []

    for idx, cell_bbox in enumerate(cells):
        text = (row[idx] or "").strip() if idx < len(row) else ""
        is_arabic_text = bool(ARABIC_CHAR_RE.search(text))

        if cell_bbox is not None:
            rect = pymupdf.Rect(cell_bbox)
            if rect.x0 < midpoint and text and not is_arabic_text:
                en_parts.append(text)
            if rect.x1 > midpoint:
                ar_x0 = max(rect.x0, midpoint)
                ar_rects.append(pymupdf.Rect(ar_x0, rect.y0, rect.x1, rect.y1))
        elif text and not is_arabic_text and idx < len(cells) - 1:
            en_parts.append(text)

    en_raw = " ".join(en_parts).strip()
    if not ar_rects:
        return en_raw, None

    merged_ar = ar_rects[0]
    for r in ar_rects[1:]:
        merged_ar |= r

    return en_raw, (merged_ar.x0, merged_ar.y0, merged_ar.x1, merged_ar.y1)


def extract_corpus(
    pdf_path: Path = settings.RAW_DATA_PATH,
    output_path: Path = settings.RAW_CORPUS_PATH,
    max_pages: int | None = None,
) -> list[ArticleSchema]:
    ocr_engine = create_ocr_engine()
    tracker = HierarchyTracker()

    articles_by_num: dict[int, dict[str, Any]] = {}
    current_art_num: int | None = None
    stats = {"empty_ar": 0, "empty_en": 0, "dropped_rows": 0}

    with pymupdf.open(pdf_path) as doc:
        total_pages = min(len(doc), max_pages) if max_pages else len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            logger.info(
                f"Processing page {page_num}/{total_pages} "
                f"(Articles extracted: {len(articles_by_num)})"
            )

            for table in page.find_tables():
                for row_idx, row in enumerate(table.extract()):
                    if not row:
                        continue

                    en_raw, ar_bbox = _extract_en_and_ar_bbox(
                        row, table.rows[row_idx].cells, page.rect.width
                    )

                    if not en_raw:
                        if (
                            ar_bbox is not None
                            and current_art_num
                            and not articles_by_num[current_art_num]["is_repealed"]
                        ):
                            ar_cont = ocr_arabic_cell(
                                page, ar_bbox, ocr_engine, settings.OCR_ZOOM_FACTOR
                            )
                            if ar_cont:
                                rec = articles_by_num[current_art_num]
                                rec["ar_text"] = f"{rec['ar_text']} {ar_cont}".strip()
                        continue

                    if m := REPEALED_RANGE_RE.search(en_raw):
                        for num in range(int(m.group(1)), int(m.group(2)) + 1):
                            articles_by_num[num] = tracker.build_record(
                                num,
                                settings.REPEALED_TEXT_AR,
                                settings.REPEALED_TEXT_EN,
                                True,
                                page_num,
                            )
                        current_art_num = None

                    elif m := ARTICLE_START_RE.match(en_raw):
                        current_art_num = int(m.group(1))
                        ar_clean = (
                            ocr_arabic_cell(
                                page, ar_bbox, ocr_engine, settings.OCR_ZOOM_FACTOR
                            )
                            if ar_bbox
                            else ""
                        )
                        if not ar_clean:
                            logger.warning(
                                f"Empty Arabic text for Article {current_art_num} on page {page_num}"
                            )
                            stats["empty_ar"] += 1
                        articles_by_num[current_art_num] = tracker.build_record(
                            current_art_num,
                            ar_clean,
                            normalize_english(en_raw),
                            False,
                            page_num,
                        )

                    elif tracker.try_update(en_raw):
                        current_art_num = None

                    elif (
                        current_art_num
                        and not articles_by_num[current_art_num]["is_repealed"]
                    ):
                        rec = articles_by_num[current_art_num]
                        en_cont = normalize_english(en_raw)
                        if en_cont:
                            rec["en_text"] = f"{rec['en_text']} {en_cont}".strip()
                        else:
                            logger.warning(
                                f"Empty English continuation for Article {current_art_num} on page {page_num}"
                            )
                            stats["empty_en"] += 1

                        if ar_bbox is not None:
                            ar_cont = ocr_arabic_cell(
                                page, ar_bbox, ocr_engine, settings.OCR_ZOOM_FACTOR
                            )
                            if ar_cont:
                                rec["ar_text"] = f"{rec['ar_text']} {ar_cont}".strip()
                            else:
                                logger.warning(
                                    f"Empty Arabic continuation for Article {current_art_num} on page {page_num}"
                                )
                                stats["empty_ar"] += 1
                    else:
                        logger.warning(f"Dropped row on page {page_num}: {en_raw!r}")
                        stats["dropped_rows"] += 1

    logger.info(
        f"Extraction summary: {len(articles_by_num)} articles | "
        f"empty_ar: {stats['empty_ar']} | empty_en: {stats['empty_en']} | "
        f"dropped_rows: {stats['dropped_rows']}"
    )

    sorted_records = [articles_by_num[n] for n in sorted(articles_by_num)]

    raw_save_path = getattr(settings, "RAW_CORPUS_PATH", Path("data/raw_corpus.json"))
    raw_save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_save_path.write_text(
        json.dumps(sorted_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved raw extracted records -> {raw_save_path}")

    validated: list[ArticleSchema] = []
    for rec in sorted_records:
        try:
            validated.append(ArticleSchema(**rec))
        except ValidationError as exc:
            logger.error(
                f"Validation failed for Article {rec['article_number']} "
                f"(page {rec['source_page']}): en_text={rec['en_text']!r}, ar_text={rec['ar_text']!r} -> {exc}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([r.model_dump() for r in validated], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return validated


if __name__ == "__main__":
    configure_logging()
    records = extract_corpus()
    logger.info(
        f"Extracted and validated {len(records)} articles -> {settings.RAW_CORPUS_PATH}"
    )
