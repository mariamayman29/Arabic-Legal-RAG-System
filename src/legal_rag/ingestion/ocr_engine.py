import logging
import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
import pymupdf
from paddleocr import PaddleOCR

from legal_rag.ingestion.normalize_text import normalize_arabic

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def create_ocr_engine() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _reconstruct_paragraphs(rec_texts: list[str], dt_polys: list) -> str:

    if not rec_texts:
        return ""
    if not dt_polys or len(rec_texts) != len(dt_polys):
        logger.warning(
            "OCR result missing dt_polys or length mismatch; returning raw text"
        )
        return "\n".join(t for t in rec_texts if t)

    body_items: list[tuple[str, list]] = []
    for text, poly in zip(rec_texts, dt_polys):
        clean = text.strip()
        if not clean or clean.startswith("مادة"):
            continue
        body_items.append((clean, poly))

    if not body_items:
        return ""

    paragraphs: list[str] = []
    current_lines: list[str] = []

    for idx, (text, poly) in enumerate(body_items):
        bottom_y = max(float(poly[2][1]), float(poly[3][1]))

        is_last_line = idx == len(body_items) - 1
        has_vertical_gap = False
        if not is_last_line:
            next_poly = body_items[idx + 1][1]
            next_top_y = min(float(next_poly[0][1]), float(next_poly[1][1]))
            if (
                (next_top_y - bottom_y) > 4.0
            ):  # NOTE: 12.0px is a more conservative threshold at 2.5x zoom to avoid false mid-sentence periods.
                has_vertical_gap = True

        if (is_last_line or has_vertical_gap) and not text.endswith(
            (".", "!", "؟", ":", "،", "؛")
        ):
            text += "."

        current_lines.append(text)

        if has_vertical_gap or is_last_line:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

    return " ".join(paragraphs)


def ocr_arabic_cell(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float] | None,
    ocr_engine: PaddleOCR,
    zoom: float,
) -> str:
    if not bbox:
        logger.warning("No bounding box provided for OCR; returning empty string")
        return ""

    clip_rect = pymupdf.Rect(bbox)
    if clip_rect.width > 6 and clip_rect.height > 6:
        clip_rect = pymupdf.Rect(
            clip_rect.x0 + 1.9,
            clip_rect.y0 + 1.9,
            clip_rect.x1 - 1.9,
            clip_rect.y1 - 1.9,
        )

    if clip_rect.is_empty or clip_rect.width <= 0 or clip_rect.height <= 0:
        logger.warning("Invalid bounding box for OCR; returning empty string")
        return ""

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom), clip=clip_rect, alpha=False
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = np.pad(
        img, ((24, 24), (24, 24), (0, 0)), mode="constant", constant_values=255
    )

    results = list(ocr_engine.predict(img))
    if not results:
        logger.warning("No OCR results; returning empty string")
        return ""

    first_res = results[0]
    rec_texts = (
        first_res.get("rec_texts", [])
        if isinstance(first_res, dict)
        else getattr(first_res, "rec_texts", [])
    )
    dt_polys = (
        first_res.get("dt_polys", [])
        if isinstance(first_res, dict)
        else getattr(first_res, "dt_polys", [])
    )
    raw_text = _reconstruct_paragraphs(rec_texts, dt_polys)
    return normalize_arabic(raw_text)
