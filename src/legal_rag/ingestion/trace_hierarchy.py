import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BOOK_RE = re.compile(r"^BOOK\s+[IVXLCDM\d]+\b[:\.\-\s]*(.*)", re.IGNORECASE)
PART_RE = re.compile(
    r"^(?:PART\s+(?:[IVXLCDM\d]+|ONE|TWO|THREE|FOUR|FIVE)|"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+PART)\b[:\.\-\s]*(.*)",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM\d]+\b[:\.\-\s]*(.*)", re.IGNORECASE)
SECTION_RE = re.compile(r"^SECTION\s+[IVXLCDM\d]+\b[:\.\-\s]*(.*)", re.IGNORECASE)
TOPIC_RE = re.compile(r"^\(?\d+[\.\-\)]\s*(.*)")

REPEALED_RANGE_RE = re.compile(
    r"^(?:Article\s*\d+\b\s*[\.\-:–—]?\s*)?\*?\s*Articles?\s+(\d+)\s*(?:[-–—]|to)\s*(\d+)\b[^\n]*repealed",
    re.IGNORECASE | re.MULTILINE,
)
ARTICLE_START_RE = re.compile(r"^Article\s*(\d+)\b", re.IGNORECASE)
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass
class HierarchyTracker:
    book: str = "Provisions of the Civil Code"
    chapter: str = "Introductory Chapter"
    section: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    _pending_level: str | None = None

    def try_update(self, cell_text: str) -> bool:

        if ARABIC_CHAR_RE.search(cell_text):
            logger.debug("Cell contains Arabic text; skipping hierarchy update")
            return False

        lines = [line.strip() for line in cell_text.splitlines() if line.strip()]
        if not lines:
            return False

        if len(lines) >= 2 and (topic_match := TOPIC_RE.match(lines[0])):
            self.topic = topic_match.group(1).strip() or lines[0]
            self.subtopic = " ".join(lines[1:])
            self._pending_level = None
            logger.debug(
                f"TOPIC + SUBTOPIC in single cell -> topic={self.topic!r}, subtopic={self.subtopic!r}"
            )
            return True

        clean_line = " ".join(lines)

        if match := BOOK_RE.match(clean_line):
            title = match.group(1).strip()
            self.book = title or clean_line
            self.chapter = self.section = self.topic = self.subtopic = None
            self._pending_level = "book" if not title else None
            logger.debug(f"BOOK -> {self.book!r} (pending={self._pending_level})")
            return True

        if match := PART_RE.match(clean_line):
            title = match.group(1).strip()
            self._pending_level = "part" if not title else None
            logger.info(f"Ignored PART header: {clean_line!r}")
            return True

        if match := CHAPTER_RE.match(clean_line):
            title = match.group(1).strip()
            self.chapter = title or clean_line
            self.section = self.topic = self.subtopic = None
            self._pending_level = "chapter" if not title else None
            logger.debug(f"CHAPTER -> {self.chapter!r} (pending={self._pending_level})")
            return True

        if match := SECTION_RE.match(clean_line):
            title = match.group(1).strip()
            self.section = title or clean_line
            self.topic = self.subtopic = None
            self._pending_level = "section" if not title else None
            logger.debug(f"SECTION -> {self.section!r} (pending={self._pending_level})")
            return True

        if match := TOPIC_RE.match(clean_line):
            self.topic = match.group(1).strip() or clean_line
            self.subtopic = None
            self._pending_level = None
            logger.debug(f"TOPIC -> {self.topic!r}")
            return True

        if (
            self._pending_level
            and len(clean_line) < 120
            and not clean_line.endswith(".")
        ):
            if self._pending_level == "part":
                logger.info(f"Ignored split PART title: {clean_line!r}")
            else:
                setattr(self, self._pending_level, clean_line)
                logger.info(
                    f"Resolved split {self._pending_level.upper()} title -> {clean_line!r}"
                )
            self._pending_level = None
            return True

        if len(clean_line) < 70 and not clean_line.endswith("."):
            self.subtopic = clean_line
            logger.info(f"Generic subtopic match (audit this): {clean_line!r}")
            return True

        logger.debug(f"No hierarchy match for: {clean_line!r}")
        return False

    def build_record(
        self,
        article_number: int,
        ar_text: str,
        en_text: str,
        is_repealed: bool,
        source_page: int,
    ) -> dict[str, Any]:

        self._pending_level = None
        return {
            "article_number": article_number,
            "book": self.book,
            "chapter": self.chapter,
            "section": self.section,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "ar_text": ar_text,
            "en_text": en_text,
            "is_repealed": is_repealed,
            "source_page": source_page,
            "citation": f"Egyptian Civil Code, Article {article_number}",
        }
