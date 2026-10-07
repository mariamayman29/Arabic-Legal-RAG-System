import re


def normalize_arabic(text: str) -> str:

    if not text:
        return ""

    text = re.sub(r"^\s*مادة\s*[\(\d٠-٩\)]*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*مادة\s+[\(\d٠-٩\)]+\s*", "", text, flags=re.MULTILINE)

    # 2. Strip tatweel (kashida), Arabic diacritics (tashkeel), and stray OCR single quotes/backticks
    text = re.sub(r"[\u0640\u064B-\u065F\u0670'`´‘’]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = re.sub(r"^[\(\[]?[\d٠-٩]+[\)\]\.\-]?\s+", "", line)
        if line:
            cleaned_lines.append(line)

    joined = " ".join(cleaned_lines)

    # 6. Clean spaces before punctuation and fix introductory list colons
    joined = re.sub(r"\s+([.،؛:؟!])", r"\1", joined)
    joined = re.sub(
        r"(الاحوال الاتية|الشروط الاتية|الحالات الاتية|ما ياتي|ما يلي|كالاتي)\s*[\.]",
        r"\1:",
        joined,
    )

    if joined and not joined.endswith((".", "!", "؟")):
        joined += "."

    return joined


def normalize_english(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    lines[0] = re.sub(
        r"^Article\s*\d+\b\s*[\.\-:–—]?\s*", "", lines[0], flags=re.IGNORECASE
    )
    cleaned_lines = [
        re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()
    ]
    return " ".join(cleaned_lines)
