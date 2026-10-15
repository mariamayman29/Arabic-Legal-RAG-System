import re


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "en"


def extract_article_numbers(text: str) -> list[int]:
    matches = re.findall(
        r"(?:\barticle|\bart\.?|المادة|مادة|الماده|ماده)\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    return list({int(m) for m in matches})
