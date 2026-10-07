from legal_rag.ingestion.normalize_text import normalize_arabic, normalize_english


def test_normalize_arabic():
    raw_ocr = (
        "مادة 1\n"
        "تسرى النصوص التشريعية على جميع المسائل\n"
        "التى تتناولها هذه النصوص في لفظها أو في فحواها."
    )
    cleaned = normalize_arabic(raw_ocr)
    assert "مادة" not in cleaned
    assert "\n" not in cleaned
    assert cleaned == (
        "تسرى النصوص التشريعية على جميع المسائل "
        "التى تتناولها هذه النصوص في لفظها او في فحواها."
    )


def test_normalize_english():
    raw_en = "Article 3\nPeriods of limitation will be calculated\naccording to the Gregorian calendar."
    cleaned = normalize_english(raw_en)
    assert (
        cleaned
        == "Periods of limitation will be calculated according to the Gregorian calendar."
    )
