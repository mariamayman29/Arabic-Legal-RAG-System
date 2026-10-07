import pytest
from pydantic import ValidationError

from legal_rag.schemas import ArticleSchema
from legal_rag.utils.config import settings


def test_repealed_article_with_wrong_text():
    with pytest.raises(ValidationError, match="must be"):
        ArticleSchema(
            article_number=55,
            book="Provisions of the Civil Code",
            chapter="Introductory Chapter",
            ar_text="نص عادي غير ملغي",
            en_text="Normal non-repealed text",
            is_repealed=True,
            source_page=9,
            citation="Egyptian Civil Code, Article 55",
        )


def test_active_article_with_repealed_placeholder():
    with pytest.raises(ValidationError, match="neither"):
        ArticleSchema(
            article_number=2,
            book="Provisions of the Civil Code",
            chapter="Introductory Chapter",
            ar_text=settings.REPEALED_TEXT_AR,
            en_text=settings.REPEALED_TEXT_EN,
            is_repealed=False,
            source_page=1,
            citation="Egyptian Civil Code, Article 2",
        )


def test_invalid_citation_format():
    with pytest.raises(ValidationError, match="Invalid citation"):
        ArticleSchema(
            article_number=10,
            book="Provisions of the Civil Code",
            chapter="Introductory Chapter",
            ar_text="نص المادة العاشرة",
            en_text="Text of article ten",
            is_repealed=False,
            source_page=2,
            citation="Egyptian Civil Code, Article 999",
        )
