from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_rag.utils.config import settings


class ArticleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    article_number: int = Field(..., ge=1, description="official article number")
    book: str = Field(..., min_length=1, description="top-level book title")
    chapter: str = Field(..., min_length=1, description="chapter title")
    section: str | None = Field(
        default=None,
        description="section title or None if the article does not belong to a section",
    )
    topic: str | None = Field(
        default=None,
        description="topic title or None if the article does not belong to a topic",
    )
    subtopic: str | None = Field(
        default=None,
        description="subtopic title or None if the article does not belong to a subtopic",
    )
    ar_text: str = Field(
        ...,
        min_length=1,
        description="Arabic text of the article or (repealed) if the article has been repealed",
    )
    en_text: str = Field(
        ...,
        min_length=1,
        description="English text of the article or (repealed) if the article has been repealed",
    )
    is_repealed: bool = Field(
        default=False,
        description="True if the article has been repealed, False otherwise",
    )
    source_page: int = Field(
        ...,
        ge=1,
        description="page number in the source document where the article is found",
    )
    citation: str = Field(
        ...,
        min_length=1,
        description="formatted legal citation of the article in the source document",
    )

    @model_validator(mode="after")
    def validate_repealed_text(self):
        if self.is_repealed:
            if (
                self.ar_text != settings.REPEALED_TEXT_AR
                or self.en_text != settings.REPEALED_TEXT_EN
            ):
                raise ValueError(
                    "If 'is_repealed' is True, both 'ar_text' and 'en_text' must be '(repealed)'."
                )
        else:
            if (
                self.ar_text == settings.REPEALED_TEXT_AR
                or self.en_text == settings.REPEALED_TEXT_EN
            ):
                raise ValueError(
                    "If 'is_repealed' is False, neither 'ar_text' nor 'en_text' can be '(repealed)'."
                )

        return self

    @model_validator(mode="after")
    def verify_citation_matches_article(self):
        expected = f"Egyptian Civil Code, Article {self.article_number}"
        if self.citation != expected:
            raise ValueError(
                f"Invalid citation '{self.citation}'. Expected '{expected}'."
            )
        return self
