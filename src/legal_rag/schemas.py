import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

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


class ChunkMetadata(BaseModel):
    article_number: int = Field(..., ge=1)
    language: Literal["en", "ar"]
    strategy: str = Field(..., description="Chunking strategy name")
    chunk_index: int = Field(0, ge=0)
    total_chunks: int = Field(1, ge=1)
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    source_page: int = Field(..., ge=1)
    is_repealed: bool = False
    citation: str


class ChunkSchema(BaseModel):
    chunk_id: str = Field(
        ...,
        description="Deterministic unique ID, e.g., 'art_12_en_0'",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The core article body text used for LLM synthesis.",
    )
    context_header: str = Field(
        default="",
        description="Prepended breadcrumbs (e.g., Book > Chapter > Section).",
    )
    metadata: ChunkMetadata

    @computed_field
    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.chunk_id))

    @property
    def full_text(self) -> str:
        if self.context_header:
            return f"{self.context_header}\n{self.content}".strip()
        return self.content

    def get_embedding_text(self, prefix: str = "") -> str:
        base_text = self.full_text
        return f"{prefix}{base_text}" if prefix else base_text

    def to_qdrant_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "context_header": self.context_header,
            **self.metadata.model_dump(),
        }
