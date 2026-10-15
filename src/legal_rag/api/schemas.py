from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's legal question")
    stream: bool = Field(False)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
