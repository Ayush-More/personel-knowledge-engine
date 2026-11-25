from pydantic import BaseModel , Field , field_validator    

class AskRequest(BaseModel):
    question: str = Field(min_length=1 , max_length=2000)
    top_k: int = Field(default=5 , ge=1 , le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot contain only whitespace.")

        return value

class CitationResponse(BaseModel):
    chunk_id: str
    text: str

class AskResponse(BaseModel):
    question: str
    answer: str
    citation: list[CitationResponse]
    insufficient_evidence: bool

    