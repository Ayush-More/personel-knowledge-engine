from datetime import UTC, datetime

from pydantic import BaseModel, Field , model_validator
from typing import Literal , Any


class Document(BaseModel):
    document_id: str
    content: str = Field(min_length=1)
    source: str
    metadata: dict[str , str] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory = lambda: datetime.now(UTC)
    ) 

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content : str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    source: str = ""
    metadata: dict[str , str] = Field(default_factory=dict)

class SourceCitation(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)

class GeneratedAnswer(BaseModel):
    question: str
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False

class RetrievedEvidence(BaseModel):
    chunk_id: str
    document_id: str
    content: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    source: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float

class ToolResult(BaseModel):
    tool_name: str
    query: str
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    result_count: int = Field(ge=0)

class ToolCall(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    type: Literal["tool_call","final_answer"]
    tool_call: ToolCall | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentDecision":
        if self.type == "tool_call":
            if self.tool_call is None or self.answer is not None:
                raise ValueError("A tool_call decision must contain tool call"
                                 "And must not contain answer"
                                 )
        if self.type == "final_answer":
            if self.answer is None or self.tool_call is not None or not self.answer.strip():
                raise ValueError("A final_answer decision must contain answer"
                                 "And must not contain tool call"
                                 )

        return self

class AgentRunResult(BaseModel):
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    tool_calls_made: int = Field(ge=0)
    final_answer: str | None = None
    status: Literal["completed", "max_calls_reached", "failed"]

        


    




