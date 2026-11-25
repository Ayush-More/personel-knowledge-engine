import os
from google import genai
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, Field

from knowledge_engine.context_builder import build_context
from knowledge_engine.models import DocumentChunk , SourceCitation , GeneratedAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

class GeminiDraft(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

class GeminiAnswerGenerator:
    def __init__(self , model : str = "gemini-3.5-flash-lite" , api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API key is not configured")

        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self , question: str , result: list[tuple[DocumentChunk , float]])->GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not result:
            return GeneratedAnswer(
                question=question,
                answer="I do not have the enough evidence to answer",
                citations=[],
                insufficient_evidence=True
            )

        context = build_context(result)

        prompt = f"""
You are a careful assistant answering questions about a user notes.

Answer the question using ONLY the supplied context

Rules:
1. Do not use outside the knowledge or invent fact
2. if the context does not provide the enough evidence, say no clealy
3. Set insufficient_evidence to true when the notes are insufficient.
4. Cite only chunk ids that appear in the supplied context.
5. Include only chunk Ids that support your context
6. Keep the answer clear and concise.

return:
- answer: your answer to the question
- cited_chunk_ids: list of supporting chunk IDs
- insuffient_evidence: whether the notes lack enough evidence

CONTEXT
{context}

QUESTION
{question}

"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema= GeminiDraft,
            ),
        )

        if response is None:
            raise ValueError("Gemini returned no structured answer")

        draft = response.parsed

        available_chunks = {
            chunk.chunk_id: chunk for chunk , score in result
        }

        valid_ids = list(dict.fromkeys(
            chunk_id for chunk_id in draft.cited_chunk_ids if chunk_id in available_chunks
        ))

        citations = [
            SourceCitation(
                chunk_id=chunk_id,
                text = available_chunks[chunk_id].content,
            )
            for chunk_id in valid_ids
        ]

        insufficient = draft.insufficient_evidence

        if insufficient:
            citations = []

        return GeneratedAnswer(
            question=question,
            answer=draft.answer,
            citations=citations,
            insufficient_evidence=insufficient
        )

    def close(self) -> None:
        self.client.close()


    

