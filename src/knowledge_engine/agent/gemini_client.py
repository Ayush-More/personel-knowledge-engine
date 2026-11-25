import os

from google import genai
from google.genai import types

from knowledge_engine.models import AgentDecision

class GeminiAgentClient:
    def __init__(self , model: str = "gemini-3.5-flash-lite" , api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise ValueError("Gemini API Key is not configured")

        self.client = genai.Client(api_key=key)
        self.model = model

    def decide(self , prompt: str) -> AgentDecision:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.client.models.generate_content(
            model = self.model,
            contents = prompt,
            config= types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AgentDecision
            )
        )

        if response is None or response.parsed is None:
            raise ValueError("Gemini returned no structured decision")

        return response.parsed

