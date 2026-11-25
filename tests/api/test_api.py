
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

from knowledge_engine.api.main import app


class FakeAnswerer:
    """Simulates the RAG answerer without calling Gemini."""

    def answer(self, question: str, top_k: int = 5):
        if "cloud region" in question.lower():
            return SimpleNamespace(
                question=question,
                answer="The available evidence does not specify the cloud region.",
                citations=[],
                insufficient_evidence=True,
            )

        return SimpleNamespace(
            question=question,
            answer="You used Postman for microservice API testing.",
            citations=[
                SimpleNamespace(
                    chunk_id="test-chunk-001",
                    text="Postman was used for microservice API testing.",
                )
            ],
            insufficient_evidence=False,
        )
class FailingAnswerer:
    """Simulates an unexpected internal RAG failure."""

    def answer(self, question: str, top_k: int = 5):
        raise RuntimeError("Simulated internal failure")


@pytest.fixture(scope="module")
def client():

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.answerer = FakeAnswerer()
        yield

    # Replace the real startup/shutdown process during API tests.
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_answerable_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "Which tool did I use for microservice API testing?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "Which tool did I use for microservice API testing?"
    assert "Postman" in data["answer"]
    assert data["insufficient_evidence"] is False
    assert len(data["citation"]) == 1
    assert data["citation"][0]["chunk_id"] == "test-chunk-001"


def test_ask_unanswerable_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "Which cloud region hosted the production services?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["insufficient_evidence"] is True
    assert data["citation"] == []


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/ask",
        json={
            "question": "Test question",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

def test_ask_rejects_whitespace_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 422

def test_ask_returns_generic_500_on_internal_error(client, monkeypatch):
    monkeypatch.setattr(
        app.state,
        "answerer",
        FailingAnswerer(),
    )

    response = client.post(
        "/ask",
        json={
            "question": "Which tool did I use?",
            "top_k": 5,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred while processing your question."
    }
    assert "Simulated internal failure" not in response.text
