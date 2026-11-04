from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def mock_llm(question):

    return {
        "answer": "You can reset your password using account settings.",
        "model": "test-model",
        "input_tokens": 10,
        "output_tokens": 15,
        "total_tokens": 25,
        "latency_seconds": 0.25
    }


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert "LLMOps" in response.json()["message"]


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_chat(monkeypatch):

    monkeypatch.setattr(
        "app.main.generate_answer",
        mock_llm
    )

    response = client.post(
        "/chat",
        json={
            "question": "How do I reset my password?"
        }
    )

    assert response.status_code == 200

    assert response.json()["model"] == "test-model"

    assert response.json()["total_tokens"] == 25


def test_chat_invalid_input():

    response = client.post(
        "/chat",
        json={
            "question": "Hi"
        }
    )

    assert response.status_code == 422


def test_metrics():

    response = client.get("/metrics")

    assert response.status_code == 200

    assert "http_requests_total" in response.text