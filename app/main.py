
import json
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.llm import generate_answer
from app.config import APP_ENV

from app.observability import (
    HTTP_REQUESTS,
    HTTP_LATENCY,
    LLM_REQUESTS,
    LLM_ERRORS,
    INPUT_TOKENS,
    OUTPUT_TOKENS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger("llmops")

app = FastAPI(
    title="LLMOps Production POC",
    description="LLM application with Azure OpenAI, MLflow and observability",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000
    )


class ChatResponse(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float


@app.get("/")
def home():
    return {
        "message": "LLMOps Support Assistant is running",
        "environment": APP_ENV
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "LLMOps"
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    start_time = time.perf_counter()
    status = "200"

    try:
        LLM_REQUESTS.inc()

        result = generate_answer(request.question)

        INPUT_TOKENS.inc(result["input_tokens"])
        OUTPUT_TOKENS.inc(result["output_tokens"])

        return {
            "answer": result["answer"],
            "model": result["model"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "latency_seconds": result["latency_seconds"]
        }

    except Exception as error:

        status = "500"
        LLM_ERRORS.inc()

        logger.exception(
            json.dumps({
                "event": "llm_request_failed",
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
        )

        raise HTTPException(
            status_code=500,
            detail="LLM request failed. Check application logs."
        ) from error

    finally:

        duration = time.perf_counter() - start_time

        HTTP_REQUESTS.labels(
            method="POST",
            endpoint="/chat",
            status=status
        ).inc()

        HTTP_LATENCY.observe(duration)

        logger.info(
            json.dumps({
                "event": "chat_request_completed",
                "status": status,
                "latency_seconds": round(duration, 4)
            })
        )
