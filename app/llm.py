
import ssl
import time
import httpx
import truststore

from openai import OpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    load_system_prompt
)


# Use Windows trusted certificates
ssl_context = truststore.SSLContext(
    ssl.PROTOCOL_TLS_CLIENT
)

http_client = httpx.Client(
    verify=ssl_context,
    timeout=30.0
)

# Azure Foundry OpenAI-compatible client
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT.rstrip("/") + "/",
    http_client=http_client
)


def generate_answer(question: str):

    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure API key is missing")

    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError("Azure endpoint is missing")

    if not AZURE_OPENAI_DEPLOYMENT_NAME:
        raise RuntimeError("Azure deployment name is missing")

    system_prompt = load_system_prompt()

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    latency = time.perf_counter() - start_time

    answer = response.choices[0].message.content or ""

    usage = response.usage

    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return {
        "answer": answer,
        "model": AZURE_OPENAI_DEPLOYMENT_NAME,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_seconds": round(latency, 4)
    }
