from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration"
)

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests"
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "Total LLM errors"
)

INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Total input tokens"
)

OUTPUT_TOKENS = Counter(
    "llm_output_tokens_total",
    "Total output tokens"
)