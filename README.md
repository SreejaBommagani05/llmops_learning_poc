# llmops_learning_poc
LLMOps POC with MLflow, LLM evaluation, observability, monitoring and GitHub Actions CI/CD.
# LLMOps Production POC

An end-to-end LLMOps project implementing
LLM integration, experiment tracking, evaluation,
observability, automated testing and CI/CD.

## Technologies

- Python
- FastAPI
- OpenAI API
- MLflow
- Prometheus
- Pytest
- Ruff
- GitHub Actions
- Azure App Service

## Features

1. LLM-powered customer support assistant
2. External prompt management
3. MLflow experiment tracking
4. Dataset-based LLM evaluation
5. Token usage monitoring
6. Request latency monitoring
7. Structured application logging
8. Prometheus metrics endpoint
9. Automated unit and API testing
10. GitHub Actions CI/CD
11. Azure deployment without Docker

## Run Locally

Create a virtual environment:

python -m venv .venv

Activate:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Configure .env with your API key.

Run the application:

uvicorn app.main:app --reload

## API Documentation

http://127.0.0.1:8000/docs

## Health

http://127.0.0.1:8000/health

## Metrics

http://127.0.0.1:8000/metrics

## MLflow

Run evaluation:

python -m evaluation.evaluate

Start MLflow UI:

mlflow ui --backend-store-uri sqlite:///mlflow.db

Open:

http://127.0.0.1:5000

## Testing

python -m pytest -v

## CI/CD

GitHub Actions automatically executes
code quality checks and tests on pushes
and pull requests to main.

Successful pushes to main trigger
Azure App Service deployment.