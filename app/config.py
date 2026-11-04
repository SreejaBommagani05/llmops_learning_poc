
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT", ""
)


AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_NAME", ""
)

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "LLMOps-Support-Assistant"
)

# Application Configuration
APP_ENV = os.getenv("APP_ENV", "development")

# System Prompt Configuration
PROMPT_PATH = ROOT_DIR / "prompts" / "system_prompt.txt"

if not PROMPT_PATH.exists():
    raise FileNotFoundError(
        f"System prompt not found: {PROMPT_PATH}"
    )


def load_system_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")


# Validate Azure Configuration
def validate_azure_config():
    required = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    }

    missing = [
        key for key, value in required.items() if not value
    ]

    if missing:
        raise ValueError(
            f"Missing Azure configuration: {', '.join(missing)}"
        )
