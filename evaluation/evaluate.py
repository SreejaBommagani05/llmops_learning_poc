
import json
import statistics
from pathlib import Path

import mlflow

from app.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    APP_ENV,
    load_system_prompt
)

from app.llm import generate_answer


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "evaluation" / "dataset.json"

RESULTS_PATH = ROOT_DIR / "evaluation" / "evaluation_results.json"


def run_evaluation():

    # Validate dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty")

    # MLflow configuration
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results = []

    latencies = []
    input_tokens = []
    output_tokens = []
    scores = []

    with mlflow.start_run(run_name="Azure-LLM-Evaluation"):

        # Log parameters
        mlflow.log_param(
            "model",
            AZURE_OPENAI_DEPLOYMENT_NAME
        )

        mlflow.log_param(
            "provider",
            "Azure OpenAI"
        )

        mlflow.log_param(
            "environment",
            APP_ENV
        )

        mlflow.log_param(
            "dataset_size",
            len(dataset)
        )

        mlflow.log_param(
            "prompt_version",
            "v1"
        )

        # Evaluate each question
        for index, item in enumerate(dataset):

            response = generate_answer(item["question"])

            answer = response["answer"].lower()

            expected = [
                keyword.lower()
                for keyword in item.get("expected_keywords", [])
            ]

            matched = sum(
                1 for keyword in expected
                if keyword in answer
            )

            score = (
                matched / len(expected)
                if expected else 0
            )

            latencies.append(response["latency_seconds"])

            input_tokens.append(response["input_tokens"])

            output_tokens.append(response["output_tokens"])

            scores.append(score)

            results.append({
                "question": item["question"],
                "answer": response["answer"],
                "keyword_coverage": score,
                "latency_seconds": response["latency_seconds"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "total_tokens": response["total_tokens"]
            })

            print(
                f"Evaluated {index + 1}/{len(dataset)}"
            )

        # Calculate evaluation metrics
        avg_latency = statistics.mean(latencies)

        avg_input_tokens = statistics.mean(input_tokens)

        avg_output_tokens = statistics.mean(output_tokens)

        avg_score = statistics.mean(scores)

        total_tokens = sum(
            item["total_tokens"] for item in results
        )

        # Log MLflow metrics
        mlflow.log_metric(
            "average_latency_seconds",
            avg_latency
        )

        mlflow.log_metric(
            "average_input_tokens",
            avg_input_tokens
        )

        mlflow.log_metric(
            "average_output_tokens",
            avg_output_tokens
        )

        mlflow.log_metric(
            "average_keyword_coverage",
            avg_score
        )

        mlflow.log_metric(
            "total_tokens",
            total_tokens
        )

        # Save evaluation results
        RESULTS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            RESULTS_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )

        # Log evaluation artifacts
        mlflow.log_artifact(str(DATASET_PATH))

        mlflow.log_artifact(str(RESULTS_PATH))

        mlflow.log_text(
            load_system_prompt(),
            "system_prompt.txt"
        )

        # Display summary
        print("\nAzure LLM Evaluation Completed")

        print("----------------------------")

        print("Provider: Azure OpenAI")

        print("Deployment:", AZURE_OPENAI_DEPLOYMENT_NAME)

        print("Dataset size:", len(dataset))

        print("Average latency:", round(avg_latency, 4))

        print("Average input tokens:", round(avg_input_tokens, 2))

        print("Average output tokens:", round(avg_output_tokens, 2))

        print("Total tokens:", total_tokens)

        print("Keyword coverage:", round(avg_score, 4))

        print("----------------------------")


if __name__ == "__main__":

    run_evaluation()
