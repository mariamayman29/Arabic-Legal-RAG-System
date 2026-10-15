import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
if "LANGFUSE_HOST" in os.environ and "LANGFUSE_BASE_URL" not in os.environ:
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"].rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import types

import mlflow
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse import Langfuse

dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def trigger_alert(metric_name: str, score: float, threshold: float):
    logger.critical(
        f"ALERT: {metric_name} score dropped to {score:.2f} (below threshold of {threshold})"
    )


def run_production_evaluation():
    langfuse = Langfuse()

    logger.info("Fetching recent production traces from Langfuse")
    traces_response = langfuse.fetch_traces(tags=["legal_rag_api"], limit=50)
    if not traces_response.data:
        logger.info("No new traces found. Exiting")
        return

    eval_samples = {"question": [], "contexts": [], "answer": [], "trace_id": []}

    for trace in traces_response.data:
        question, context_str, answer = "", "", ""

        observations = []
        try:
            if hasattr(langfuse, "get_trace"):
                observations = langfuse.get_trace(trace.id).observations or []
            elif hasattr(langfuse, "client"):
                observations = langfuse.client.trace.get(trace.id).observations or []
            else:
                observations = getattr(trace, "observations", [])
        except Exception: # noqa: BLE001, S112
            continue

        for obs in observations:
            if obs.input and isinstance(obs.input, dict):
                question = obs.input.get("question", question)

                ctx = obs.input.get("context", context_str)
                if isinstance(ctx, list):
                    context_str = "\n\n---\n\n".join([str(c) for c in ctx])
                elif isinstance(ctx, str):
                    context_str = ctx

            if obs.output:
                if isinstance(obs.output, str):
                    answer = obs.output
                elif isinstance(obs.output, dict):
                    answer = obs.output.get(
                        "output",
                        obs.output.get("answer", obs.output.get("content", answer)),
                    )

        if question and context_str and answer:
            eval_samples["question"].append(question)
            eval_samples["contexts"].append(context_str.split("\n\n---\n\n"))
            eval_samples["answer"].append(answer)
            eval_samples["trace_id"].append(trace.id)

    if not eval_samples["question"]:
        logger.info("No parsable trace data found")
        return

    logger.info(f"Evaluating {len(eval_samples['question'])} traces...")
    dataset = Dataset.from_dict(
        {
            "question": eval_samples["question"],
            "contexts": eval_samples["contexts"],
            "answer": eval_samples["answer"],
            "reference": eval_samples["answer"],
        }
    )

    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]
    scores = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=generator_llm,
        embeddings=ragas_embeddings,
    )

    df_results = scores.to_pandas()

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for idx, row in df_results.iterrows():
        for metric in metric_names:
            if not pd.isna(row.get(metric)):
                langfuse.score(
                    trace_id=eval_samples["trace_id"][idx],
                    name=metric,
                    value=row[metric],
                    comment="Automated production RAGAS evaluation",
                )

    langfuse.flush()
    logger.info("All RAGAS metric scores successfully attached to Langfuse traces")

    mlflow.set_experiment("Legal_RAG_Production_Monitoring")
    with mlflow.start_run(run_name="production_batch_eval"):
        mlflow.log_param("batch_size", len(eval_samples["question"]))

        for metric in metric_names:
            if metric in df_results.columns:
                avg_score = df_results[metric].mean()
                mlflow.log_metric(f"production_{metric}_avg", avg_score)
                logger.info(
                    f"Average Production {metric.capitalize()}: {avg_score:.3f}"
                )

                if avg_score < 0.80:
                    trigger_alert(metric.capitalize(), avg_score, 0.80)


if __name__ == "__main__":
    run_production_evaluation()
