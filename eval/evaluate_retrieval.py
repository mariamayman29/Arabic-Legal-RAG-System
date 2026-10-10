import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import json
import logging

import mlflow
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

configure_logging(log_path=settings.EVAL_LOG_PATH)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = settings.GOLDEN_DATASET_PATH
QDRANT_PATH = settings.QDRANT_PATH
TOP_K = 3

EXPERIMENTS = {
    "baseline_small_recursive": {
        "size": 150,
        "overlap": 30,
        "model": "intfloat/multilingual-e5-small",
        "use_header": False,
    },
    "baseline_medium_recursive": {
        "size": 300,
        "overlap": 50,
        "model": "intfloat/multilingual-e5-small",
        "use_header": False,
    },
    "baseline_large_recursive": {
        "size": 500,
        "overlap": 100,
        "model": "intfloat/multilingual-e5-small",
        "use_header": False,
    },
    "article_level_raw": {
        "size": None,
        "overlap": None,
        "model": "intfloat/multilingual-e5-small",
        "use_header": False,
    },
    "article_level_enriched": {
        "size": None,
        "overlap": None,
        "model": "intfloat/multilingual-e5-small",
        "use_header": True,
    },
    "article_level_enriched_bgem3": {
        "size": None,
        "overlap": None,
        "model": "BAAI/bge-m3",
        "use_header": True,
    },
}


def get_target_articles(item: dict) -> set[int]:
    meta = item.get("metadata", {})
    if "target_articles" in meta:
        return set(meta["target_articles"])
    if "article_number" in meta:
        return {meta["article_number"]}
    return set()


def evaluate_query(
    retrieved_articles: list[int], target_articles: set[int], k: int
) -> dict:
    if not target_articles:
        return {"hit": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}

    unique_retrieved = list(dict.fromkeys(retrieved_articles[:k]))
    matched = set(unique_retrieved).intersection(target_articles)

    hit = 1.0 if len(matched) > 0 else 0.0

    recall = len(matched) / len(target_articles)

    precision = len(matched) / k

    mrr = 0.0
    for rank, art_id in enumerate(retrieved_articles[:k], start=1):
        if art_id in target_articles:
            mrr = 1.0 / rank
            break

    return {"hit": hit, "recall": recall, "precision": precision, "mrr": mrr}


def run_retrieval_benchmarks():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    client = QdrantClient(path=str(QDRANT_PATH))
    mlflow.set_experiment("Legal_RAG_Retrieval_Benchmark")

    current_model_name = None
    embed_model = None

    for exp_name, exp_config in EXPERIMENTS.items():
        if not client.collection_exists(exp_name):
            logger.info(f"Collection {exp_name} not found in {QDRANT_PATH}. Skipping.")
            continue

        logger.info(f"Evaluating Retrieval: {exp_name}")

        if current_model_name != exp_config["model"]:
            logger.info(f"Loading embedding model: {exp_config['model']}")
            embed_model = SentenceTransformer(exp_config["model"])
            current_model_name = exp_config["model"]

        results = []

        for item in tqdm(golden_data, desc=f"Querying {exp_name}"):
            q_text = item["question"]
            lang = item["metadata"]["language"]
            target_arts = get_target_articles(item)

            query_text = (
                f"query: {q_text}" if "e5" in exp_config["model"].lower() else q_text
            )
            query_vector = embed_model.encode(
                query_text, normalize_embeddings=True
            ).tolist()

            lang_filter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=lang))]
            )

            search_result = client.query_points(
                collection_name=exp_name,
                query=query_vector,
                query_filter=lang_filter,
                limit=TOP_K,
            )
            hits = search_result.points
            retrieved_articles = [hit.payload["article_number"] for hit in hits]
            metrics = evaluate_query(retrieved_articles, target_arts, TOP_K)

            results.append(
                {
                    "question": q_text,
                    "language": lang,
                    "query_type": item["metadata"].get("query_type", "unknown"),
                    "target_articles": list(target_arts),
                    "retrieved_articles": retrieved_articles,
                    **metrics,
                }
            )

        df_results = pd.DataFrame(results)

        with mlflow.start_run(run_name=exp_name):
            mlflow.log_param("strategy", exp_name)
            mlflow.log_param("chunk_size", exp_config["size"] or "article_level")
            mlflow.log_param("overlap", exp_config["overlap"] or 0)
            mlflow.log_param("embedding_model", exp_config["model"])
            mlflow.log_param("use_header", exp_config["use_header"])
            mlflow.log_param("top_k", TOP_K)

            # Overall Metrics
            mlflow.log_metric("hit_rate_overall", df_results["hit"].mean())
            mlflow.log_metric("recall_overall", df_results["recall"].mean())
            mlflow.log_metric("precision_overall", df_results["precision"].mean())
            mlflow.log_metric("mrr_overall", df_results["mrr"].mean())

            for lang in ["ar", "en"]:
                df_lang = df_results[df_results["language"] == lang]
                if not df_lang.empty:
                    mlflow.log_metric(f"hit_rate_{lang}", df_lang["hit"].mean())
                    mlflow.log_metric(f"recall_{lang}", df_lang["recall"].mean())
                    mlflow.log_metric(f"precision_{lang}", df_lang["precision"].mean())
                    mlflow.log_metric(f"mrr_{lang}", df_lang["mrr"].mean())

            run_dir = PROJECT_ROOT / "eval" / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_file = run_dir / f"{exp_name}_retrieval_metrics.csv"
            df_results.to_csv(artifact_file, index=False)
            mlflow.log_artifact(str(artifact_file))

            logger.info(
                f"[{exp_name}] Hit@{TOP_K}: {df_results['hit'].mean():.3f} | "
                f"Recall@{TOP_K}: {df_results['recall'].mean():.3f} | "
                f"MRR@{TOP_K}: {df_results['mrr'].mean():.3f}"
            )
    client.close()
    logger.info("Retrieval benchmarks completed")


if __name__ == "__main__":
    run_retrieval_benchmarks()
