import logging
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

configure_logging(log_path=settings.LOG_PATH)
logger = logging.getLogger(__name__)


class RAGConfigModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return "Egyptian Civil Code Retriever Config"


def register_champion():
    client = MlflowClient()
    mlflow.set_experiment("Legal_RAG_Retrieval_Benchmark")

    with mlflow.start_run(run_name="Champion_Config_Registration"):
        mlflow.log_param("strategy", "article_level_enriched_bgem3")
        mlflow.log_param("embedding_model", "BAAI/bge-m3")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="chunking_config",
            python_model=RAGConfigModel(),
            registered_model_name="Egyptian_Civil_Code_Retriever",
        )

    client.set_registered_model_alias(
        name="Egyptian_Civil_Code_Retriever",
        alias="champion",
        version=model_info.registered_model_version,
    )

    logger.info(
        f"Success: Registered version {model_info.registered_model_version} of 'Egyptian_Civil_Code_Retriever' as the Champion"
    )


if __name__ == "__main__":
    register_champion()
