import json
import logging
import os
import sys
import types
from pathlib import Path

import mlflow
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

os.environ["HF_HUB_OFFLINE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
configure_logging(log_path=settings.EVAL_LOG_PATH)
logger = logging.getLogger(__name__)

WINNING_STRATEGY = "article_level_enriched_bgem3"
EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3

PROMPT_CANDIDATES = {
    "prompt_a_direct": """أنت مستشار قانوني متخصص حصرياً في القانون المدني المصري. مهمتك الإجابة بدقة استناداً فقط إلى المواد القانونية المرفقة أدناه.
القيود الإلزامية:
1. النطاق والموضوع: يُمنع منعاً باتاً الإجابة عن أي سؤال خارج نطاق القانون المدني المصري
2. المصدر والتأصيل: استخرج إجابتك حصرياً من المواد المرفقة، واذكر أرقام المواد صراحة في إجابتك. يُمنع تماماً الاعتماد على أي معلومات مسبقة خارج السياق
3. لغة الإجابة: التزم بلغة السؤال بدقة؛ إذا كان السؤال بالإنجليزية أجب بالإنجليزية، وإذا كان بالعربية أجب بالعربية
4.المواد الملغاة: إذا تضمن السياق مادة ملغاة، يجب التنبيه بوضوح وصراحة على أن المادة (ملغاة ولا يجوز الاستناد إليها كحكم سارٍ حالياً)
5. بروتوكول الرفض: إذا كان السؤال خارج النطاق أو لم تتضمن المواد المرفقة ما يكفي للإجابة، يجب أن ترد حصرياً بهذه العبارة دون أي زيادة:
"أعتذر، لا يمكنني الإجابة على هذا السؤال لأنه يقع خارج النطاق المتاح في نصوص القانون المدني المصري المرفقة" """,
    "prompt_b_structured": """أنت مستشار قانوني متخصص حصرياً في القانون المدني المصري. مهمتك تقديم تحليل قانوني رصين استناداً فقط إلى المواد القانونية المرفقة أدناه.
القيود والمنهجية الإلزامية:
1. النطاق والموضوع: لا تجب على أي استفسار يقع خارج نطاق القانون المدني المصري
2. هيكل الإجابة: صغ إجابتك حصرياً وفق الأقسام التالية:
   - التكييف القانوني (Legal Issue)
   - النص الواجب التطبيق مع ذكر رقم المادة (Applicable Rule & Article Number)
   - التطبيق والخلاصة (Application & Conclusion)
3. لغة الإجابة: التزم بلغة السؤال (عربية أو إنجليزية).
4.المواد الملغاة: إذا تضمن السياق مادة ملغاة، يجب التنبيه بوضوح وصراحة على أن المادة (ملغاة ولا يجوز الاستناد إليها كحكم سارٍ حالياً)
5. بروتوكول الرفض: إذا كان السؤال خارج التخصص أو كانت المواد المرفقة غير كافية لحسم المسألة، يجب أن تقتصر إجابتك نصاً على هذه العبارة دون أي إضافة:
"أعتذر، لا يمكنني الإجابة على هذا السؤال لأنه يقع خارج النطاق المتاح في نصوص القانون المدني المصري المرفقة" """,
}


def generate_answer(
    llm: ChatOpenAI, system_prompt: str, question: str, contexts: list[str]
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "المواد القانونية المسترجعة:\n{context}\n\nالسؤال:\n{question}"),
        ]
    )
    chain = prompt | llm
    res = chain.invoke({"context": "\n\n---\n\n".join(contexts), "question": question})
    return res.content


def run_prompt_comparison_benchmark():
    logger.info("Loading golden dataset and setting up retriever...")
    with open(settings.GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    client = QdrantClient(path=str(settings.QDRANT_PATH))
    retrieval_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    logger.info(
        f"Retrieving contexts for {len(golden_data)} questions from {WINNING_STRATEGY}..."
    )
    cached_retrievals = []
    for item in tqdm(golden_data, desc="Retrieving from Qdrant"):
        q = item["question"]
        lang = item["metadata"]["language"]

        q_vec = retrieval_model.encode(q, normalize_embeddings=True).tolist()
        lang_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=lang))]
        )

        hits = client.query_points(
            collection_name=WINNING_STRATEGY,
            query=q_vec,
            query_filter=lang_filter,
            limit=TOP_K,
        ).points

        contexts = []
        for hit in hits:
            header = hit.payload.get("context_header", "")
            content = hit.payload.get("content", "")
            core_text = f"{header}\n{content}".strip() if header else content
            contexts.append(core_text)

        cached_retrievals.append(
            {
                "question": q,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
                "language": lang,
            }
        )

    client.close()

    mlflow.set_experiment("Legal_RAG_Prompt_Optimization")

    for prompt_name, system_prompt in PROMPT_CANDIDATES.items():
        logger.info(f"\n--- Benchmarking Candidate: {prompt_name} ---")

        eval_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": [],
            "language": [],
        }

        for sample in tqdm(cached_retrievals, desc=f"Generating with {prompt_name}"):
            ans = generate_answer(
                generator_llm, system_prompt, sample["question"], sample["contexts"]
            )
            eval_data["question"].append(sample["question"])
            eval_data["contexts"].append(sample["contexts"])
            eval_data["answer"].append(ans)
            eval_data["ground_truth"].append(sample["ground_truth"])
            eval_data["language"].append(sample["language"])

        logger.info(f"Running RAGAS evaluation for {prompt_name}...")
        dataset = Dataset.from_dict(eval_data)
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=generator_llm,
            embeddings=ragas_embeddings,
        )
        df_results = scores.to_pandas()
        df_results["language"] = eval_data["language"]

        with mlflow.start_run(run_name=prompt_name):
            mlflow.log_param("prompt_name", prompt_name)
            mlflow.log_param("system_prompt", system_prompt)
            mlflow.log_param("generator_model", "gpt-4o-mini")
            mlflow.log_param("temperature", 0.0)
            mlflow.log_param("top_k", TOP_K)

            metrics_to_log = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            for m in metrics_to_log:
                overall_score = df_results[m].mean()
                mlflow.log_metric(f"{m}_overall", overall_score)
                logger.info(f"[{prompt_name}] {m.title()}: {overall_score:.3f}")

            run_dir = PROJECT_ROOT / "eval" / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_file = run_dir / f"{prompt_name}_metrics.csv"
            df_results.to_csv(artifact_file, index=False)
            mlflow.log_artifact(str(artifact_file))

            for lang in ["ar", "en"]:
                df_lang = df_results[df_results["language"] == lang]
                if not df_lang.empty:
                    for m in metrics_to_log:
                        mlflow.log_metric(f"{m}_{lang}", df_lang[m].mean())

    logger.info("Prompt comparison benchmark completed")


if __name__ == "__main__":
    run_prompt_comparison_benchmark()
