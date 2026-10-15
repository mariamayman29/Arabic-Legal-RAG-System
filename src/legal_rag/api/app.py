from dotenv import load_dotenv

load_dotenv()
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from prometheus_client import Counter, make_asgi_app
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from legal_rag.api.schemas import AskRequest, AskResponse
from legal_rag.retrieval.retriever import COLLECTION_NAME, retrieve_legal_context
from legal_rag.utils.config import settings
from legal_rag.utils.helper import detect_language, extract_article_numbers
from legal_rag.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


INPUT_TOKENS = Counter("llm_input_tokens_total", "Total input tokens")
OUTPUT_TOKENS = Counter("llm_output_tokens_total", "Total output tokens")

FALLBACK_MESSAGE_AR = "أعتذر، لا يمكنني الإجابة على هذا السؤال لأنه يقع خارج النطاق المتاح في نصوص القانون المدني المصري المرفقة."
FALLBACK_MESSAGE_EN = "I apologize, I cannot answer this question because it falls outside the scope available in the provided texts of the Egyptian Civil Code."

PROMPT_AR = """أنت مستشار قانوني متخصص حصرياً في القانون المدني المصري. مهمتك الإجابة بدقة استناداً فقط إلى المواد القانونية المرفقة أدناه.
القيود الإلزامية:
1. النطاق والموضوع: يُمنع منعاً باتاً الإجابة عن أي سؤال خارج نطاق القانون المدني المصري
2. المصدر والتأصيل: استخرج إجابتك حصرياً من المواد المرفقة، واذكر أرقام المواد صراحة في إجابتك. يُمنع تماماً الاعتماد على أي معلومات مسبقة خارج السياق
3. لغة الإجابة: أجب باللغة العربية بالكامل
4. المواد الملغاة: إذا تضمن السياق مادة ملغاة، يجب التنبيه بوضوح وصراحة على أن المادة (ملغاة ولا يجوز الاستناد إليها كحكم سارٍ حالياً) وبيان حكم إلغائها بدلاً من استخدام بروتوكول الرفض
5. بروتوكول الرفض: إذا كان السؤال خارج النطاق أو لم تتضمن المواد المرفقة ما يكفي للإجابة، يجب أن ترد حصرياً بهذه العبارة دون أي زيادة:
"أعتذر، لا يمكنني الإجابة على هذا السؤال لأنه يقع خارج النطاق المتاح في نصوص القانون المدني المصري المرفقة" """

PROMPT_EN = """You are a legal advisor specializing exclusively in the Egyptian Civil Code. Your task is to answer accurately based solely on the legal articles provided below.

Mandatory Constraints:
1. Scope and Subject: You are strictly forbidden from answering any question outside the scope of the Egyptian Civil Code.
2. Source and Attribution: Extract your answer exclusively from the provided articles, and explicitly cite the article numbers in your response. Relying on any prior knowledge outside the provided context is strictly prohibited.
3. Response Language: Answer entirely in English.
4. Repealed Articles: If the context includes a repealed article, you must clearly and explicitly state that the article is "(repealed and cannot be relied upon as a currently valid ruling)". Do NOT use the refusal protocol in this case.
5. Refusal Protocol: If the question is out of scope or the provided articles do not contain sufficient information to answer, you must respond exclusively with this exact phrase, without any additions:
"I apologize, I cannot answer this question because it falls outside the scope available in the provided texts of the Egyptian Civil Code." """

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_db")

def redact_pii(text: str) -> str:
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED EMAIL]", text
    )
    text = re.sub(r"(?:\+20|0)?1[0125]\d{8}", "[REDACTED PHONE]", text)
    return text


rag_state = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing API services")
    start_time = time.time()
    try:
        rag_state["retriever"] = SentenceTransformer("BAAI/bge-m3", device="cpu")
        rag_state["qdrant"] = QdrantClient(path=QDRANT_PATH)
        rag_state["generator"] = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
        rag_state["langfuse"] = Langfuse()
        rag_state["baseline_emb"] = rag_state["retriever"].encode(
            "ما هي شروط صحة العقد في القانون المدني؟"
        )
        logger.info(
            f"API services initialized in {time.time() - start_time:.2f} seconds"
        )
    except Exception as e:
        logger.critical(f"Failed to initialize API services: {e}", exc_info=True)
        raise RuntimeError(f"Startup failed: {e}") from e

    yield

    logger.info("Shutting down API services")
    if rag_state.get("langfuse"):
        rag_state["langfuse"].flush()
    if rag_state.get("qdrant"):
        rag_state["qdrant"].close()


app = FastAPI(title="Egyptian Civil Code API", lifespan=lifespan)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health_check():
    try:
        count_res = rag_state["qdrant"].count(
            collection_name=COLLECTION_NAME, exact=True
        )
        return {"status": "healthy", "documents_indexed": count_res.count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e!s}") from e


def get_production_prompt(lang: str) -> ChatPromptTemplate:
    system_msg = PROMPT_AR if lang == "ar" else PROMPT_EN
    user_msg = (
        "المواد القانونية المسترجعة:\n{context}\n\nالسؤال:\n{question}"
        if lang == "ar"
        else "Retrieved Legal Articles:\n{context}\n\nQuestion:\n{question}"
    )
    return ChatPromptTemplate.from_messages(
        [("system", system_msg), ("user", user_msg)]
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    req_start_time = time.time()

    query_emb = rag_state["retriever"].encode(request.question)
    baseline = rag_state["baseline_emb"]
    cosine_sim = np.dot(baseline, query_emb) / (
        np.linalg.norm(baseline) * np.linalg.norm(query_emb)
    )
    logger.info(f"Data Drift Alert - Cosine similarity to baseline: {cosine_sim:.4f}")

    lang = detect_language(request.question)
    explicit_articles = extract_article_numbers(request.question)

    try:
        contexts, sources, _semantic_scores = retrieve_legal_context(
            qdrant_client=rag_state["qdrant"],
            retriever=rag_state["retriever"],
            question=request.question,
            lang=lang,
            explicit_articles=explicit_articles,
        )
        qdrant_error = None
    except Exception as e: # noqa: BLE001
        logger.error(f"Vector retrieval failed: {e}")
        contexts, sources = [], []
        qdrant_error = e

    if not contexts and not qdrant_error:
        fallback_msg = FALLBACK_MESSAGE_AR if lang == "ar" else FALLBACK_MESSAGE_EN
        return AskResponse(answer=fallback_msg, sources=[])

    if qdrant_error:
        raise HTTPException(status_code=500, detail="Database retrieval failed")

    prompt = get_production_prompt(lang=lang)
    chain = prompt | rag_state["generator"]
    context_str = "\n\n---\n\n".join(contexts)

    handler = CallbackHandler()
    trace_config = {
        "callbacks": [handler],
        "run_name": "legal-rag-ask",
        "metadata": {"langfuse_tags": ["legal_rag_api"]},
    }

    if request.stream:

        async def token_generator():
            try:
                async for chunk in chain.astream(
                    {"context": context_str, "question": request.question},
                    config=trace_config,
                ):
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        INPUT_TOKENS.inc(chunk.usage_metadata.get("input_tokens", 0))
                        OUTPUT_TOKENS.inc(chunk.usage_metadata.get("output_tokens", 0))
                    yield chunk.content
            except Exception as e: # noqa: BLE001
                logger.error(f"Streaming failed: {e}")
                yield "\n[Error: Upstream language model failed]"
            finally:
                rag_state["langfuse"].flush()
                logger.info(f"Langfuse Trace ID: {handler.last_trace_id}")

        return StreamingResponse(token_generator(), media_type="text/plain")

    try:
        res = await chain.ainvoke(
            {"context": context_str, "question": request.question},
            config=trace_config,
        )
        if hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
            usage = res.response_metadata["token_usage"]
            INPUT_TOKENS.inc(usage.get("prompt_tokens", 0))
            OUTPUT_TOKENS.inc(usage.get("completion_tokens", 0))

        logger.info(f"JSON request completed in {time.time() - req_start_time:.2f}s")
        logger.info(f"Langfuse Trace ID: {handler.last_trace_id}")

        is_llm_refusal = res.content.strip().startswith(("I apologize", "أعتذر"))
        final_sources = [] if is_llm_refusal else list(sources)

        rag_state["langfuse"].flush()

        safe_answer = redact_pii(res.content)
        return AskResponse(answer=safe_answer, sources=final_sources)

    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(
            status_code=502, detail="Failed to generate an answer"
        ) from e
