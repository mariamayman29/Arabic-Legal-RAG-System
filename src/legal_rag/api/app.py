import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue , MatchAny
from sentence_transformers import SentenceTransformer

from legal_rag.schemas import AskRequest, AskResponse
from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

COLLECTION_NAME = "article_level_enriched_bgem3"
SIMILARITY_THRESHOLD = 0.5

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




def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    return "en"

def extract_article_numbers(text: str) -> list[int]:
    matches = re.findall(
        r"(?:\barticle|\bart\.?|المادة|مادة|الماده|ماده)\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    return list({int(m) for m in matches})


rag_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing API services")
    start_time = time.time()

    try:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing from environment/config")

        os.environ["HF_HUB_OFFLINE"] = "1"
        rag_state["retriever"] = SentenceTransformer("BAAI/bge-m3", device="cpu")
        rag_state["qdrant"] = QdrantClient(path=str(settings.QDRANT_PATH))
        rag_state["generator"] = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
        logger.info(f"API services initialized in {time.time() - start_time:.2f} seconds")
    except (ValueError, OSError, RuntimeError) as e:
        logger.critical(f"Failed to initialize API services during startup: {e}", exc_info=True)
        raise RuntimeError(f"Startup initialization failed: {e}") from e

    try:
        yield
    finally:
        logger.info("Shutting down API services")
        if rag_state.get("qdrant") is not None:
            rag_state["qdrant"].close()


app = FastAPI(title="Egyptian Civil Code API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    try:
        qdrant_client = rag_state["qdrant"]
        _ = rag_state["retriever"]

        count_res = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )
        return {"status": "healthy", "documents_indexed": count_res.count}
    except (KeyError, ValueError, RuntimeError, UnexpectedResponse) as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}",
        ) from e


def get_production_prompt(lang: str) -> ChatPromptTemplate:
    if lang == "ar":
        return ChatPromptTemplate.from_messages([
            ("system", PROMPT_AR),
            ("user", "المواد القانونية المسترجعة:\n{context}\n\nالسؤال:\n{question}"),
        ])
    return ChatPromptTemplate.from_messages([
        ("system", PROMPT_EN),
        ("user", "Retrieved Legal Articles:\n{context}\n\nQuestion:\n{question}"),
    ])


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    req_start_time = time.time()
    logger.info(f"Received question: {request.question}")

    lang = detect_language(request.question)
    explicit_articles = extract_article_numbers(request.question)

    try:
        q_vec = rag_state["retriever"].encode(
            request.question, normalize_embeddings=True
        ).tolist()
        lang_condition = FieldCondition(key="language", match=MatchValue(value=lang))

        semantic_hits = rag_state["qdrant"].query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=Filter(must=[lang_condition]),
            limit=4,
        ).points

        explicit_hits = []
        if explicit_articles:
            logger.info(f"Explicit article numbers detected: {explicit_articles}")
            explicit_hits = rag_state["qdrant"].query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                query_filter=Filter(
                    must=[
                        lang_condition,
                        FieldCondition(
                            key="article_number",
                            match=MatchAny(any=explicit_articles),
                        ),
                    ]
                ),
                limit=len(explicit_articles),
            ).points
    except (KeyError, ValueError, RuntimeError, UnexpectedResponse) as e:
        logger.error(f"Vector retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve legal articles from the vector database",
        ) from e

    contexts = []
    sources = set()
    seen_ids = set()

    for hit in explicit_hits + semantic_hits:
        if hit.id in seen_ids:
            continue
        is_explicit = hit.payload.get("article_number") in explicit_articles

        if is_explicit or hit.score >= SIMILARITY_THRESHOLD:
            seen_ids.add(hit.id)
            content = hit.payload.get("content", "")
            header = hit.payload.get("context_header")
            if hit.payload.get("is_repealed", False):
                content = f"[NOTE: THIS ARTICLE IS REPEALED / هذه المادة ملغاة]\n{content}"

            source_name = (
                f"Egyptian Civil Code, {header}"
                if header
                else f"Article {hit.payload.get('article_number')}"
            )
            contexts.append(f"[{source_name}]\n{content}")
            sources.add(source_name)
            logger.debug(f"Retrieved: {source_name} (Score: {hit.score:.3f})")
        else:
            logger.warning(
                f"Chunk rejected. Score {hit.score:.3f} below threshold {SIMILARITY_THRESHOLD}"
            )

    if not contexts:
        logger.info("No context met the similarity threshold. Returning fallback refusal")
        fallback_msg = FALLBACK_MESSAGE_AR if lang == "ar" else FALLBACK_MESSAGE_EN
        return AskResponse(answer=fallback_msg, sources=[])

    prompt = get_production_prompt(lang=lang)
    chain = prompt | rag_state["generator"]
    context_str = "\n\n---\n\n".join(contexts)

    if request.stream:
        async def token_generator():
            try:
                async for chunk in chain.astream(
                    {"context": context_str, "question": request.question}
                ):
                    yield chunk.content
                yield "\n\n[Sources]:\n"
                for s in list(sources):
                    yield f"- {s}\n"
                logger.info(
                    f"Streaming request completed in {time.time() - req_start_time:.2f}s"
                )
            except RateLimitError as e:
                logger.warning(f"OpenAI rate limit reached during stream: {e}")
                yield "\n[Error: Rate limit exceeded. Please retry shortly.]"
            except (APITimeoutError, APIConnectionError, APIError) as e:
                logger.error(f"Streaming LLM generation failed: {e}", exc_info=True)
                yield "\n[Error: Upstream language model failed while streaming response.]"

        return StreamingResponse(token_generator(), media_type="text/plain")

    try:
        res = await chain.ainvoke({"context": context_str, "question": request.question})
        logger.info(f"JSON request completed in {time.time() - req_start_time:.2f}s")
        is_llm_refusal = res.content.strip().startswith(("I apologize", "أعتذر"))
        final_sources = [] if is_llm_refusal else list(sources)
        return AskResponse(answer=res.content, sources=final_sources)
    except RateLimitError as e:
        logger.warning(f"OpenAI rate limit exceeded: {e}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded with the LLM provider. Please try again",
        ) from e
    except (APITimeoutError, APIConnectionError) as e:
        logger.error(f"OpenAI connection or timeout error: {e}", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail="Upstream language model timed out or was unreachable",
        ) from e
    except APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer from the language model provider",
        ) from e