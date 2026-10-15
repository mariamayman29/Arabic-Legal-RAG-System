import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "article_level_enriched_bgem3"
SIMILARITY_THRESHOLD = 0.5


def retrieve_legal_context(
    qdrant_client: QdrantClient,
    retriever: SentenceTransformer,
    question: str,
    lang: str,
    explicit_articles: list[int],
) -> tuple[list[str], list[str], list[float]]:

    q_vec = retriever.encode(question, normalize_embeddings=True).tolist()
    lang_condition = FieldCondition(key="language", match=MatchValue(value=lang))

    semantic_hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        query_filter=Filter(must=[lang_condition]),
        limit=4,
    ).points

    explicit_hits = []
    if explicit_articles:
        logger.info(f"Explicit article numbers detected: {explicit_articles}")
        explicit_hits = qdrant_client.query_points(
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

    semantic_scores = [round(h.score, 4) for h in semantic_hits]
    return contexts, list(sources), semantic_scores
