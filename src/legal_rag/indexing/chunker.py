import argparse
import json
import logging
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from legal_rag.schemas import ChunkMetadata, ChunkSchema
from legal_rag.utils.config import settings
from legal_rag.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

CORPUS_PATH = settings.PROCESSED_CORPUS_PATH
QDRANT_PATH = settings.QDRANT_PATH


def build_context_header(article: dict) -> str:
    parts = [
        article.get("book"),
        article.get("chapter"),
        article.get("section"),
        article.get("topic"),
        article.get("subtopic"),
    ]
    valid_parts = [p for p in parts if p]
    if not valid_parts:
        return f"Article {article['article_number']}"
    return f"Article {article['article_number']} - " + " > ".join(valid_parts)


def generate_chunks(
    corpus: list[dict],
    strategy: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_header: bool = False,
) -> list[ChunkSchema]:
    chunks = []

    if strategy.startswith("article_level"):
        for art in corpus:
            header = build_context_header(art) if use_header else ""
            for lang, text_key in [("en", "en_text"), ("ar", "ar_text")]:
                chunks.append(
                    ChunkSchema(
                        chunk_id=f"art_{art['article_number']}_{lang}_0",
                        content=art[text_key],
                        context_header=header,
                        metadata=ChunkMetadata(
                            article_number=art["article_number"],
                            language=lang,
                            strategy=strategy,
                            chunk_index=0,
                            total_chunks=1,
                            source_page=art["source_page"],
                            is_repealed=art["is_repealed"],
                            citation=art["citation"],
                        ),
                    )
                )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "،", " ", ""],
        )
        for art in corpus:
            header = build_context_header(art) if use_header else ""
            for lang, text_key in [("en", "en_text"), ("ar", "ar_text")]:
                splits = splitter.split_text(art[text_key])
                for idx, text in enumerate(splits):
                    chunks.append(
                        ChunkSchema(
                            chunk_id=f"art_{art['article_number']}_{lang}_{idx}_rec",
                            content=text,
                            context_header=header,
                            metadata=ChunkMetadata(
                                article_number=art["article_number"],
                                language=lang,
                                strategy=strategy,
                                chunk_index=idx,
                                total_chunks=len(splits),
                                source_page=art["source_page"],
                                is_repealed=art["is_repealed"],
                                citation=art["citation"],
                            ),
                        )
                    )
    return chunks


def index_to_qdrant(
    collection_name: str,
    chunks: list[ChunkSchema],
    model: SentenceTransformer,
    model_name: str,
    client: QdrantClient,
):
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection {collection_name} exists. Recreating it")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=model.get_embedding_dimension(), distance=Distance.COSINE
        ),
    )

    batch_size = 128
    for i in tqdm(
        range(0, len(chunks), batch_size), desc=f"Indexing {collection_name}"
    ):
        batch = chunks[i : i + batch_size]
        texts = []

        for chunk in batch:
            core_text = (
                f"{chunk.context_header}\n{chunk.content}".strip()
                if chunk.context_header
                else chunk.content
            )
            if "e5" in model_name.lower():
                texts.append(f"passage: {core_text}")
            else:
                texts.append(core_text)

        embeddings = model.encode(texts, normalize_embeddings=True)

        points = []
        for chunk, embedding in zip(batch, embeddings):
            point_id = chunk.point_id
            payload = chunk.to_qdrant_payload()
            points.append(
                PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)
            )

        client.upsert(collection_name=collection_name, points=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Name of the experiment strategy to run",
    )
    args = parser.parse_args()

    raw_data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    client = QdrantClient(path=str(QDRANT_PATH))

    experiments = {
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

    if args.strategy not in experiments:
        logger.error(f"Strategy {args.strategy} not found.")
        sys.exit(1)

    exp = experiments[args.strategy]
    logger.info(
        f"Loading embedding model: {exp['model']} for strategy: {args.strategy}"
    )
    model = SentenceTransformer(exp["model"])

    chunks = generate_chunks(
        raw_data, args.strategy, exp["size"], exp["overlap"], exp["use_header"]
    )
    index_to_qdrant(args.strategy, chunks, model, exp["model"], client)
    client.close()
