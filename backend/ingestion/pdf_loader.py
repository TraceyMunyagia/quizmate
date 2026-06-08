from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from backend.core.config import Settings, get_settings
from backend.core.logger import get_logger


logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when an external ingestion service fails."""


def load_pdf(pdf_path: str | Path) -> list[Document]:
    path = Path(pdf_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    logger.info("Loading PDF pages from %s", path)
    return PyPDFLoader(str(path)).load()


def validate_gemini_api_key(api_key: str) -> None:
    if api_key.startswith("your-") or api_key.strip() != api_key:
        raise IngestionError(
            "Gemini API key is not configured correctly. Put the raw key value in "
            ".env without placeholder text or extra spaces."
        )

    if not api_key.startswith("AIza"):
        raise IngestionError(
            "The loaded Gemini API key does not look like a Google AI Studio API key. "
            "Create a Gemini API key in Google AI Studio and set it as GEMINI_API_KEY "
            "or GOOGLE_API_KEY in .env."
        )


def split_documents(
    documents: list[Document],
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split %s pages into %s chunks", len(documents), len(chunks))
    return chunks


def get_or_create_index(settings: Settings):
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = set(pc.list_indexes().names())

    if settings.pinecone_index_name not in existing_indexes:
        logger.info("Creating Pinecone index %s", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        wait_for_index_ready(pc, settings.pinecone_index_name)

    return pc.Index(settings.pinecone_index_name)


def wait_for_index_ready(pc: Pinecone, index_name: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        description = pc.describe_index(index_name)
        status = getattr(description, "status", None) or {}
        is_ready = status.get("ready") if isinstance(status, dict) else status.ready

        if is_ready:
            return

        time.sleep(2)

    raise TimeoutError(f"Pinecone index was not ready after {timeout_seconds} seconds")


def build_vectors(
    chunks: list[Document],
    *,
    embeddings: Embeddings,
    source_name: str,
    user_id: str = "default",
) -> list[dict]:
    texts = [chunk.page_content for chunk in chunks]
    try:
        embedded_texts = embeddings.embed_documents(texts)
    except Exception as exc:
        raise IngestionError(
            "Gemini embedding request failed. Check that GEMINI_API_KEY or "
            "GOOGLE_API_KEY is valid, "
            "Gemini API access is enabled for the key, and EMBEDDING_MODEL is set "
            "to a supported embedding model such as models/gemini-embedding-001."
        ) from exc

    vectors = []
    for chunk, embedding in zip(chunks, embedded_texts, strict=True):
        metadata = {
            "source": source_name,
            "user_id": user_id,
            "page": chunk.metadata.get("page"),
            "text": chunk.page_content,
        }
        vectors.append(
            {
                "id": f"{source_name}:{chunk.metadata.get('page', 'unknown')}:{uuid4()}",
                "values": embedding,
                "metadata": metadata,
            }
        )

    return vectors


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest_pdf(
    pdf_path: str | Path,
    settings: Settings | None = None,
    *,
    user_id: str = "default",
) -> int:
    settings = settings or get_settings()
    path = Path(pdf_path).expanduser().resolve()

    pages = load_pdf(path)
    chunks = split_documents(
        pages,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    if not chunks:
        logger.warning("No text chunks produced for %s", path)
        return 0

    validate_gemini_api_key(settings.google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    vectors = build_vectors(
        chunks,
        embeddings=embeddings,
        source_name=path.name,
        user_id=user_id,
    )

    try:
        index = get_or_create_index(settings)
        for batch in batched(vectors, settings.upsert_batch_size):
            index.upsert(vectors=batch)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            "Pinecone upsert failed. Check PINECONE_API_KEY, PINECONE_INDEX_NAME, "
            "PINECONE_CLOUD, PINECONE_REGION, and that the index dimension matches "
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions}."
        ) from exc

    logger.info(
        "Ingested %s vectors from %s into Pinecone index %s",
        len(vectors),
        path.name,
        settings.pinecone_index_name,
    )
    return len(vectors)
