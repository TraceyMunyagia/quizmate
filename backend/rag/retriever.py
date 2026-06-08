from __future__ import annotations

from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.core.config import Settings, get_settings
from backend.ingestion.pdf_loader import validate_gemini_api_key
from backend.prompts.templates import GROUNDED_QA_PROMPT


class RAGError(RuntimeError):
    """Raised when retrieval or answer generation fails."""


def build_metadata_filter(
    *,
    user_id: str | None = "default",
    source: str | None = None,
) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}

    if user_id:
        filters["user_id"] = {"$eq": user_id}

    if source:
        filters["source"] = {"$eq": source}

    return filters or None


def create_embeddings(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    settings = settings or get_settings()
    validate_gemini_api_key(settings.google_api_key)

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def create_vector_store(settings: Settings | None = None) -> PineconeVectorStore:
    settings = settings or get_settings()

    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=create_embeddings(settings),
        pinecone_api_key=settings.pinecone_api_key,
    )


def create_retriever(
    *,
    user_id: str | None = "default",
    source: str | None = None,
    k: int = 4,
    settings: Settings | None = None,
):
    vector_store = create_vector_store(settings)
    metadata_filter = build_metadata_filter(user_id=user_id, source=source)
    search_kwargs: dict[str, Any] = {"k": k}

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "No relevant notes were found."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        page_label = format_page_label(document.metadata.get("page"))
        formatted.append(
            f"[{index}] Source: {source}, {page_label}\n{document.page_content}"
        )

    return "\n\n".join(formatted)


def format_page_label(page: object) -> str:
    if page is None:
        return "unknown page"

    if isinstance(page, (int, float)):
        return f"page {int(page) + 1}"

    return f"page {page}"


def create_rag_chain(
    *,
    user_id: str | None = "default",
    source: str | None = None,
    k: int = 4,
    settings: Settings | None = None,
):
    settings = settings or get_settings()
    retriever = create_retriever(
        user_id=user_id,
        source=source,
        k=k,
        settings=settings,
    )
    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        api_key=settings.google_api_key,
        temperature=0,
    )

    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_documents
        )
        | GROUNDED_QA_PROMPT
        | llm
        | StrOutputParser()
    )


def answer_question(
    question: str,
    *,
    user_id: str | None = "default",
    source: str | None = None,
    k: int = 4,
    settings: Settings | None = None,
) -> str:
    chain = create_rag_chain(user_id=user_id, source=source, k=k, settings=settings)
    try:
        return chain.invoke({"question": question})
    except Exception as exc:
        raise RAGError(
            "RAG answer generation failed. Check Pinecone connectivity, Gemini chat "
            "model quota, and CHAT_MODEL in .env."
        ) from exc
