from langchain_core.documents import Document

from backend.ingestion.pdf_loader import batched, build_vectors
from backend.rag.retriever import build_metadata_filter, format_documents, format_page_label


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


def test_build_vectors_includes_source_user_page_and_text_metadata():
    chunks = [
        Document(
            page_content="Photosynthesis turns light into chemical energy.",
            metadata={"page": 2},
        )
    ]

    vectors = build_vectors(
        chunks,
        embeddings=FakeEmbeddings(),
        source_name="biology.pdf",
        user_id="student-123",
    )

    assert len(vectors) == 1
    assert vectors[0]["values"] == [0.0, 0.0, 1.0]
    assert vectors[0]["metadata"] == {
        "source": "biology.pdf",
        "user_id": "student-123",
        "page": 2,
        "text": "Photosynthesis turns light into chemical energy.",
    }


def test_batched_splits_items_by_batch_size():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_build_metadata_filter_combines_user_and_source():
    assert build_metadata_filter(user_id="student-123", source="biology.pdf") == {
        "user_id": {"$eq": "student-123"},
        "source": {"$eq": "biology.pdf"},
    }


def test_format_documents_includes_source_page_and_content():
    output = format_documents(
        [
            Document(
                page_content="A force can change motion.",
                metadata={"source": "physics.pdf", "page": 4},
            )
        ]
    )

    assert "physics.pdf" in output
    assert "page 5" in output
    assert "A force can change motion." in output


def test_format_page_label_converts_zero_based_numeric_pages():
    assert format_page_label(0.0) == "page 1"
    assert format_page_label(2) == "page 3"
    assert format_page_label(None) == "unknown page"
