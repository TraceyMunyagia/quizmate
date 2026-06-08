from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.ingestion.pdf_loader import IngestionError
from backend.rag.retriever import RAGError, answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a grounded question against notes stored in Pinecone."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What are the main ideas in these notes?",
        help="Question to ask the notes.",
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="Only retrieve chunks for this user_id metadata value.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source PDF filename filter, for example my-notes.pdf.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        answer = answer_question(
            args.question,
            user_id=args.user_id,
            source=args.source,
            k=args.k,
        )
    except (IngestionError, RAGError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nQuestion:")
    print(args.question)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
