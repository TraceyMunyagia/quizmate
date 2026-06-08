from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.ingestion.pdf_loader import IngestionError, ingest_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Pinecone using Gemini embeddings."
    )
    parser.add_argument("pdf_path", help="Path to the PDF notes file.")
    parser.add_argument(
        "--user-id",
        default="default",
        help="User identifier stored in Pinecone metadata for filtered retrieval.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        parser.error(f"Expected a .pdf file, got: {pdf_path}")

    args.pdf_path = pdf_path
    return args


def main() -> None:
    args = parse_args()
    try:
        count = ingest_pdf(args.pdf_path, user_id=args.user_id)
    except IngestionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Stored {count} vectors in Pinecone.")


if __name__ == "__main__":
    main()
