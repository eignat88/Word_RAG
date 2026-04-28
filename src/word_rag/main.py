from __future__ import annotations

import argparse
import json

from .config import Settings
from .embeddings import OllamaError
from .rag_service import RagService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Word RAG utility")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest .docx directory")
    ingest.add_argument("directory")
    ingest.add_argument("--no-replace", action="store_true", help="Do not delete old chunks for the same document")

    search = sub.add_parser("search", help="Semantic search")
    search.add_argument("question")
    search.add_argument("--fd-number")
    search.add_argument("--section")
    search.add_argument("--top-k", type=int)

    ask = sub.add_parser("ask", help="Generate answer with sources")
    ask.add_argument("question")
    ask.add_argument("--fd-number")
    ask.add_argument("--section")
    ask.add_argument("--top-k", type=int)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    service = RagService(Settings())

    try:
        if args.command == "ingest":
            def on_progress(event: dict) -> None:
                event_type = event.get("event")
                if event_type == "start":
                    print(f"[ingest] start: directory={event['directory']} files={event['total_documents']}", flush=True)
                elif event_type == "document_start":
                    print(f"[ingest] [{event['index']}/{event['total_documents']}] processing {event['document_name']} ...", flush=True)
                elif event_type == "document_done":
                    print(
                        f"[ingest] [{event['index']}/{event['total_documents']}] done {event['document_name']} "
                        f"(inserted={event['inserted_chunks']}, skipped={event['skipped_chunks']}, {event['elapsed_sec']}s)",
                        flush=True,
                    )
                elif event_type == "done":
                    print(
                        f"[ingest] finished: documents={event['documents']}, chunks={event['chunks']}, "
                        f"skipped={event['skipped_chunks']}, elapsed={event['elapsed_sec']}s",
                        flush=True,
                    )

            result = service.ingest_directory(args.directory, replace=not args.no_replace, progress_callback=on_progress)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "search":
            result = service.search(
                question=args.question,
                fd_number=args.fd_number,
                section=args.section,
                top_k=args.top_k,
            )
            print(json.dumps([r.__dict__ for r in result], ensure_ascii=False, indent=2))
        elif args.command == "ask":
            result = service.answer(
                question=args.question,
                fd_number=args.fd_number,
                section=args.section,
                top_k=args.top_k,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except OllamaError as exc:
        print(f"[error] {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
