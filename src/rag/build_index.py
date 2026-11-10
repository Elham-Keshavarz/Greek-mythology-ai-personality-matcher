"""CLI: python -m src.rag.build_index [--rebuild]"""
from __future__ import annotations

import argparse
from .pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare semantic search for Greek mythology knowledge")
    parser.add_argument("--rebuild", action="store_true", help="Recompute embeddings even if cache is current")
    args = parser.parse_args()
    store = build_pipeline(rebuild=args.rebuild)
    print(f"FAISS index ready: {len(store.chunks)} passages; embedding model: {store.embedder.model_name}")
    print("Try: python -m src.rag.ask 'Who are Apollo and Artemis?'")


if __name__ == "__main__":
    main()
