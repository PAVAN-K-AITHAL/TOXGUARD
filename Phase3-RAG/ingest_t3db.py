"""Ingest T3DB data into the ChromaDB vector store.

Reads T3DB CSV files, builds document chunks, and indexes them
in the vector store with PubMedBERT embeddings.

Usage:
    python ingest_t3db.py
    python ingest_t3db.py --data-dir ../data/t3db --db-dir ./chroma_db
    python ingest_t3db.py --clear  # Clear and re-ingest
"""

import argparse
import logging
import os
import sys
import time

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest T3DB data into ChromaDB vector store"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "t3db"),
        help="Path to T3DB data directory"
    )
    parser.add_argument(
        "--processed-csv", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "t3db_processed.csv"),
        help="Path to t3db_processed.csv (for IUPAC names)"
    )
    parser.add_argument(
        "--db-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "Phase3-RAG", "chroma_db"),
        help="ChromaDB persistent storage directory"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear existing data before ingesting"
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Batch size for ChromaDB upsert"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate data files
    toxin_data_path = os.path.join(args.data_dir, "all_toxin_data.csv")
    mechanisms_path = os.path.join(args.data_dir, "target_mechanisms.csv")

    if not os.path.exists(toxin_data_path):
        print(f"ERROR: T3DB data file not found: {toxin_data_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Phase 3 RAG: T3DB Knowledge Base Ingestion")
    print(f"{'='*60}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Processed CSV: {args.processed_csv}")
    print(f"  ChromaDB dir:  {args.db_dir}")
    print(f"{'='*60}\n")

    # Step 1: Build documents from T3DB
    print("Step 1: Building document chunks from T3DB...")
    from knowledge_base import build_t3db_documents

    start = time.time()
    documents = build_t3db_documents(
        toxin_data_path=toxin_data_path,
        target_mechanisms_path=mechanisms_path if os.path.exists(mechanisms_path) else None,
        t3db_processed_path=args.processed_csv if os.path.exists(args.processed_csv) else None,
    )
    build_time = time.time() - start
    print(f"  Built {len(documents)} documents in {build_time:.1f}s")

    # Step 2: Initialize vector store
    print("\nStep 2: Initializing ChromaDB vector store...")
    from vector_store import ToxVectorStore

    store = ToxVectorStore(persist_dir=args.db_dir)

    if args.clear:
        print("  Clearing existing data...")
        store.clear()

    # Step 3: Index documents
    print(f"\nStep 3: Indexing {len(documents)} documents with PubMedBERT embeddings...")
    print("  (First run will download the PubMedBERT model — ~440 MB)")

    start = time.time()
    added = store.add_documents(documents, batch_size=args.batch_size)
    index_time = time.time() - start

    print(f"  Indexed {added} documents in {index_time:.1f}s")

    # Step 4: Verify
    print("\nStep 4: Verification...")
    stats = store.get_stats()
    print(f"  Total documents in store: {stats['total_documents']}")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Embedding model: {stats['embedding_model']}")

    # Test a semantic query
    print("\n  Testing semantic search...")
    test_results = store.query(
        query_text="nitro group toxicity mechanism reactive metabolites",
        n_results=3,
    )
    for r in test_results:
        mol = r["metadata"].get("molecule_name", "?")
        section = r["metadata"].get("section", "?")
        score = r.get("score", 0)
        print(f"    → {mol} [{section}] (score={score:.3f})")

    # Test exact match
    print("\n  Testing exact match...")
    exact_results = store.query_by_metadata(
        where={"molecule_name": "Arsenic"},
        limit=5,
    )
    print(f"    → Found {len(exact_results)} documents for 'Arsenic'")

    print(f"\n{'='*60}")
    print(f"  Ingestion complete!")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Index time: {index_time:.1f}s")
    print(f"  DB location: {args.db_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
