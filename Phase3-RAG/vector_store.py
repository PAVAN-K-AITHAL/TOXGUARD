"""ChromaDB vector store wrapper for toxicological document storage.

Handles:
    - Embedding documents using PubMedBERT (768-dim)
    - Storing and persisting in ChromaDB
    - Semantic similarity search
    - Metadata-filtered retrieval
    - Collection management (create, clear, stats)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default embedding model — PubMedBERT fine-tuned for NLI + STS tasks
DEFAULT_EMBEDDING_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
DEFAULT_COLLECTION_NAME = "toxguard_knowledge_base"


class ToxVectorStore:
    """ChromaDB-based vector store for toxicological documents.

    Wraps ChromaDB with a domain-specific API for adding, querying,
    and managing toxicological document embeddings.

    Args:
        persist_dir: Directory for ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.
        embedding_model: HuggingFace model name for sentence embeddings.
            Default: PubMedBERT (768-dim, biomedical-specialized).
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self._embedding_model_name = embedding_model

        # Lazy-initialize to avoid import cost at module level
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB client and embedding function."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.utils import embedding_functions

        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # Use PubMedBERT for embedding
        logger.info(f"Loading embedding model: {self._embedding_model_name}")
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model_name,
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        logger.info(
            f"Collection '{self.collection_name}' ready: "
            f"{self._collection.count()} documents"
        )

    @property
    def collection(self):
        """Get the ChromaDB collection (lazy init)."""
        self._ensure_initialized()
        return self._collection

    def add_documents(
        self,
        documents: list,
        batch_size: int = 500,
    ) -> int:
        """Add ToxDocument objects to the vector store.

        Deduplicates by doc_id — if a document with the same ID already
        exists, it is skipped.

        Args:
            documents: List of ToxDocument objects from knowledge_base.py.
            batch_size: Number of documents per ChromaDB upsert call.

        Returns:
            Number of documents actually added.
        """
        self._ensure_initialized()

        # Get existing IDs to avoid duplicates
        existing_count = self._collection.count()
        if existing_count > 0:
            # ChromaDB doesn't have a fast "list all IDs" for large collections,
            # but we can do an upsert which handles deduplication automatically.
            pass

        total_added = 0

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            ids = [doc.doc_id for doc in batch]
            texts = [doc.content for doc in batch]
            metadatas = [doc.to_dict() for doc in batch]

            # Upsert handles duplicates gracefully
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)

            if total_added % 2000 == 0 or total_added == len(documents):
                logger.info(
                    f"  Indexed {total_added}/{len(documents)} documents"
                )

        final_count = self._collection.count()
        logger.info(
            f"Vector store now contains {final_count} documents "
            f"(added {total_added} in this batch)"
        )
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search the vector store by semantic similarity.

        Args:
            query_text: Natural language query for embedding.
            n_results: Number of results to return.
            where: Metadata filter dict (e.g., {"source": "t3db"}).
            where_document: Document content filter.

        Returns:
            List of result dicts, each with:
                - id: document ID
                - content: document text
                - metadata: document metadata dict
                - distance: cosine distance (lower = more similar)
                - score: similarity score (1 - distance)
        """
        self._ensure_initialized()

        query_params = {
            "query_texts": [query_text],
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return []

        # Flatten results into list of dicts
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine distance → similarity
                })

        return output

    def query_by_metadata(
        self,
        where: Dict,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve documents by metadata filter only (no semantic search).

        Useful for exact-match retrieval by molecule name, CAS number, etc.

        Args:
            where: Metadata filter (e.g., {"molecule_name": "Arsenic"}).
            limit: Maximum results to return.

        Returns:
            List of result dicts with id, content, metadata.
        """
        self._ensure_initialized()

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"Metadata query failed: {e}")
            return []

        output = []
        if results and results["ids"]:
            for i in range(len(results["ids"])):
                output.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    "distance": 0.0,  # exact match
                    "score": 1.0,
                })

        return output

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        self._ensure_initialized()
        count = self._collection.count()

        # Sample to get source/section distribution
        sample = self._collection.peek(limit=min(count, 100))
        sources = set()
        sections = set()
        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                sources.add(meta.get("source", "unknown"))
                sections.add(meta.get("section", "unknown"))

        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_model": self._embedding_model_name,
            "sample_sources": list(sources),
            "sample_sections": list(sections),
        }

    def clear(self):
        """Delete all documents from the collection."""
        self._ensure_initialized()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'")

    def count(self) -> int:
        """Get the number of documents in the store."""
        self._ensure_initialized()
        return self._collection.count()
