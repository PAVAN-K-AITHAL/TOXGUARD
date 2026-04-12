"""Phase 3: Retrieval-Augmented Generation for Toxicological Safety Profiles.

Retrieves comprehensive toxicological data from curated knowledge bases
(T3DB + PubChem) and synthesizes structured safety profiles via LLM.

Pipeline:
    1. ToxGuard Phase 1 → P(toxic) + attention + toxicophores
    2. Phase 2 CoT → mechanistic reasoning
    3. Phase 3 RAG → retrieve relevant safety data → synthesize profile
       a. Build query from Phase 1 + Phase 2 outputs
       b. Retrieve relevant documents from ChromaDB vector store
       c. Synthesize comprehensive safety profile via LLM

Data Sources:
    - T3DB (3,512 toxins × 9 sections = ~31K documents)
    - PubChem (on-demand safety/hazard data for any compound)

Reference: MolRAG molecular retrieval-augmented generation
"""

__version__ = "0.1.0"

from .knowledge_base import ToxDocument, build_t3db_documents
from .vector_store import ToxVectorStore
from .retriever import HybridRetriever, RetrievalResult
from .safety_profile import SafetyProfile
from .rag_pipeline import RAGPipeline

__all__ = [
    "ToxDocument",
    "build_t3db_documents",
    "ToxVectorStore",
    "HybridRetriever",
    "RetrievalResult",
    "SafetyProfile",
    "RAGPipeline",
]
