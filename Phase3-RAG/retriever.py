"""Hybrid retrieval system for toxicological knowledge base.

Combines three retrieval strategies:
    1. Exact Match: Direct lookup by molecule name, CAS number, or SMILES
    2. Semantic Search: PubMedBERT embedding similarity for mechanism/pathway queries
    3. Reranking: Score-based merging with section diversity bonus

The retriever is designed to maximize recall for safety-critical queries
while maintaining relevance through reranking.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document with relevance metadata."""
    doc_id: str
    content: str
    metadata: Dict
    score: float                  # Final relevance score [0, 1]
    retrieval_method: str         # "exact_match", "semantic", "pubchem"
    section: str = ""             # Document section type
    molecule_name: str = ""       # Molecule name from metadata
    source: str = ""              # Data source (t3db, pubchem)

    def __repr__(self):
        return (
            f"RetrievalResult(mol={self.molecule_name!r}, "
            f"section={self.section!r}, score={self.score:.3f}, "
            f"method={self.retrieval_method!r})"
        )


class HybridRetriever:
    """Hybrid retrieval combining exact match + semantic search.

    Retrieval pipeline:
        1. Try exact match by molecule name / CAS → get all sections
        2. Build semantic query from functional groups + mechanisms
        3. Semantic search in vector store → top-K results
        4. Merge, deduplicate, and rerank results
        5. Optionally fetch PubChem data for unknown molecules

    Args:
        vector_store: ToxVectorStore instance.
        pubchem_fetcher: Optional PubChemFetcher for on-demand data.
        semantic_top_k: Number of semantic search results to retrieve.
        final_top_k: Final number of results after reranking.
        exact_match_boost: Score boost for exact-match results.
        section_diversity_bonus: Bonus for covering diverse sections.
    """

    # Section importance weights for reranking
    SECTION_WEIGHTS = {
        "mechanism": 1.0,
        "toxicity": 0.95,
        "lethaldose": 1.0,
        "health_effects": 0.90,
        "symptoms": 0.85,
        "treatment": 0.95,
        "metabolism": 0.80,
        "carcinogenicity": 0.80,
        "description": 0.70,
        "protein_targets": 0.75,
        "ghs_classification": 0.90,
        "hazard_statements": 0.85,
        "safety_measures": 0.95,
        "pharmacology": 0.80,
    }

    def __init__(
        self,
        vector_store,
        pubchem_fetcher=None,
        semantic_top_k: int = 20,
        final_top_k: int = 12,
        exact_match_boost: float = 0.3,
        section_diversity_bonus: float = 0.08,
        min_relevance_threshold: float = 0.35,
    ):
        self.store = vector_store
        self.pubchem_fetcher = pubchem_fetcher
        self.semantic_top_k = semantic_top_k
        self.final_top_k = final_top_k
        self.exact_match_boost = exact_match_boost
        self.section_diversity_bonus = section_diversity_bonus
        self.min_relevance_threshold = min_relevance_threshold

    def retrieve(
        self,
        query_name: str,
        cas_number: Optional[str] = None,
        functional_groups: Optional[List[str]] = None,
        mechanism: Optional[str] = None,
        pathways: Optional[str] = None,
        fetch_pubchem: bool = True,
        exclude_molecule: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant toxicological documents for a molecule.

        Args:
            query_name: Molecule name (IUPAC or common name).
            cas_number: CAS registry number (optional, for exact match).
            functional_groups: List of identified functional groups (from Phase 2).
            mechanism: Mechanism of action text (from Phase 2 CoT).
            pathways: Biological pathways text (from Phase 2 CoT).
            fetch_pubchem: Whether to try PubChem for unknown molecules.
            exclude_molecule: Molecule name to EXCLUDE from results. Used in
                Phase 4 to prevent cross-contamination — when querying for a
                detoxified candidate, exclude docs from the parent toxic molecule.

        Returns:
            List of RetrievalResult objects, sorted by relevance score.
        """
        all_results = {}  # doc_id -> RetrievalResult (deduplication)

        # ── Phase A: Exact Match by Name/CAS ──────────────────────────
        exact_results = self._exact_match_retrieval(query_name, cas_number)
        for r in exact_results:
            all_results[r.doc_id] = r

        # ── Phase B: Semantic Search ──────────────────────────────────
        semantic_query = self._build_semantic_query(
            query_name, functional_groups, mechanism, pathways
        )
        semantic_results = self._semantic_search(semantic_query)

        for r in semantic_results:
            if r.doc_id in all_results:
                # Already found via exact match — boost score
                existing = all_results[r.doc_id]
                existing.score = min(1.0, existing.score + r.score * 0.3)
            else:
                all_results[r.doc_id] = r

        logger.info(
            f"Semantic search: {len(semantic_results)} results, "
            f"total unique: {len(all_results)}"
        )

        # ── Phase C: PubChem Fetch (if few results or missing toxicity data) ───
        has_tox_data = any(
            r.section in ("lethaldose", "pharmacology") for r in exact_results
        )
        if fetch_pubchem and self.pubchem_fetcher and (len(exact_results) < 3 or not has_tox_data):
            pubchem_results = self._fetch_pubchem_docs(query_name)
            for r in pubchem_results:
                if r.doc_id not in all_results:
                    all_results[r.doc_id] = r

            if pubchem_results:
                logger.info(
                    f"PubChem fetch: {len(pubchem_results)} additional documents"
                )

        # ── Phase C.5: Parent Exclusion Filter ────────────────────────
        if exclude_molecule:
            exclude_lower = exclude_molecule.lower().strip()
            before = len(all_results)
            all_results = {
                doc_id: r for doc_id, r in all_results.items()
                if r.molecule_name.lower().strip() != exclude_lower
            }
            excluded = before - len(all_results)
            if excluded:
                logger.info(
                    f"Parent exclusion: removed {excluded} docs from '{exclude_molecule}'"
                )

        # ── Phase D: Rerank and Select Top-K ──────────────────────────
        ranked = self._rerank(list(all_results.values()), query_name=query_name)
        final = ranked[: self.final_top_k]

        logger.info(
            f"Final retrieval for '{query_name}': {len(final)} documents "
            f"(from {len(all_results)} total)"
        )

        return final

    def _exact_match_retrieval(
        self,
        query_name: str,
        cas_number: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Try exact metadata match by molecule name or CAS number.

        ChromaDB where-filters are case-sensitive, but molecule names
        may be stored in different cases across sources (T3DB uses
        title case, PubChem uses uppercase, users type lowercase).
        We try all common case variants to ensure completeness.
        """
        results = []
        seen_ids = set()

        # Try multiple case variants for molecule name
        name_variants = list(dict.fromkeys([
            query_name,
            query_name.lower(),
            query_name.title(),
            query_name.upper(),
        ]))

        for variant in name_variants:
            name_results = self.store.query_by_metadata(
                where={"molecule_name": variant},
                limit=50,
            )
            for r in name_results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(RetrievalResult(
                        doc_id=r["id"],
                        content=r["content"],
                        metadata=r["metadata"],
                        score=1.0,  # Perfect match
                        retrieval_method="exact_match",
                        section=r["metadata"].get("section", ""),
                        molecule_name=r["metadata"].get("molecule_name", ""),
                        source=r["metadata"].get("source", ""),
                    ))

        # Also try iupac_name field if still no results
        if not results:
            for variant in name_variants:
                iupac_results = self.store.query_by_metadata(
                    where={"iupac_name": variant},
                    limit=50,
                )
                for r in iupac_results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        results.append(RetrievalResult(
                            doc_id=r["id"],
                            content=r["content"],
                            metadata=r["metadata"],
                            score=1.0,
                            retrieval_method="exact_match",
                            section=r["metadata"].get("section", ""),
                            molecule_name=r["metadata"].get("molecule_name", ""),
                            source=r["metadata"].get("source", ""),
                        ))
                if results:
                    break

        # Synonym-based lookup — bridges IUPAC↔common name gap.
        # If querying 'methanal', finds 'Formaldehyde' via its synonym list.
        if not results:
            for variant in name_variants:
                synonym_results = self.store.query_by_synonym(
                    synonym=variant,
                    limit=50,
                )
                for r in synonym_results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        results.append(RetrievalResult(
                            doc_id=r["id"],
                            content=r["content"],
                            metadata=r["metadata"],
                            score=0.95,  # High but slightly below direct exact match
                            retrieval_method="synonym_match",
                            section=r["metadata"].get("section", ""),
                            molecule_name=r["metadata"].get("molecule_name", ""),
                            source=r["metadata"].get("source", ""),
                        ))
                if results:
                    break

        # Try by CAS number
        if cas_number and not results:
            cas_results = self.store.query_by_metadata(
                where={"cas_number": cas_number},
                limit=50,
            )
            for r in cas_results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(RetrievalResult(
                        doc_id=r["id"],
                        content=r["content"],
                        metadata=r["metadata"],
                        score=1.0,
                        retrieval_method="exact_match",
                        section=r["metadata"].get("section", ""),
                        molecule_name=r["metadata"].get("molecule_name", ""),
                        source=r["metadata"].get("source", ""),
                    ))

        logger.info(f"Exact match for '{query_name}': {len(results)} documents")
        return results

    def _build_semantic_query(
        self,
        query_name: str,
        functional_groups: Optional[List[str]] = None,
        mechanism: Optional[str] = None,
        pathways: Optional[str] = None,
    ) -> str:
        """Build a rich semantic query from Phase 2 CoT outputs.

        The query is designed to match relevant toxicological documents
        even for molecules not in T3DB, by focusing on structural and
        mechanistic similarity.
        """
        parts = [f"Toxicological information for {query_name}."]

        if functional_groups:
            groups_str = ", ".join(functional_groups[:5])
            parts.append(f"Contains functional groups: {groups_str}.")

        if mechanism:
            # Take first 200 chars to keep query focused
            parts.append(f"Mechanism of toxicity: {mechanism[:200]}")

        if pathways:
            parts.append(f"Biological pathways: {pathways[:150]}")

        return " ".join(parts)

    def _semantic_search(self, query: str) -> List[RetrievalResult]:
        """Run semantic similarity search on the vector store.

        Filters out results below the minimum relevance threshold.
        """
        raw_results = self.store.query(
            query_text=query,
            n_results=self.semantic_top_k,
        )

        results = []
        for r in raw_results:
            score = r["score"]
            # Drop low-relevance results early
            if score < self.min_relevance_threshold:
                continue
            results.append(RetrievalResult(
                doc_id=r["id"],
                content=r["content"],
                metadata=r["metadata"],
                score=score,
                retrieval_method="semantic",
                section=r["metadata"].get("section", ""),
                molecule_name=r["metadata"].get("molecule_name", ""),
                source=r["metadata"].get("source", ""),
            ))

        return results

    def _fetch_pubchem_docs(self, query_name: str) -> List[RetrievalResult]:
        """Fetch safety data from PubChem for molecules not in T3DB."""
        if not self.pubchem_fetcher:
            return []

        try:
            records = self.pubchem_fetcher.fetch_safety_data(query_name)
            if not records:
                return []

            # Convert PubChem records to documents
            try:
                from .knowledge_base import build_pubchem_documents
            except ImportError:
                from knowledge_base import build_pubchem_documents
            docs = build_pubchem_documents(records)

            # Add to vector store for future queries
            if docs:
                self.store.add_documents(docs)

            results = []
            for doc in docs:
                results.append(RetrievalResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.to_dict(),
                    score=0.85,  # High but below exact match
                    retrieval_method="pubchem",
                    section=doc.section,
                    molecule_name=doc.molecule_name,
                    source="pubchem",
                ))

            return results

        except Exception as e:
            logger.warning(f"PubChem fetch failed for '{query_name}': {e}")
            return []

    def _rerank(
        self,
        results: List[RetrievalResult],
        query_name: str = "",
    ) -> List[RetrievalResult]:
        """Rerank results using section weights, diversity bonus, and name overlap.

        Scoring formula:
            final_score = base_score
                        + exact_match_boost (if exact match)
                        + section_weight * 0.1
                        + diversity_bonus (if section not yet seen)
                        * name_penalty (0.5 if no token overlap with query)
        """
        if not results:
            return []

        seen_sections = set()
        query_tokens = set(query_name.lower().split()) if query_name else set()

        for r in results:
            # Section weight bonus
            section_weight = self.SECTION_WEIGHTS.get(r.section, 0.5)
            r.score += section_weight * 0.1

            # Exact match boost
            if r.retrieval_method == "exact_match":
                r.score += self.exact_match_boost

            # Diversity bonus for unique sections
            if r.section not in seen_sections:
                r.score += self.section_diversity_bonus
                seen_sections.add(r.section)

            # Penalize semantic-only results whose molecule name has
            # no token overlap with the query — these are likely noise
            if (
                query_tokens
                and r.retrieval_method == "semantic"
                and r.molecule_name
            ):
                mol_tokens = set(r.molecule_name.lower().split())
                if not query_tokens & mol_tokens:
                    r.score *= 0.5

            # Clamp to [0, 1]
            r.score = min(1.0, max(0.0, r.score))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def retrieve_with_details(
        self,
        query_name: str,
        cas_number: Optional[str] = None,
        functional_groups: Optional[List[str]] = None,
        mechanism: Optional[str] = None,
        pathways: Optional[str] = None,
        fetch_pubchem: bool = True,
    ) -> Dict:
        """Retrieve documents with full diagnostic metadata.

        Same as retrieve() but returns additional metadata needed for
        Pillar 1 evaluation diagnostics: timing, raw pre-rerank scores,
        pathway breakdown, and section coverage.

        Returns:
            Dict with:
                results: List[RetrievalResult] (final ranked results)
                diagnostics: Dict with timing, pathway counts, section coverage
        """
        import time as _time

        t0 = _time.time()

        # Phase A: Exact match
        t_exact_start = _time.time()
        exact_results = self._exact_match_retrieval(query_name, cas_number)
        t_exact = _time.time() - t_exact_start

        all_results = {}
        for r in exact_results:
            all_results[r.doc_id] = r

        # Phase B: Semantic search
        t_sem_start = _time.time()
        semantic_query = self._build_semantic_query(
            query_name, functional_groups, mechanism, pathways
        )
        semantic_results = self._semantic_search(semantic_query)
        t_semantic = _time.time() - t_sem_start

        raw_semantic_scores = [r.score for r in semantic_results]

        for r in semantic_results:
            if r.doc_id in all_results:
                existing = all_results[r.doc_id]
                existing.score = min(1.0, existing.score + r.score * 0.3)
            else:
                all_results[r.doc_id] = r

        # Phase C: PubChem
        t_pubchem_start = _time.time()
        pubchem_results = []
        has_tox_data = any(
            r.section in ("lethaldose", "pharmacology") for r in exact_results
        )
        if fetch_pubchem and self.pubchem_fetcher and (
            len(exact_results) < 3 or not has_tox_data
        ):
            pubchem_results = self._fetch_pubchem_docs(query_name)
            for r in pubchem_results:
                if r.doc_id not in all_results:
                    all_results[r.doc_id] = r
        t_pubchem = _time.time() - t_pubchem_start

        # Phase D: Rerank
        t_rerank_start = _time.time()
        pre_rerank_scores = {
            r.doc_id: r.score for r in all_results.values()
        }
        ranked = self._rerank(list(all_results.values()), query_name=query_name)
        final = ranked[:self.final_top_k]
        t_rerank = _time.time() - t_rerank_start

        total_time = _time.time() - t0

        # Section coverage
        sections_covered = set(r.section for r in final if r.section)

        # Pathway breakdown
        pathway_counts = {}
        for r in final:
            method = r.retrieval_method
            pathway_counts[method] = pathway_counts.get(method, 0) + 1

        diagnostics = {
            "timing_ms": {
                "total": total_time * 1000,
                "exact_match": t_exact * 1000,
                "semantic": t_semantic * 1000,
                "pubchem": t_pubchem * 1000,
                "rerank": t_rerank * 1000,
            },
            "counts": {
                "exact_match_docs": len(exact_results),
                "semantic_docs": len(semantic_results),
                "pubchem_docs": len(pubchem_results),
                "total_unique": len(all_results),
                "final_top_k": len(final),
            },
            "pathway_breakdown": pathway_counts,
            "sections_covered": list(sections_covered),
            "raw_semantic_scores": raw_semantic_scores[:10],
            "pre_rerank_scores": {
                k: v for k, v in list(pre_rerank_scores.items())[:10]
            },
        }

        return {"results": final, "diagnostics": diagnostics}
