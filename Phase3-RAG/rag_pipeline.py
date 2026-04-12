"""Main RAG pipeline orchestrator for Phase 3.

Connects Phase 1 (ToxGuard prediction) + Phase 2 (CoT reasoning) with
Phase 3 (retrieval-augmented safety profile generation).

Pipeline:
    1. Phase 1: Get P(toxic), severity, attention tokens, toxicophores
    2. Phase 2: Get CoT analysis with functional groups, mechanism
    3. Phase 3: Retrieve relevant documents → synthesize safety profile

Usage:
    from Phase3_RAG import RAGPipeline

    pipeline = RAGPipeline(
        vector_store_dir="./chroma_db",
        groq_api_key="gsk_...",
    )
    profile = pipeline.generate_safety_profile("nitrobenzene")
    print(profile.detailed_report())
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional

try:
    from .retriever import HybridRetriever, RetrievalResult
    from .safety_profile import SafetyProfile
    from .vector_store import ToxVectorStore
    from .fetch_pubchem import PubChemFetcher
except ImportError:
    from retriever import HybridRetriever, RetrievalResult
    from safety_profile import SafetyProfile
    from vector_store import ToxVectorStore
    from fetch_pubchem import PubChemFetcher

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for toxicological safety profiles.

    Integrates three phases:
        Phase 1: ToxGuard prediction (optional — can accept pre-computed)
        Phase 2: CoT reasoning (optional — can accept pre-computed)
        Phase 3: RAG retrieval + generation

    Args:
        vector_store_dir: Path to ChromaDB persistent storage.
        groq_api_key: Groq API key for LLM generation.
        groq_model: Groq model shorthand or full name.
        predictor: Optional ToxGuardPredictor (Phase 1).
        cot_analyzer: Optional CoTAnalyzer (Phase 2).
        enable_pubchem: Whether to fetch PubChem data for unknown molecules.
        temperature: LLM generation temperature.
        max_tokens: Maximum tokens for LLM response.
    """

    def __init__(
        self,
        vector_store_dir: str = "./chroma_db",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        predictor=None,
        cot_analyzer=None,
        enable_pubchem: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ):
        # Vector store
        self.store = ToxVectorStore(persist_dir=vector_store_dir)

        # PubChem fetcher
        self.pubchem_fetcher = PubChemFetcher() if enable_pubchem else None

        # Retriever
        self.retriever = HybridRetriever(
            vector_store=self.store,
            pubchem_fetcher=self.pubchem_fetcher,
        )

        # LLM client (reuse Phase 2's Groq client)
        self._llm = None
        self._groq_api_key = groq_api_key
        self._groq_model = groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Phase 1 + 2 (optional)
        self.predictor = predictor
        self.cot_analyzer = cot_analyzer

    def _ensure_llm(self):
        """Lazy-initialize the LLM client."""
        if self._llm is not None:
            return

        # Add Phase 2 to import path (append, NOT insert, to avoid
        # shadowing Phase3-RAG modules like prompts.py)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        phase2_path = os.path.join(project_root, "Phase2-CoT")
        if phase2_path not in sys.path:
            sys.path.append(phase2_path)

        from llm_client import GroqLLMClient
        self._llm = GroqLLMClient(
            api_key=self._groq_api_key,
            model=self._groq_model,
        )
        logger.info(f"Initialized LLM client: {self._groq_model}")

    def generate_safety_profile(
        self,
        iupac_name: str,
        common_name: str = "",
        cas_number: str = "",
        smiles: str = "",
        inchikey: str = "",
        # Optional pre-computed Phase 1 results
        toxicity_score: Optional[float] = None,
        severity_label: Optional[str] = None,
        is_toxic: Optional[bool] = None,
        top_tokens: Optional[List[dict]] = None,
        toxicophore_hits: Optional[List[dict]] = None,
        # Optional pre-computed Phase 2 results
        functional_groups: Optional[List[str]] = None,
        cot_mechanism: Optional[str] = None,
        cot_pathways: Optional[str] = None,
    ) -> SafetyProfile:
        """Generate a comprehensive safety profile for a molecule.

        Can run in three modes:
            1. Full pipeline: Phase 1 → 2 → 3 (if predictor & cot_analyzer set)
            2. Partial: Phase 3 with pre-computed Phase 1/2 outputs
            3. RAG-only: Phase 3 with minimal input (just name)

        Args:
            iupac_name: IUPAC name of the molecule.
            common_name: Common name (optional).
            cas_number: CAS registry number (optional).
            smiles: SMILES string (optional).
            toxicity_score: Pre-computed P(toxic) from Phase 1.
            severity_label: Pre-computed severity from Phase 1.
            is_toxic: Pre-computed binary label from Phase 1.
            top_tokens: Pre-computed attention tokens from Phase 1.
            toxicophore_hits: Pre-computed toxicophore hits from Phase 1.
            functional_groups: Pre-computed functional groups from Phase 2.
            cot_mechanism: Pre-computed mechanism from Phase 2 CoT.
            cot_pathways: Pre-computed pathways from Phase 2 CoT.

        Returns:
            SafetyProfile with all 9 sections populated.
        """
        self._ensure_llm()

        # ── Step 1: Phase 1 Prediction (if needed + available) ────────
        if toxicity_score is None and self.predictor is not None:
            logger.info(f"Running Phase 1 prediction for: {iupac_name}")
            try:
                prediction = self.predictor.predict(
                    iupac_name,
                    return_attention=True,
                    attention_top_k=10,
                )
                toxicity_score = prediction.toxicity_score
                severity_label = prediction.severity_label
                is_toxic = prediction.is_toxic
                top_tokens = prediction.top_tokens or []
                toxicophore_hits = prediction.toxicophore_hits or []
            except Exception as e:
                logger.warning(f"Phase 1 prediction failed: {e}")

        # Defaults if Phase 1 not available
        if toxicity_score is None:
            toxicity_score = 0.5
        if severity_label is None:
            severity_label = "Unknown"
        if is_toxic is None:
            is_toxic = toxicity_score >= 0.5

        # ── Step 2: Phase 2 CoT (if needed + available) ───────────────
        if functional_groups is None and self.cot_analyzer is not None:
            logger.info(f"Running Phase 2 CoT analysis for: {iupac_name}")
            try:
                cot_result = self.cot_analyzer.analyze_from_prediction(
                    iupac_name=iupac_name,
                    toxicity_score=toxicity_score,
                    severity_label=severity_label,
                    is_toxic=is_toxic,
                    top_tokens=top_tokens,
                    toxicophore_hits=toxicophore_hits,
                )
                functional_groups = cot_result.functional_groups
                cot_mechanism = cot_result.mechanism_of_action
                cot_pathways = cot_result.biological_pathways
            except Exception as e:
                logger.warning(f"Phase 2 CoT analysis failed: {e}")

        functional_groups = functional_groups or []
        cot_mechanism = cot_mechanism or ""
        cot_pathways = cot_pathways or ""

        # ── Step 2.5: Auto-populate identifiers via PubChem ───────────
        if self.pubchem_fetcher and (not cas_number or not smiles):
            logger.info(f"Looking up identifiers for: {iupac_name}")
            try:
                ids = self.pubchem_fetcher.lookup_identifiers(iupac_name)
                cas_number = cas_number or ids.get("cas_number", "")
                smiles = smiles or ids.get("smiles", "")
                inchikey = inchikey or ids.get("inchikey", "")
                common_name = common_name or ids.get("name", "")
                logger.info(
                    f"PubChem identifiers: CAS={cas_number}, "
                    f"SMILES={smiles[:30] if smiles else ''}, "
                    f"InChIKey={inchikey}"
                )
            except Exception as e:
                logger.warning(f"PubChem identifier lookup failed: {e}")

        # ── Step 3: Phase 3 RAG Retrieval ─────────────────────────────
        logger.info(f"Retrieving documents for: {iupac_name}")
        retrieved = self.retriever.retrieve(
            query_name=common_name or iupac_name,
            cas_number=cas_number,
            functional_groups=functional_groups,
            mechanism=cot_mechanism,
            pathways=cot_pathways,
        )

        # Also try IUPAC name if common_name was used
        if common_name and common_name != iupac_name:
            iupac_results = self.retriever.retrieve(
                query_name=iupac_name,
                functional_groups=functional_groups,
                mechanism=cot_mechanism,
                fetch_pubchem=False,  # Don't double-fetch PubChem
            )
            # Merge deduplicated
            existing_ids = {r.doc_id for r in retrieved}
            for r in iupac_results:
                if r.doc_id not in existing_ids:
                    retrieved.append(r)

        logger.info(f"Retrieved {len(retrieved)} documents total")

        # ── Step 4: LLM Generation ────────────────────────────────────
        # Use importlib to guarantee we load Phase3-RAG/prompts.py, not
        # Phase2-CoT/prompts.py (which can shadow it via sys.path).
        try:
            from .prompts import (
                RAG_SYSTEM_PROMPT,
                build_rag_prompt,
                parse_rag_response,
                SECTION_TO_FIELD,
            )
        except ImportError:
            import importlib.util as _ilu
            _prompts_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "prompts.py"
            )
            _spec = _ilu.spec_from_file_location("rag_prompts", _prompts_path)
            _rag_prompts = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_rag_prompts)
            RAG_SYSTEM_PROMPT = _rag_prompts.RAG_SYSTEM_PROMPT
            build_rag_prompt = _rag_prompts.build_rag_prompt
            parse_rag_response = _rag_prompts.parse_rag_response
            SECTION_TO_FIELD = _rag_prompts.SECTION_TO_FIELD

        groups_str = ", ".join(functional_groups) if functional_groups else ""
        user_prompt = build_rag_prompt(
            iupac_name=iupac_name,
            common_name=common_name,
            cas_number=cas_number,
            smiles=smiles,
            tox_score=toxicity_score,
            severity=severity_label,
            functional_groups=groups_str,
            cot_mechanism=cot_mechanism,
            retrieved_docs=[
                {
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.score,
                    "retrieval_method": r.retrieval_method,
                }
                for r in retrieved
            ],
        )

        logger.info("Generating safety profile via LLM...")
        start_time = time.time()
        response = self._llm.generate(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = (time.time() - start_time) * 1000

        # ── Step 5: Parse Response ────────────────────────────────────
        sections = parse_rag_response(response.content)

        # Build SafetyProfile
        profile_kwargs = {}
        for section_name, field_name in SECTION_TO_FIELD.items():
            profile_kwargs[field_name] = sections.get(section_name, "")

        # Filter unused references
        refs_text = profile_kwargs.get("references", "")
        if refs_text:
            import re
            # Combine all content sections (excluding references)
            all_content = " ".join(
                v for k, v in profile_kwargs.items() if k != "references"
            )
            # Find all cited DOC-N tags
            cited_docs = set(re.findall(r"\[DOC-\d+\]", all_content))
            
            # Filter the references section lines
            filtered_refs = []
            for line in refs_text.split("\n"):
                match = re.search(r"\[DOC-\d+\]", line)
                if match:
                    # Keep if cited
                    if match.group(0) in cited_docs:
                        filtered_refs.append(line)
                else:
                    # Keep lines without tags (multi-line ref parts)
                    filtered_refs.append(line)
                    
            profile_kwargs["references"] = "\n".join(filtered_refs).strip()

        profile = SafetyProfile(
            iupac_name=iupac_name,
            common_name=common_name,
            cas_number=cas_number,
            smiles=smiles,
            inchikey=inchikey,
            toxicity_score=toxicity_score,
            severity_label=severity_label,
            is_toxic=is_toxic,
            cot_mechanism=cot_mechanism[:500] if cot_mechanism else "",
            cot_functional_groups=functional_groups,
            num_retrieved_docs=len(retrieved),
            retrieval_sources=list(set(r.source for r in retrieved)),
            llm_model=response.model,
            llm_latency_ms=latency,
            **profile_kwargs,
        )

        logger.info(f"Safety profile generated: {profile.summary()}")
        return profile

    def generate_batch(
        self,
        molecules: List[Dict],
        delay_between: float = 2.0,
        save_path: Optional[str] = None,
    ) -> List[SafetyProfile]:
        """Generate safety profiles for multiple molecules.

        Args:
            molecules: List of dicts, each with at least 'iupac_name'.
                Optional keys: common_name, cas_number, smiles,
                toxicity_score, severity_label, is_toxic.
            delay_between: Seconds between API calls.
            save_path: If provided, save profiles incrementally to JSON.

        Returns:
            List of SafetyProfile objects.
        """
        import json

        profiles = []

        for i, mol in enumerate(molecules):
            iupac = mol.get("iupac_name", "")
            if not iupac:
                logger.warning(f"Skipping molecule {i}: no iupac_name")
                continue

            logger.info(f"[{i+1}/{len(molecules)}] Processing: {iupac}")
            try:
                profile = self.generate_safety_profile(**mol)
                profiles.append(profile)
                logger.info(f"  → {profile.summary()}")

                # Incremental save
                if save_path:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    with open(save_path, "w") as f:
                        json.dump(
                            [p.to_dict() for p in profiles],
                            f, indent=2,
                        )

            except Exception as e:
                logger.error(f"  → FAILED: {e}")

            if i < len(molecules) - 1:
                time.sleep(delay_between)

        logger.info(f"Generated {len(profiles)}/{len(molecules)} safety profiles")
        return profiles
