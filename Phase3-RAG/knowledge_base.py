"""Document model and knowledge base construction from T3DB data.

Converts T3DB CSV records into section-based ToxDocument objects
suitable for embedding and indexing in the vector store.

Each T3DB toxin record is split into up to 9 section-based documents:
    description, mechanism, metabolism, toxicity, lethaldose,
    health_effects, symptoms, treatment, carcinogenicity

This gives ~28,000-31,500 documents for ~3,500 toxins.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Document Model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ToxDocument:
    """A single document chunk for the toxicological knowledge base.

    Each document represents one section of information about one molecule.
    For example, the "mechanism" section of arsenic, or the "symptoms"
    section of lead.

    Attributes:
        doc_id: Unique document identifier (e.g., "t3db_T3D0001_mechanism")
        molecule_name: Common name of the molecule
        iupac_name: IUPAC systematic name (if available)
        cas_number: CAS registry number (if available)
        source: Data source identifier ("t3db", "pubchem")
        section: Section type ("mechanism", "symptoms", "treatment", etc.)
        content: Actual text content for embedding and retrieval
        metadata: Additional fields (smiles, pubchem_id, t3db_id, etc.)
    """
    doc_id: str
    molecule_name: str
    iupac_name: str = ""
    cas_number: str = ""
    source: str = "t3db"
    section: str = ""
    content: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for ChromaDB metadata."""
        return {
            "doc_id": self.doc_id,
            "molecule_name": self.molecule_name,
            "iupac_name": self.iupac_name,
            "cas_number": self.cas_number,
            "source": self.source,
            "section": self.section,
            **{k: str(v) for k, v in self.metadata.items() if v is not None},
        }

    def __repr__(self):
        return (
            f"ToxDocument(id={self.doc_id!r}, mol={self.molecule_name!r}, "
            f"section={self.section!r}, len={len(self.content)})"
        )


# ──────────────────────────────────────────────────────────────────────
# T3DB Section Mapping
# ──────────────────────────────────────────────────────────────────────

# Maps T3DB CSV columns to document sections
T3DB_SECTION_MAP = {
    "description": "description",
    "mechanism_of_toxicity": "mechanism",
    "metabolism": "metabolism",
    "toxicity": "toxicity",
    "lethaldose": "lethaldose",
    "health_effects": "health_effects",
    "symptoms": "symptoms",
    "treatment": "treatment",
    "carcinogenicity": "carcinogenicity",
}

# Minimum content length to create a document (skip empty/trivial entries)
MIN_CONTENT_LENGTH = 20


# ──────────────────────────────────────────────────────────────────────
# Document Builder
# ──────────────────────────────────────────────────────────────────────

def _clean_text(text) -> str:
    """Clean and normalize text from T3DB CSV fields."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    # Remove common T3DB artifacts
    if text.lower() in ("not available", "n/a", "na", "none", "-", ""):
        return ""
    return text


def build_t3db_documents(
    toxin_data_path: str,
    target_mechanisms_path: Optional[str] = None,
    t3db_processed_path: Optional[str] = None,
) -> List[ToxDocument]:
    """Build document chunks from T3DB CSV files.

    Reads the main T3DB toxin data CSV and splits each toxin record
    into section-based documents for embedding.

    Args:
        toxin_data_path: Path to all_toxin_data.csv
        target_mechanisms_path: Path to target_mechanisms.csv (optional,
            adds protein target information)
        t3db_processed_path: Path to t3db_processed.csv (optional,
            adds IUPAC names and SMILES)

    Returns:
        List of ToxDocument objects ready for indexing.
    """
    logger.info(f"Loading T3DB data from {toxin_data_path}")
    df = pd.read_csv(toxin_data_path)
    logger.info(f"Loaded {len(df)} toxin records with {len(df.columns)} columns")

    # Load processed T3DB for IUPAC names (which the raw T3DB doesn't have)
    iupac_lookup = {}
    smiles_lookup = {}
    if t3db_processed_path and os.path.exists(t3db_processed_path):
        processed = pd.read_csv(t3db_processed_path)
        for _, row in processed.iterrows():
            t3db_id = _clean_text(row.get("t3db_id", ""))
            common = _clean_text(row.get("common_name", ""))
            iupac = _clean_text(row.get("iupac_name", ""))
            smiles = _clean_text(row.get("smiles", ""))
            if t3db_id:
                iupac_lookup[t3db_id] = iupac
                smiles_lookup[t3db_id] = smiles
            if common:
                iupac_lookup[common.lower()] = iupac
                smiles_lookup[common.lower()] = smiles
        logger.info(f"Loaded {len(iupac_lookup)} IUPAC name mappings from processed T3DB")

    # Build synonyms lookup from raw T3DB synonyms_list column.
    # T3DB stores synonyms as \r\n or \r-separated strings.
    synonyms_lookup = {}  # common_name.lower() -> list of synonym strings
    if "synonyms_list" in df.columns:
        for _, row in df.iterrows():
            common = _clean_text(row.get("common_name", "")) or _clean_text(row.get("title", ""))
            raw_syns = row.get("synonyms_list", "")
            if not common or pd.isna(raw_syns) or not raw_syns:
                continue
            # Parse \r\n, \r, or \n separators
            import re as _re_syn
            syns = [s.strip() for s in _re_syn.split(r'[\r\n]+', str(raw_syns)) if s.strip()]
            # Also include IUPAC name as a synonym if available
            iupac_for_mol = iupac_lookup.get(common.lower(), "")
            if iupac_for_mol and iupac_for_mol.lower() not in [s.lower() for s in syns]:
                syns.append(iupac_for_mol)
            synonyms_lookup[common.lower()] = syns
        logger.info(f"Built synonyms lookup for {len(synonyms_lookup)} molecules")

    # Load target mechanisms for protein target info
    target_info = {}
    if target_mechanisms_path and os.path.exists(target_mechanisms_path):
        targets_df = pd.read_csv(target_mechanisms_path)
        for _, row in targets_df.iterrows():
            t3db_id = _clean_text(row.get("Toxin T3DB ID", ""))
            if not t3db_id:
                continue
            if t3db_id not in target_info:
                target_info[t3db_id] = []
            target_name = _clean_text(row.get("Target Bio Name", ""))
            mechanism = _clean_text(row.get("Mechanism of Action", ""))
            if target_name or mechanism:
                target_info[t3db_id].append({
                    "target": target_name,
                    "mechanism": mechanism,
                })
        logger.info(f"Loaded target mechanisms for {len(target_info)} toxins")

    # Build documents
    documents = []
    skipped = 0

    import re as _re

    def _is_valid_t3db_id(val: str) -> bool:
        """Check if the value looks like a real T3DB ID (e.g. T3D0001)."""
        return bool(_re.match(r"^T3D\d{4,6}$", val))

    row_counter = 0
    for _, row in df.iterrows():
        row_counter += 1
        raw_id = _clean_text(row.get("id", ""))
        common_name = _clean_text(row.get("common_name", "")) or _clean_text(row.get("title", ""))
        cas = _clean_text(row.get("cas", ""))
        pubchem_id = _clean_text(row.get("pubchem_id", ""))

        if not common_name:
            skipped += 1
            continue

        # Use the raw T3DB ID only if it looks valid; otherwise use
        # the row counter to guarantee uniqueness.  The 'id' column
        # sometimes contains HTML fragments or lethal-dose text that
        # would produce duplicate doc_ids across different records.
        if _is_valid_t3db_id(raw_id):
            unique_prefix = raw_id           # e.g. "T3D0001"
            t3db_id = raw_id
        else:
            unique_prefix = f"r{row_counter}"  # e.g. "r42"
            t3db_id = raw_id                   # keep original for metadata

        # Look up IUPAC name from processed data
        iupac = (
            iupac_lookup.get(t3db_id, "")
            or iupac_lookup.get(common_name.lower(), "")
        )
        smiles = (
            smiles_lookup.get(t3db_id, "")
            or smiles_lookup.get(common_name.lower(), "")
        )

        # Resolve synonyms for this molecule
        mol_synonyms = synonyms_lookup.get(common_name.lower(), [])
        # Store as pipe-delimited string (ChromaDB metadata must be scalar)
        synonyms_str = "|".join(mol_synonyms) if mol_synonyms else ""

        # Base metadata shared by all sections of this molecule
        base_meta = {
            "t3db_id": t3db_id,
            "pubchem_id": pubchem_id,
            "smiles": smiles,
            "synonyms": synonyms_str,
        }

        # Create one document per non-empty section
        for csv_col, section_name in T3DB_SECTION_MAP.items():
            content = _clean_text(row.get(csv_col, ""))
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            # Build synonym hint for embedding context
            synonym_hint = ""
            if mol_synonyms:
                top_syns = mol_synonyms[:5]  # Include top 5 synonyms
                synonym_hint = f"\nAlso known as: {', '.join(top_syns)}"

            # Prefix content with molecule name and section for richer
            # embedding context
            enriched_content = (
                f"Molecule: {common_name}"
                + (f" (IUPAC: {iupac})" if iupac else "")
                + (f" [CAS: {cas}]" if cas else "")
                + synonym_hint
                + f"\nSection: {section_name}\n\n{content}"
            )

            doc = ToxDocument(
                doc_id=f"t3db_{unique_prefix}_{section_name}",
                molecule_name=common_name,
                iupac_name=iupac,
                cas_number=cas,
                source="t3db",
                section=section_name,
                content=enriched_content,
                metadata=base_meta.copy(),
            )
            documents.append(doc)

        # Add target mechanism document if available
        if t3db_id in target_info:
            targets = target_info[t3db_id]
            target_lines = []
            for t in targets[:20]:  # Cap at 20 targets per molecule
                line = t.get("target", "Unknown target")
                mech = t.get("mechanism", "")
                if mech:
                    line += f": {mech}"
                target_lines.append(line)

            if target_lines:
                target_content = (
                    f"Molecule: {common_name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: protein_targets\n\n"
                    f"Protein targets and mechanisms of action:\n"
                    + "\n".join(f"- {line}" for line in target_lines)
                )

                doc = ToxDocument(
                    doc_id=f"t3db_{unique_prefix}_targets",
                    molecule_name=common_name,
                    iupac_name=iupac,
                    cas_number=cas,
                    source="t3db",
                    section="protein_targets",
                    content=target_content,
                    metadata=base_meta.copy(),
                )
                documents.append(doc)

    logger.info(
        f"Built {len(documents)} documents from {len(df)} toxins "
        f"(skipped {skipped} records with no name)"
    )

    # Report section distribution
    from collections import Counter
    section_dist = Counter(d.section for d in documents)
    for section, count in sorted(section_dist.items()):
        logger.info(f"  {section}: {count} documents")

    return documents


def build_pubchem_documents(pubchem_records: List[dict]) -> List[ToxDocument]:
    """Build document chunks from PubChem safety data.

    Args:
        pubchem_records: List of dicts from fetch_pubchem.py, each containing
            safety/hazard information for one compound.

    Returns:
        List of ToxDocument objects.
    """
    documents = []

    for record in pubchem_records:
        cid = str(record.get("cid", ""))
        name = record.get("name", "")
        iupac = record.get("iupac_name", "")
        cas = record.get("cas_number", "")
        smiles = record.get("smiles", "")

        if not name:
            continue

        base_meta = {
            "pubchem_cid": cid,
            "smiles": smiles,
        }

        # GHS classification section
        ghs = record.get("ghs_classification", "")
        if ghs and len(str(ghs)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_ghs",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="ghs_classification",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: GHS Classification\n\n{ghs}"
                ),
                metadata=base_meta.copy(),
            ))

        # Hazard statements
        hazards = record.get("hazard_statements", "")
        if hazards and len(str(hazards)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_hazards",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="hazard_statements",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: Hazard Statements\n\n{hazards}"
                ),
                metadata=base_meta.copy(),
            ))

        # Safety / first aid
        safety = record.get("safety_measures", "")
        if safety and len(str(safety)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_safety",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="safety_measures",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: Safety Measures\n\n{safety}"
                ),
                metadata=base_meta.copy(),
            ))

        # First aid measures
        first_aid = record.get("first_aid", "")
        if first_aid and len(str(first_aid)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_first_aid",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="treatment",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: First Aid & Emergency Procedures\n\n{first_aid}"
                ),
                metadata=base_meta.copy(),
            ))

        # Acute toxicity / dose-response data
        acute_tox = record.get("acute_toxicity", "")
        if acute_tox and len(str(acute_tox)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_acute_toxicity",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="lethaldose",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: Acute Toxicity & Dose-Response Data\n\n{acute_tox}"
                ),
                metadata=base_meta.copy(),
            ))

        # Pharmacology / toxicology
        pharma = record.get("pharmacology", "")
        if pharma and len(str(pharma)) >= MIN_CONTENT_LENGTH:
            documents.append(ToxDocument(
                doc_id=f"pubchem_{cid}_pharmacology",
                molecule_name=name,
                iupac_name=iupac,
                cas_number=cas,
                source="pubchem",
                section="pharmacology",
                content=(
                    f"Molecule: {name}"
                    + (f" (IUPAC: {iupac})" if iupac else "")
                    + f"\nSection: Pharmacology & Toxicology\n\n{pharma}"
                ),
                metadata=base_meta.copy(),
            ))

    logger.info(f"Built {len(documents)} documents from {len(pubchem_records)} PubChem records")
    return documents
