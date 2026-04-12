"""RAG generation prompt templates for safety profile synthesis.

Contains:
    1. System prompt defining the safety analyst persona
    2. Generation prompt template with retrieved document context
    3. Output parsing utilities for 9-section safety profiles
"""

import re
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are an expert toxicological safety analyst. Your task is to synthesize \
comprehensive safety profiles from retrieved toxicological databases and \
chain-of-thought analysis.

CRITICAL RULES:
1. Base ALL claims on the provided RETRIEVED DOCUMENTS. Cite sources by \
their [DOC-N] reference numbers.
2. If information is NOT available in the retrieved documents, explicitly \
state "Data not available from retrieved sources" — do NOT speculate or \
fabricate data.
3. For dose-response data (LD50, LC50, NOAEL), ONLY report values found in \
the retrieved documents. Check the LETHAL DOSE, TOXICITY, and ACUTE TOXICITY \
sections carefully — these often contain quantitative dose-response values \
that should be reported in Section 4. If quantitative LD50/LC50 values are \
present in ANY retrieved document, you MUST include them.
4. For first aid and emergency procedures (Section 5), check the TREATMENT, \
FIRST AID, and SAFETY MEASURES sections of retrieved documents. These \
contain specific emergency procedures that should be reported verbatim.
5. Distinguish between CONFIRMED information (from databases) and \
INFERRED information (from mechanistic reasoning).
6. For regulatory classification, cite the specific standard (GHS, WHO, \
EPA, IARC) and category.
7. Write in a clear, professional, safety-data-sheet style.
8. Output MUST follow the exact 10-section format provided.
9. For the CONFIDENCE RATIONALE section, explain WHY confidence is \
HIGH/MEDIUM/LOW — cite the number of corroborating sources, whether \
the molecule was found by exact match in the database, and any data \
gaps that limit confidence.

SAFETY GUARDRAILS:
- For first aid procedures, NEVER recommend induced emesis or vomiting \
unless the retrieved documents EXPLICITLY and SPECIFICALLY confirm it is \
safe for this compound. Many chemicals (oily, volatile, or corrosive \
substances) pose aspiration risk if vomited. When in doubt, state \
"Do not induce vomiting — seek emergency medical attention immediately."
- For the REFERENCES section, you MUST list each referenced document \
individually. Do NOT summarize as "[DOC-1] to [DOC-N] are referenced."\
 Each source must be listed on its own line with its DOC number, source \
database, molecule name, and section type.
"""


# ──────────────────────────────────────────────────────────────────────
# Generation Prompt Template
# ──────────────────────────────────────────────────────────────────────

RAG_GENERATION_PROMPT = """\
Generate a comprehensive toxicological safety profile for the following \
molecule based on the retrieved documents and chain-of-thought analysis.

═══════════════════════════════════════════════════════════════
MOLECULE INFORMATION
═══════════════════════════════════════════════════════════════

IUPAC Name: {iupac_name}
Common Name: {common_name}
CAS Number: {cas_number}
SMILES: {smiles}

Phase 1 Prediction: P(toxic) = {tox_score:.3f} — {severity}
Phase 2 CoT Summary:
  Functional Groups: {functional_groups}
  Mechanism: {cot_mechanism}

═══════════════════════════════════════════════════════════════
RETRIEVED DOCUMENTS ({num_docs} documents from {sources})
═══════════════════════════════════════════════════════════════

{retrieved_docs}

═══════════════════════════════════════════════════════════════
GENERATE SAFETY PROFILE
═══════════════════════════════════════════════════════════════

Using the retrieved documents above, generate a structured safety profile \
with EXACTLY these 10 sections. For each claim, cite the source document \
using [DOC-N] notation.

1. TOXICITY MECHANISM:
2. AFFECTED ORGANS & SYSTEMS:
3. SYMPTOMS OF EXPOSURE:
4. DOSE-RESPONSE DATA:
(Include ALL LD50, LC50, NOAEL, LOAEL values found in ANY retrieved document. \
Check LETHAL DOSE, TOXICITY, and ACUTE TOXICITY sections. Report species, \
route, and values. If quantitative values exist in the documents, they MUST \
appear here.)
5. FIRST AID & EMERGENCY PROCEDURES:
(Include specific procedures for inhalation, skin contact, eye contact, \
ingestion. Check TREATMENT and FIRST AID docs. Do NOT recommend induced \
vomiting unless explicitly confirmed safe for this specific compound.)
6. HANDLING & STORAGE PRECAUTIONS:
7. REGULATORY CLASSIFICATION:
8. STRUCTURALLY RELATED TOXIC COMPOUNDS:
9. REFERENCES:
(List EVERY referenced document individually on its own line. Format: \
[DOC-N] Source, Molecule_Name — Section_Type. Do NOT summarize as \
"[DOC-1] to [DOC-N]" — each document must be listed separately.)
10. CONFIDENCE RATIONALE:
(Explain why confidence is HIGH/MEDIUM/LOW. Cite number of corroborating \
sources, exact-match vs semantic-only retrieval, and any data gaps.)
"""


# ──────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────

def build_rag_prompt(
    iupac_name: str,
    common_name: str = "",
    cas_number: str = "",
    smiles: str = "",
    tox_score: float = 0.0,
    severity: str = "",
    functional_groups: str = "",
    cot_mechanism: str = "",
    retrieved_docs: List[dict] = None,
) -> str:
    """Build the RAG generation prompt with retrieved document context.

    Args:
        iupac_name: IUPAC name of the query molecule.
        common_name: Common/trade name.
        cas_number: CAS registry number.
        smiles: SMILES string.
        tox_score: P(toxic) from Phase 1.
        severity: Severity label from Phase 1.
        functional_groups: Comma-separated functional groups from Phase 2.
        cot_mechanism: Mechanism of action from Phase 2 CoT.
        retrieved_docs: List of retrieval result dicts, each with
            'content', 'metadata', 'score', 'retrieval_method'.

    Returns:
        Complete prompt string for LLM generation.
    """
    retrieved_docs = retrieved_docs or []

    # Format retrieved documents with reference numbers
    doc_blocks = []
    sources = set()
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        mol_name = meta.get("molecule_name", "")
        method = doc.get("retrieval_method", "")
        score = doc.get("score", 0.0)
        sources.add(source)

        header = (
            f"[DOC-{i}] Source: {source.upper()} | "
            f"Molecule: {mol_name} | Section: {section} | "
            f"Relevance: {score:.2f} ({method})"
        )
        content = doc.get("content", "")
        # Truncate very long documents
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"

        doc_blocks.append(f"{header}\n{content}")

    docs_text = "\n\n---\n\n".join(doc_blocks) if doc_blocks else "No documents retrieved."
    sources_str = ", ".join(sorted(sources)) if sources else "None"

    return RAG_GENERATION_PROMPT.format(
        iupac_name=iupac_name or "Unknown",
        common_name=common_name or "Unknown",
        cas_number=cas_number or "Not available",
        smiles=smiles or "Not available",
        tox_score=tox_score,
        severity=severity or "Unknown",
        functional_groups=functional_groups or "Not identified",
        cot_mechanism=cot_mechanism[:300] if cot_mechanism else "Not available",
        num_docs=len(retrieved_docs),
        sources=sources_str,
        retrieved_docs=docs_text,
    )


# ──────────────────────────────────────────────────────────────────────
# Output Parser
# ──────────────────────────────────────────────────────────────────────

# Section headers to parse from RAG output
RAG_SECTIONS = [
    "TOXICITY MECHANISM",
    "AFFECTED ORGANS & SYSTEMS",
    "SYMPTOMS OF EXPOSURE",
    "DOSE-RESPONSE DATA",
    "FIRST AID & EMERGENCY PROCEDURES",
    "HANDLING & STORAGE PRECAUTIONS",
    "REGULATORY CLASSIFICATION",
    "STRUCTURALLY RELATED TOXIC COMPOUNDS",
    "REFERENCES",
    "CONFIDENCE RATIONALE",
]

# Mapping from parsed section names to SafetyProfile field names
SECTION_TO_FIELD = {
    "TOXICITY MECHANISM": "toxicity_mechanism",
    "AFFECTED ORGANS & SYSTEMS": "affected_organs",
    "SYMPTOMS OF EXPOSURE": "symptoms_of_exposure",
    "DOSE-RESPONSE DATA": "dose_response",
    "FIRST AID & EMERGENCY PROCEDURES": "first_aid",
    "HANDLING & STORAGE PRECAUTIONS": "handling_precautions",
    "REGULATORY CLASSIFICATION": "regulatory_classification",
    "STRUCTURALLY RELATED TOXIC COMPOUNDS": "related_compounds",
    "REFERENCES": "references",
    "CONFIDENCE RATIONALE": "confidence_rationale",
}


def parse_rag_response(raw_text: str) -> Dict[str, str]:
    """Parse a structured RAG response into the 10 safety profile sections.

    Uses position-based splitting: finds all section header positions
    first, sorts them, then extracts text strictly between consecutive
    headers. This prevents content duplication that occurs with
    independent regex matching when section content contains words
    that partially match other section headers.

    Handles various formatting styles:
        - "1. TOXICITY MECHANISM:" (numbered)
        - "TOXICITY MECHANISM:" (unnumbered)
        - "**TOXICITY MECHANISM:**" (bold markdown)

    Args:
        raw_text: Raw LLM response text.

    Returns:
        Dict mapping section names to their content.
    """
    result = {section: "" for section in RAG_SECTIONS}

    # Step 1: Find the position and end of each section header in the text
    header_positions = []  # (position, header_end_position, section_name)

    for section in RAG_SECTIONS:
        escaped = re.escape(section)
        # Match: optional number + optional bold + section name + optional bold + colon/whitespace
        pattern = rf"(?:\d+\.?\s*)?(?:\*\*)?{escaped}(?:\*\*)?[:\s]*"
        for m in re.finditer(pattern, raw_text, re.IGNORECASE):
            header_positions.append((m.start(), m.end(), section))

    if not header_positions:
        return result

    # Step 2: Sort by position in the text (first occurrence wins for ordering)
    header_positions.sort(key=lambda x: x[0])

    # Step 3: Deduplicate — keep only the FIRST occurrence of each section
    seen_sections = set()
    unique_headers = []
    for start, end, section in header_positions:
        if section not in seen_sections:
            seen_sections.add(section)
            unique_headers.append((start, end, section))

    # Step 4: Extract content between consecutive headers
    for i, (start, header_end, section) in enumerate(unique_headers):
        if i + 1 < len(unique_headers):
            # Content runs from this header's end to next header's start
            content_end = unique_headers[i + 1][0]
        else:
            # Last section — content runs to end of text
            content_end = len(raw_text)

        content = raw_text[header_end:content_end].strip()

        # Clean up trailing artifacts (e.g., markdown separators)
        content = re.sub(r'\n[-=]{3,}\s*$', '', content).strip()

        result[section] = content

    return result

