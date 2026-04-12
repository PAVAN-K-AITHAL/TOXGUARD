"""CoT prompt templates and few-shot exemplars for toxicity reasoning.

Contains:
    1. System prompt defining the toxicologist persona
    2. Few-shot exemplars curated from T3DB with known mechanisms
    3. Query template for new molecules
    4. Output parsing utilities

Reference: CoTox (Park et al., BIBM 2025) prompt design.
"""

import re
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert computational toxicologist with deep knowledge of:
- Organic chemistry and IUPAC nomenclature
- Structure-activity relationships (SAR) in toxicology
- Toxicophore identification (reactive functional groups)
- Biological pathways of toxicity (NR, SR, oxidative stress)
- Organ-specific toxicity mechanisms
- Gene Ontology (GO) annotations for toxicity endpoints

Given a molecule's IUPAC name, its predicted toxicity probability from a \
machine learning model, and highlighted structural features, you must provide \
a rigorous chain-of-thought toxicity analysis.

IMPORTANT RULES:
1. Base your analysis on established toxicology knowledge, not speculation.
2. If a functional group's toxicity mechanism is uncertain, state "uncertain" \
rather than guessing.
3. Always distinguish between KNOWN mechanisms and HYPOTHESIZED ones.
4. Reference specific toxicity pathways (e.g., "NR-AhR activation", \
"mitochondrial dysfunction", "hERG channel blockade").
5. Output MUST follow the exact structured format provided.
6. In STRUCTURAL ANALYSIS, list ONLY functional groups present in the parent \
molecule's chemical structure (i.e., groups you can identify from the IUPAC \
name alone). Do NOT include metabolites, biotransformation products, or \
reactive intermediates here. Those belong in MECHANISM OF ACTION.
"""


# ──────────────────────────────────────────────────────────────────────
# Few-Shot Exemplars
# ──────────────────────────────────────────────────────────────────────

FEW_SHOT_EXEMPLARS = [
    # ── Exemplar 1: Nitrobenzene (aromatic nitro compound) ──────────
    {
        "iupac_name": "nitrobenzene",
        "toxicity_score": 0.91,
        "severity_label": "Highly toxic",
        "top_tokens": "nitro, benz, ene",
        "toxicophore_hits": "nitro (score=0.42)",
        "analysis": {
            "structural_analysis": (
                "Nitrobenzene contains a benzene ring with a nitro (-NO₂) "
                "substituent. The molecule has the structure C₆H₅NO₂. Key "
                "features: aromatic ring system providing metabolic stability, "
                "and a strongly electron-withdrawing nitro group."
            ),
            "toxicophore_identification": (
                "CONFIRMED TOXICOPHORE: The nitro group (-NO₂) is a "
                "well-established toxicophore. Aromatic nitro compounds are "
                "known to undergo nitroreduction to form reactive nitroso "
                "and hydroxylamine intermediates. The combination of aromatic "
                "ring + nitro group creates a mutagenic/carcinogenic alert "
                "(Benigni-Bossa structural alert SA_28)."
            ),
            "mechanism_of_action": (
                "1. Nitroreduction by CYP450 reductases → nitrosobenzene → "
                "phenylhydroxylamine → reactive metabolites.\n"
                "2. Phenylhydroxylamine causes methemoglobinemia by oxidizing "
                "Fe²⁺ in hemoglobin to Fe³⁺.\n"
                "3. Reactive nitroso intermediate forms covalent adducts with "
                "proteins and DNA → genotoxicity.\n"
                "4. Depletes glutathione (GSH) → oxidative stress."
            ),
            "biological_pathways": (
                "- NR-AhR: Aryl hydrocarbon receptor activation (confirmed)\n"
                "- SR-ARE: Antioxidant response element / Nrf2 pathway (oxidative stress)\n"
                "- SR-p53: p53 tumor suppressor activation (DNA damage response)\n"
                "- GO:0006979 — response to oxidative stress\n"
                "- GO:0006281 — DNA repair\n"
                "- GO:0019825 — oxygen binding (hemoglobin disruption)"
            ),
            "organ_toxicity": (
                "- BLOOD: Methemoglobinemia (primary acute toxicity) — converts "
                "hemoglobin to methemoglobin, reducing oxygen transport.\n"
                "- LIVER: Hepatotoxicity via reactive metabolite formation "
                "and GSH depletion.\n"
                "- SPLEEN: Splenic congestion from destruction of damaged "
                "red blood cells.\n"
                "- CENTRAL NERVOUS SYSTEM: Headache, dizziness, cyanosis "
                "from oxygen deprivation."
            ),
            "confidence": (
                "HIGH — Nitrobenzene is one of the most extensively studied "
                "aromatic nitro compounds. All mechanisms listed are well-"
                "documented in IARC, EPA, and ATSDR assessments. The model's "
                "high attention to the 'nitro' token aligns with the actual "
                "toxicophore."
            ),
            "verdict": (
                "TOXIC (confirmed). Nitrobenzene is classified as Category 3 "
                "acute toxicity (GHS). LD50 oral rat: 640 mg/kg. Primary risk "
                "is methemoglobinemia via dermal absorption. Known carcinogen "
                "in rodents (NTP, 2002). The model's P(toxic)=0.91 is "
                "well-calibrated for this compound."
            ),
        },
    },

    # ── Exemplar 2: Formaldehyde (simple aldehyde) ──────────────────
    {
        "iupac_name": "methanal",
        "toxicity_score": 0.85,
        "severity_label": "Highly toxic",
        "top_tokens": "meth, an, al",
        "toxicophore_hits": "",
        "analysis": {
            "structural_analysis": (
                "Methanal (formaldehyde, HCHO) is the simplest aldehyde. "
                "It contains a carbonyl group (C=O) bonded to hydrogen. "
                "Key features: highly reactive electrophilic carbonyl carbon, "
                "small molecular size enabling cross-linking of biomolecules."
            ),
            "toxicophore_identification": (
                "CONFIRMED TOXICOPHORE: The aldehyde functional group (-CHO) "
                "is a reactive electrophile that readily forms Schiff bases "
                "with primary amines in proteins and DNA bases. Formaldehyde "
                "is a potent cross-linking agent — this is the basis of its "
                "use as a fixative and also its toxicity."
            ),
            "mechanism_of_action": (
                "1. Direct electrophilic attack: carbonyl carbon reacts with "
                "nucleophilic amino groups (-NH₂) on lysine residues in "
                "proteins and exocyclic amines on DNA bases.\n"
                "2. Forms DNA-protein crosslinks (DPCs) and DNA-DNA crosslinks "
                "→ genotoxicity and mutagenesis.\n"
                "3. Induces squamous cell carcinoma in nasal epithelium via "
                "chronic irritation + genotoxicity.\n"
                "4. GSH conjugation is primary detoxification (formaldehyde "
                "dehydrogenase / ADH5)."
            ),
            "biological_pathways": (
                "- SR-p53: p53 activation from DNA crosslinks (confirmed)\n"
                "- SR-ARE: Oxidative stress response / Nrf2 pathway\n"
                "- GO:0006974 — cellular response to DNA damage stimulus\n"
                "- GO:0006281 — DNA repair\n"
                "- GO:0006954 — inflammatory response (tissue irritation)"
            ),
            "organ_toxicity": (
                "- RESPIRATORY SYSTEM: Primary target — nasal cavity irritation, "
                "nasopharyngeal carcinoma (IARC Group 1 carcinogen).\n"
                "- SKIN: Contact dermatitis, sensitization.\n"
                "- EYES: Severe irritation, lacrimation.\n"
                "- BONE MARROW: Leukemia risk (epidemiological evidence, "
                "IARC 2012)."
            ),
            "confidence": (
                "HIGH — Formaldehyde is classified as IARC Group 1 (confirmed "
                "human carcinogen). Extensive toxicological data from NTP, "
                "EPA, ATSDR. The model correctly identifies it as highly toxic."
            ),
            "verdict": (
                "TOXIC (confirmed). Methanal/formaldehyde is one of the most "
                "well-documented toxic chemicals. IARC Group 1 carcinogen. "
                "LD50 oral rat: 100 mg/kg. Inhalation LC50: 578 mg/m³ (4h, "
                "rat). The model's P(toxic)=0.85 appropriately reflects high "
                "toxicity."
            ),
        },
    },

    # ── Exemplar 3: Ethanol (common non-toxic / low-toxicity) ──────
    {
        "iupac_name": "ethanol",
        "toxicity_score": 0.28,
        "severity_label": "Unlikely toxic",
        "top_tokens": "eth, an, ol",
        "toxicophore_hits": "",
        "analysis": {
            "structural_analysis": (
                "Ethanol (C₂H₅OH) is a simple two-carbon primary alcohol. "
                "Contains a hydroxyl group (-OH) on an aliphatic chain. "
                "No aromatic rings, no reactive electrophilic groups, no "
                "halogen substituents."
            ),
            "toxicophore_identification": (
                "NO ESTABLISHED TOXICOPHORES. The hydroxyl group on a short "
                "aliphatic chain is not a structural toxicity alert. Ethanol "
                "is metabolized by well-characterized enzymatic pathways "
                "(ADH → acetaldehyde → ALDH → acetic acid). While acetaldehyde "
                "is a reactive intermediate, it is rapidly cleared under "
                "normal metabolic conditions."
            ),
            "mechanism_of_action": (
                "1. CNS depressant: Ethanol enhances GABA-A receptor activity "
                "and inhibits NMDA receptors → sedation, impaired coordination.\n"
                "2. Metabolized by ADH to acetaldehyde (transient reactive "
                "intermediate), then by ALDH to acetic acid.\n"
                "3. At chronic high doses: acetaldehyde accumulation → protein "
                "adducts, oxidative stress, hepatic steatosis.\n"
                "4. Acute toxicity only at very high doses (LD50 ~7,000 mg/kg "
                "oral rat)."
            ),
            "biological_pathways": (
                "- GO:0007268 — chemical synaptic transmission (GABAergic)\n"
                "- GO:0006069 — ethanol oxidation (ADH/ALDH pathway)\n"
                "- At chronic doses only: SR-ARE, NR-PPAR-gamma activation\n"
                "- No acute pathway activation at normal exposure levels"
            ),
            "organ_toxicity": (
                "- CNS: Acute depression at high doses (intoxication). "
                "Chronic: neurodegeneration.\n"
                "- LIVER: Chronic exposure → fatty liver → cirrhosis → "
                "hepatocellular carcinoma. NOT acutely hepatotoxic.\n"
                "- Generally low acute toxicity. IARC Group 1 carcinogen "
                "only in the context of chronic alcohol consumption."
            ),
            "confidence": (
                "HIGH — Ethanol is among the most thoroughly studied compounds. "
                "The model's low P(toxic)=0.28 correctly reflects that ethanol "
                "has low acute toxicity despite chronic health risks."
            ),
            "verdict": (
                "UNLIKELY TOXIC (at normal exposure). Ethanol has very low "
                "acute toxicity (LD50 7,060 mg/kg oral rat — practically "
                "non-toxic by Hodge-Sterner classification). The model's "
                "P(toxic)=0.28 is appropriate. Chronic toxicity from alcohol "
                "abuse is a separate epidemiological question, not captured "
                "by binary toxicity models."
            ),
        },
    },

    # ── Exemplar 4: Benzo[a]pyrene (PAH/carcinogen) ────────────────
    {
        "iupac_name": "benzo[a]pyrene",
        "toxicity_score": 0.94,
        "severity_label": "Highly toxic",
        "top_tokens": "benz, o, pyr, ene",
        "toxicophore_hits": "",
        "analysis": {
            "structural_analysis": (
                "Benzo[a]pyrene (BaP) is a polycyclic aromatic hydrocarbon "
                "(PAH) consisting of five fused benzene rings. Molecular "
                "formula C₂₀H₁₂. Key features: extended planar aromatic "
                "system, bay region between rings creating a reactive site "
                "for metabolic activation."
            ),
            "toxicophore_identification": (
                "CONFIRMED TOXICOPHORE: The 'bay region' of BaP is the "
                "structural toxicophore. CYP1A1/1B1 metabolizes BaP to "
                "benzo[a]pyrene-7,8-diol-9,10-epoxide (BPDE), a highly "
                "reactive diol-epoxide that intercalates DNA and forms "
                "covalent adducts at guanine N2. This is a textbook example "
                "of metabolic activation of a procarcinogen."
            ),
            "mechanism_of_action": (
                "1. BaP binds AhR receptor → translocates to nucleus → "
                "induces CYP1A1/CYP1B1 expression.\n"
                "2. CYP enzymes oxidize BaP → BaP-7,8-epoxide → BaP-7,8-diol "
                "→ BPDE (ultimate carcinogen).\n"
                "3. BPDE forms covalent DNA adducts (preferentially G→T "
                "transversions) → mutations in oncogenes (KRAS) and tumor "
                "suppressors (TP53).\n"
                "4. Also generates ROS during metabolic activation → "
                "oxidative DNA damage (8-oxo-dG)."
            ),
            "biological_pathways": (
                "- NR-AhR: Direct AhR agonist — primary activation pathway\n"
                "- SR-p53: p53 activation from DNA adducts\n"
                "- SR-ARE: Nrf2/ARE oxidative stress response\n"
                "- GO:0006805 — xenobiotic metabolic process\n"
                "- GO:0006281 — DNA repair (NER pathway activated)\n"
                "- GO:0070059 — intrinsic apoptotic signaling by p53"
            ),
            "organ_toxicity": (
                "- LUNG: Primary target — bronchogenic carcinoma from "
                "inhalation. BaP is present in cigarette smoke and vehicle "
                "exhaust.\n"
                "- SKIN: Skin carcinoma from dermal exposure.\n"
                "- GI TRACT: Stomach/colon cancer from ingestion (grilled food).\n"
                "- IMMUNE SYSTEM: Immunosuppression via AhR-mediated "
                "disruption of immune cell differentiation."
            ),
            "confidence": (
                "HIGH — BaP is the prototype carcinogenic PAH. IARC Group 1 "
                "(confirmed human carcinogen). One of the most extensively "
                "studied environmental carcinogens with well-established "
                "mechanism."
            ),
            "verdict": (
                "TOXIC (confirmed carcinogen). IARC Group 1. No safe exposure "
                "threshold established. The model's P(toxic)=0.94 correctly "
                "identifies BaP as highly toxic. This is a prototypical example "
                "of metabolic activation converting a stable PAH into a potent "
                "DNA-damaging agent."
            ),
        },
    },

    # ── Exemplar 5: Aspirin (moderate — context-dependent) ─────────
    {
        "iupac_name": "2-(acetyloxy)benzoic acid",
        "toxicity_score": 0.42,
        "severity_label": "Unlikely toxic",
        "top_tokens": "acet, yl, oxy, benz, oic, acid",
        "toxicophore_hits": "",
        "analysis": {
            "structural_analysis": (
                "2-(acetyloxy)benzoic acid (aspirin/acetylsalicylic acid) "
                "contains a benzene ring with two substituents: a carboxylic "
                "acid (-COOH) at position 1 and an acetyloxy (ester) group "
                "(-OCOCH₃) at position 2. Molecular formula C₉H₈O₄."
            ),
            "toxicophore_identification": (
                "NO ESTABLISHED TOXICOPHORES for typical dose. The acetyl "
                "ester is the pharmacologically active group (irreversibly "
                "acetylates COX-1/COX-2). The benzoic acid core is common "
                "in many approved drugs; it is not a structural toxicity "
                "alert. At very high doses, salicylate metabolites cause "
                "uncoupling of oxidative phosphorylation."
            ),
            "mechanism_of_action": (
                "1. Therapeutic: Irreversible acetylation of Ser530 on COX-1 "
                "→ inhibits thromboxane A₂ synthesis → antiplatelet effect.\n"
                "2. Toxic (overdose): Salicylate uncouples oxidative "
                "phosphorylation in mitochondria → metabolic acidosis.\n"
                "3. GI toxicity: COX-1 inhibition reduces protective "
                "prostaglandin E₂ in stomach → gastric ulceration.\n"
                "4. Reye syndrome risk in children with viral infections "
                "(mechanism poorly understood)."
            ),
            "biological_pathways": (
                "- GO:0006693 — prostaglandin metabolic process (COX inhibition)\n"
                "- GO:0019370 — leukotriene biosynthetic process (redirected)\n"
                "- At toxic doses: mitochondrial dysfunction, metabolic acidosis\n"
                "- NR-PPAR-gamma: Weak activation (anti-inflammatory)"
            ),
            "organ_toxicity": (
                "- GI TRACT: Gastric erosion and ulceration (therapeutic doses). "
                "This is a side effect, not acute toxicity.\n"
                "- KIDNEY: Renal impairment at high chronic doses (analgesic "
                "nephropathy).\n"
                "- BLOOD: Antiplatelet effect (therapeutic but can cause "
                "bleeding risk).\n"
                "- LIVER: Reye syndrome in children (rare, pediatric context)."
            ),
            "confidence": (
                "MEDIUM — Aspirin has a complex safety profile. It is an "
                "essential medicine (WHO list) with well-characterized "
                "pharmacology. The model's moderate P(toxic)=0.42 reflects "
                "the dual nature: safe at therapeutic doses, toxic at overdose."
            ),
            "verdict": (
                "UNLIKELY TOXIC at therapeutic doses. Aspirin is one of the "
                "most widely used OTC medications. LD50 oral rat: 200 mg/kg "
                "(moderate acute toxicity). The model's P(toxic)=0.42 is "
                "reasonable — it captures the dose-dependent toxicity profile. "
                "Context-dependent: safe drug at recommended doses, but toxic "
                "in overdose."
            ),
        },
    },
]


# ──────────────────────────────────────────────────────────────────────
# Query Template
# ──────────────────────────────────────────────────────────────────────

QUERY_TEMPLATE = """\
MOLECULE: {iupac_name}
TOXICITY SCORE: {toxicity_score:.3f} ({severity_label})
HIGH-ATTENTION TOKENS: {top_tokens}
TOXICOPHORE PATTERN MATCHES: {toxicophore_hits}

Provide a complete chain-of-thought toxicity analysis following this exact \
structure. Be specific and cite known mechanisms where possible.

1. STRUCTURAL ANALYSIS:
2. TOXICOPHORE IDENTIFICATION:
3. MECHANISM OF ACTION:
4. BIOLOGICAL PATHWAYS:
5. ORGAN TOXICITY:
6. CONFIDENCE:
7. VERDICT:
"""


# ──────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────

def build_few_shot_prompt(
    iupac_name: str,
    toxicity_score: float,
    severity_label: str,
    top_tokens: str = "",
    toxicophore_hits: str = "",
    num_exemplars: int = 3,
    exemplar_indices: Optional[List[int]] = None,
) -> str:
    """Build the full few-shot CoT prompt for a molecule.

    Selects exemplars strategically:
    - If the molecule is highly toxic (score > 0.7): include toxic exemplars
    - If the molecule is non-toxic (score < 0.3): include the non-toxic exemplar
    - Default: mix of toxic and non-toxic for balanced reasoning

    Args:
        iupac_name: IUPAC name of the query molecule.
        toxicity_score: P(toxic) from Phase 1.
        severity_label: Severity label from Phase 1.
        top_tokens: Comma-separated top-attention tokens.
        toxicophore_hits: Toxicophore pattern matches.
        num_exemplars: Number of few-shot examples (default 3).
        exemplar_indices: Explicit indices into FEW_SHOT_EXEMPLARS.
            If None, selects automatically based on toxicity_score.

    Returns:
        The complete few-shot prompt string.
    """
    if exemplar_indices is not None:
        selected = [FEW_SHOT_EXEMPLARS[i] for i in exemplar_indices
                     if i < len(FEW_SHOT_EXEMPLARS)]
    else:
        # Automatic selection based on query toxicity
        if toxicity_score >= 0.7:
            # Highly toxic → show mostly toxic exemplars
            selected = [
                FEW_SHOT_EXEMPLARS[0],  # nitrobenzene
                FEW_SHOT_EXEMPLARS[3],  # benzo[a]pyrene
                FEW_SHOT_EXEMPLARS[2],  # ethanol (counterexample)
            ]
        elif toxicity_score <= 0.3:
            # Likely non-toxic → show non-toxic + contrast
            selected = [
                FEW_SHOT_EXEMPLARS[2],  # ethanol
                FEW_SHOT_EXEMPLARS[4],  # aspirin
                FEW_SHOT_EXEMPLARS[0],  # nitrobenzene (contrast)
            ]
        else:
            # Borderline → balanced mix
            selected = [
                FEW_SHOT_EXEMPLARS[4],  # aspirin (borderline)
                FEW_SHOT_EXEMPLARS[0],  # nitrobenzene (toxic)
                FEW_SHOT_EXEMPLARS[2],  # ethanol (non-toxic)
            ]

    selected = selected[:num_exemplars]

    # Build few-shot examples
    examples_text = []
    for i, ex in enumerate(selected, 1):
        a = ex["analysis"]
        example = (
            f"--- EXAMPLE {i} ---\n"
            f"MOLECULE: {ex['iupac_name']}\n"
            f"TOXICITY SCORE: {ex['toxicity_score']:.3f} "
            f"({ex['severity_label']})\n"
            f"HIGH-ATTENTION TOKENS: {ex['top_tokens']}\n"
            f"TOXICOPHORE PATTERN MATCHES: {ex['toxicophore_hits'] or 'None'}\n"
            f"\n"
            f"1. STRUCTURAL ANALYSIS:\n{a['structural_analysis']}\n\n"
            f"2. TOXICOPHORE IDENTIFICATION:\n{a['toxicophore_identification']}\n\n"
            f"3. MECHANISM OF ACTION:\n{a['mechanism_of_action']}\n\n"
            f"4. BIOLOGICAL PATHWAYS:\n{a['biological_pathways']}\n\n"
            f"5. ORGAN TOXICITY:\n{a['organ_toxicity']}\n\n"
            f"6. CONFIDENCE:\n{a['confidence']}\n\n"
            f"7. VERDICT:\n{a['verdict']}\n"
        )
        examples_text.append(example)

    # Build query
    query = QUERY_TEMPLATE.format(
        iupac_name=iupac_name,
        toxicity_score=toxicity_score,
        severity_label=severity_label,
        top_tokens=top_tokens or "None",
        toxicophore_hits=toxicophore_hits or "None",
    )

    full_prompt = (
        "Here are examples of chain-of-thought toxicity analyses:\n\n"
        + "\n\n".join(examples_text)
        + "\n\n--- YOUR ANALYSIS ---\n"
        + query
    )

    return full_prompt


# ──────────────────────────────────────────────────────────────────────
# Output Parser
# ──────────────────────────────────────────────────────────────────────

# Section headers to parse from CoT output
COT_SECTIONS = [
    "STRUCTURAL ANALYSIS",
    "TOXICOPHORE IDENTIFICATION",
    "MECHANISM OF ACTION",
    "BIOLOGICAL PATHWAYS",
    "ORGAN TOXICITY",
    "CONFIDENCE",
    "VERDICT",
]


def parse_cot_response(raw_text: str) -> Dict[str, str]:
    """Parse a structured CoT response into sections.

    Extracts text between numbered section headers like:
        1. STRUCTURAL ANALYSIS: ...
        2. TOXICOPHORE IDENTIFICATION: ...

    Args:
        raw_text: Raw LLM response text.

    Returns:
        Dict mapping section names to their content.
        Missing sections will have empty string values.
    """
    result = {section: "" for section in COT_SECTIONS}

    # Build regex pattern for section extraction
    # Matches "1. STRUCTURAL ANALYSIS:" or "STRUCTURAL ANALYSIS:" etc.
    for i, section in enumerate(COT_SECTIONS):
        # Pattern: optional number + dot + section name + optional colon
        pattern = rf"(?:\d+\.\s*)?{re.escape(section)}[:\s]*\n?(.*?)(?=(?:\d+\.\s*)?(?:{'|'.join(re.escape(s) for s in COT_SECTIONS[i+1:])})[:\s]|\Z)"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            result[section] = match.group(1).strip()

    return result


def extract_functional_groups(cot_sections: Dict[str, str]) -> List[str]:
    """Extract functional groups from the STRUCTURAL ANALYSIS section only.

    Only scans the STRUCTURAL ANALYSIS section to capture groups present
    in the parent molecule's structure. Metabolite-derived groups
    (mentioned in MECHANISM OF ACTION or TOXICOPHORE IDENTIFICATION)
    are excluded to avoid conflating parent structure with metabolites.

    Returns:
        List of unique functional group names found in the parent structure.
    """
    known_groups = [
        "nitro", "amino", "hydroxyl", "carboxyl", "carbonyl", "aldehyde",
        "ketone", "ester", "ether", "epoxide", "azo", "nitroso", "azide",
        "isocyanate", "acyl halide", "acid chloride", "anhydride",
        "sulfhydryl", "thiol", "disulfide", "phosphate", "sulfonate",
        "halide", "chloro", "bromo", "fluoro", "iodo",
        "phenol", "quinone", "hydroquinone", "catechol",
        "aromatic amine", "aliphatic amine", "tertiary amine",
        "nitrile", "cyano", "nitrate", "sulfate",
        "peroxide", "hydroperoxide",
        "alkene", "alkyne", "allyl",
        "benzene", "naphthalene", "anthracene",
        "furan", "pyrrole", "pyridine", "imidazole",
        "Michael acceptor", "alkylating agent",
    ]

    # Only scan STRUCTURAL ANALYSIS for parent molecule groups
    text = cot_sections.get("STRUCTURAL ANALYSIS", "").lower()

    found = []
    for group in known_groups:
        if group.lower() in text:
            found.append(group)

    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_metabolite_groups(cot_sections: Dict[str, str]) -> List[str]:
    """Extract metabolite/intermediate functional groups from CoT analysis.

    Scans MECHANISM OF ACTION and TOXICOPHORE IDENTIFICATION for groups
    that arise from biotransformation (metabolites, reactive intermediates).

    Returns:
        List of unique metabolite-derived group names.
    """
    known_groups = [
        "nitro", "amino", "hydroxyl", "carboxyl", "carbonyl", "aldehyde",
        "ketone", "ester", "ether", "epoxide", "azo", "nitroso", "azide",
        "isocyanate", "acyl halide", "acid chloride", "anhydride",
        "sulfhydryl", "thiol", "disulfide", "phosphate", "sulfonate",
        "halide", "chloro", "bromo", "fluoro", "iodo",
        "phenol", "quinone", "hydroquinone", "catechol",
        "aromatic amine", "aliphatic amine", "tertiary amine",
        "nitrile", "cyano", "nitrate", "sulfate",
        "peroxide", "hydroperoxide",
        "alkene", "alkyne", "allyl",
        "benzene", "naphthalene", "anthracene",
        "furan", "pyrrole", "pyridine", "imidazole",
        "Michael acceptor", "alkylating agent",
    ]

    text = (
        cot_sections.get("MECHANISM OF ACTION", "") + " " +
        cot_sections.get("TOXICOPHORE IDENTIFICATION", "")
    ).lower()

    # Get parent groups to exclude them
    parent_text = cot_sections.get("STRUCTURAL ANALYSIS", "").lower()

    found = []
    for group in known_groups:
        gl = group.lower()
        if gl in text and gl not in parent_text:
            found.append(group)

    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_confidence_level(cot_sections: Dict[str, str]) -> str:
    """Extract confidence level (HIGH/MEDIUM/LOW) from CoT output."""
    confidence_text = cot_sections.get("CONFIDENCE", "").upper()
    if "HIGH" in confidence_text:
        return "HIGH"
    elif "MEDIUM" in confidence_text or "MODERATE" in confidence_text:
        return "MEDIUM"
    elif "LOW" in confidence_text:
        return "LOW"
    return "UNKNOWN"
