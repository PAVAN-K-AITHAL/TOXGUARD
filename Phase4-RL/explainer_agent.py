"""Explainer Agent: generates human-readable explanations for detoxification.

Analyzes structural differences between seed and candidate molecules
to explain WHY the detoxified molecule is predicted to be less toxic.

Uses purely analytical methods — no model training required:
    1. Functional group analysis (added/removed groups via SMARTS)
    2. Property change interpretation (LogP, PSA, MW implications)
    3. Toxicophore removal detection
    4. Pharmacological effect mapping

The explanation is integrated into the detoxification report.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, AllChem, rdFMCS
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ── Known functional groups and their pharmacological effects ─────────
FUNCTIONAL_GROUPS = {
    "hydroxyl": {
        "smarts": "[OX2H;!$([OX2H]-[CX3]=O)]",
        "name": "hydroxyl (-OH)",
        "effect": "increases polarity and water solubility, reduces membrane permeability",
    },
    "carboxyl": {
        "smarts": "[CX3](=O)[OX2H1]",
        "name": "carboxyl (-COOH)",
        "effect": "ionizable at physiological pH, increases hydrophilicity and renal clearance",
    },
    "amine_primary": {
        "smarts": "[NX3;H2;!$(NC=O)]",
        "name": "primary amine (-NH2)",
        "effect": "increases water solubility and hydrogen bonding capacity",
    },
    "amine_secondary": {
        "smarts": "[NX3;H1;!$(NC=O)]",
        "name": "secondary amine (-NH-)",
        "effect": "modulates basicity and receptor binding",
    },
    "amide": {
        "smarts": "[NX3][CX3](=[OX1])",
        "name": "amide (-CONH-)",
        "effect": "increases metabolic stability and reduces toxicity via H-bond formation",
    },
    "nitro": {
        "smarts": "[NX3+](=O)[O-]",
        "name": "nitro (-NO2)",
        "effect": "TOXIC: forms reactive nitroso/hydroxylamine metabolites causing DNA damage",
        "is_toxic": True,
    },
    "epoxide": {
        "smarts": "[OX2r3]",
        "name": "epoxide",
        "effect": "TOXIC: highly electrophilic, alkylates DNA and proteins",
        "is_toxic": True,
    },
    "aldehyde": {
        "smarts": "[CX3H1](=O)",
        "name": "aldehyde (-CHO)",
        "effect": "TOXIC: reactive electrophile that cross-links proteins",
        "is_toxic": True,
    },
    "sulfonyl": {
        "smarts": "[SX4](=O)(=O)",
        "name": "sulfonyl (-SO2-)",
        "effect": "increases polarity and metabolic stability",
    },
    "sulfonic_acid": {
        "smarts": "[SX4](=O)(=O)[OX2H]",
        "name": "sulfonic acid (-SO3H)",
        "effect": "strongly hydrophilic, poor membrane penetration, lower bioavailability",
    },
    "halide_F": {
        "smarts": "[F]",
        "name": "fluorine (-F)",
        "effect": "modulates metabolic stability and lipophilicity",
    },
    "halide_Cl": {
        "smarts": "[Cl]",
        "name": "chlorine (-Cl)",
        "effect": "increases lipophilicity, may contribute to bioaccumulation",
    },
    "halide_Br": {
        "smarts": "[Br]",
        "name": "bromine (-Br)",
        "effect": "increases molecular weight and lipophilicity",
    },
    "ether": {
        "smarts": "[OD2]([#6])[#6]",
        "name": "ether (-O-)",
        "effect": "increases conformational flexibility and metabolic susceptibility",
    },
    "methoxy": {
        "smarts": "[OX2][CH3]",
        "name": "methoxy (-OCH3)",
        "effect": "modulates electron density and metabolic profile",
    },
    "cyano": {
        "smarts": "[CX2]#[NX1]",
        "name": "nitrile (-CN)",
        "effect": "compact polar group, may be metabolized to amide (detoxification)",
    },
    "acyl_halide": {
        "smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
        "name": "acyl halide (-COX)",
        "effect": "TOXIC: highly reactive acylating agent",
        "is_toxic": True,
    },
    "hydrazine": {
        "smarts": "[NX3][NX3]",
        "name": "hydrazine (-NH-NH-)",
        "effect": "TOXIC: generates free radicals, causes oxidative stress",
        "is_toxic": True,
    },
    "azide": {
        "smarts": "[NX1]=[NX2]=[NX1]",
        "name": "azide (-N3)",
        "effect": "TOXIC: inhibits cytochrome c oxidase, disrupts cellular respiration",
        "is_toxic": True,
    },
    "aromatic_ring": {
        "smarts": "c1ccccc1",
        "name": "benzene ring",
        "effect": "lipophilic scaffold; fused aromatics increase carcinogenicity risk",
    },
    "carboxamide": {
        "smarts": "[CX3](=O)[NX3H2]",
        "name": "carboxamide (-CONH2)",
        "effect": "increases polarity, hydrogen bonding, and metabolic stability",
    },
}


@dataclass
class StructuralChange:
    """A detected structural change between seed and candidate."""
    change_type: str          # "added", "removed", "modified"
    group_name: str           # Human-readable group name
    effect: str               # Pharmacological effect
    is_toxic_group: bool = False


@dataclass
class DetoxExplanation:
    """Complete explanation of why the candidate is less toxic."""
    seed_smiles: str = ""
    candidate_smiles: str = ""
    structural_changes: List[StructuralChange] = field(default_factory=list)
    property_explanations: List[str] = field(default_factory=list)
    toxicophore_explanations: List[str] = field(default_factory=list)
    mcs_explanation: str = ""
    overall_summary: str = ""

    def format_report(self) -> str:
        """Format as a human-readable report section."""
        lines = [
            "",
            "  DETOXIFICATION EXPLANATION",
            "-" * 60,
        ]

        # Structural changes
        if self.structural_changes:
            lines.append("  Structural modifications:")
            for change in self.structural_changes:
                icon = "[-]" if change.change_type == "removed" else "[+]"
                if change.change_type == "removed" and change.is_toxic_group:
                    icon = "[x]"  # Toxic group removed
                lines.append(f"    {icon} {change.change_type.upper()}: {change.group_name}")
                lines.append(f"        → {change.effect}")
            lines.append("")

        # Toxicophore analysis
        if self.toxicophore_explanations:
            lines.append("  Toxicophore analysis:")
            for exp in self.toxicophore_explanations:
                lines.append(f"    • {exp}")
            lines.append("")

        # Property-based explanations
        if self.property_explanations:
            lines.append("  Property-based rationale:")
            for exp in self.property_explanations:
                lines.append(f"    • {exp}")
            lines.append("")

        # MCS / scaffold preservation
        if self.mcs_explanation:
            lines.append(f"  Scaffold preservation: {self.mcs_explanation}")
            lines.append("")

        # Overall summary
        if self.overall_summary:
            lines.append(f"  Summary: {self.overall_summary}")

        return "\n".join(lines)


class ExplainerAgent:
    """Generates human-readable explanations for molecule detoxification.

    Analyzes the structural differences between the seed (toxic) and
    candidate (detoxified) molecules to explain why toxicity decreased.

    No training required — uses RDKit substructure matching and
    pharmacological knowledge base.
    """

    def __init__(self):
        # Pre-compile SMARTS patterns
        self._patterns = {}
        if HAS_RDKIT:
            for name, info in FUNCTIONAL_GROUPS.items():
                mol = Chem.MolFromSmarts(info["smarts"])
                if mol is not None:
                    self._patterns[name] = mol

    def explain(
        self,
        seed_smiles: str,
        candidate_smiles: str,
        seed_p_toxic: float,
        candidate_p_toxic: float,
        source: str = "gpt",
    ) -> DetoxExplanation:
        """Generate explanation for why candidate is less toxic than seed.

        Args:
            seed_smiles: SMILES of the original toxic molecule.
            candidate_smiles: SMILES of the proposed detoxified molecule.
            seed_p_toxic: Predicted P(toxic) of seed.
            candidate_p_toxic: Predicted P(toxic) of candidate.
            source: "scaffold" or "gpt" — how the candidate was generated.

        Returns:
            DetoxExplanation with structured explanation.
        """
        explanation = DetoxExplanation(
            seed_smiles=seed_smiles,
            candidate_smiles=candidate_smiles,
        )

        if not HAS_RDKIT:
            explanation.overall_summary = "RDKit not available for structural analysis."
            return explanation

        seed_mol = Chem.MolFromSmiles(seed_smiles)
        cand_mol = Chem.MolFromSmiles(candidate_smiles)

        if seed_mol is None or cand_mol is None:
            explanation.overall_summary = "Could not parse one or both molecules."
            return explanation

        # ── 1. Functional group analysis ──────────────────────────
        seed_groups = self._detect_groups(seed_mol)
        cand_groups = self._detect_groups(cand_mol)

        # Find added and removed groups
        for name, info in FUNCTIONAL_GROUPS.items():
            seed_count = seed_groups.get(name, 0)
            cand_count = cand_groups.get(name, 0)

            if cand_count > seed_count:
                # Group was added
                explanation.structural_changes.append(StructuralChange(
                    change_type="added",
                    group_name=info["name"],
                    effect=info["effect"],
                    is_toxic_group=info.get("is_toxic", False),
                ))
            elif seed_count > cand_count:
                # Group was removed
                explanation.structural_changes.append(StructuralChange(
                    change_type="removed",
                    group_name=info["name"],
                    effect=info["effect"],
                    is_toxic_group=info.get("is_toxic", False),
                ))

        # ── 2. Toxicophore analysis ───────────────────────────────
        toxic_removed = [
            c for c in explanation.structural_changes
            if c.change_type == "removed" and c.is_toxic_group
        ]
        toxic_added = [
            c for c in explanation.structural_changes
            if c.change_type == "added" and c.is_toxic_group
        ]

        if toxic_removed:
            names = ", ".join(c.group_name for c in toxic_removed)
            explanation.toxicophore_explanations.append(
                f"Toxic group(s) removed: {names}. "
                f"Eliminating known toxicophores directly reduces mutagenic/carcinogenic potential."
            )

        if not toxic_removed and not toxic_added:
            explanation.toxicophore_explanations.append(
                "No known toxicophores were present in either molecule. "
                "Toxicity reduction is attributed to overall scaffold modification "
                "and physicochemical property changes."
            )

        # ── 3. Property-based explanations ────────────────────────
        seed_logp = Descriptors.MolLogP(seed_mol)
        cand_logp = Descriptors.MolLogP(cand_mol)
        seed_psa = Descriptors.TPSA(seed_mol)
        cand_psa = Descriptors.TPSA(cand_mol)
        seed_mw = Descriptors.MolWt(seed_mol)
        cand_mw = Descriptors.MolWt(cand_mol)
        seed_hbd = Lipinski.NumHDonors(seed_mol)
        cand_hbd = Lipinski.NumHDonors(cand_mol)

        # LogP interpretation
        delta_logp = cand_logp - seed_logp
        if delta_logp < -0.5:
            explanation.property_explanations.append(
                f"LogP decreased ({seed_logp:.2f}→{cand_logp:.2f}): "
                f"reduced lipophilicity lowers membrane permeability and bioaccumulation, "
                f"decreasing systemic exposure to toxic effects."
            )
        elif delta_logp > 0.5:
            explanation.property_explanations.append(
                f"LogP increased ({seed_logp:.2f}→{cand_logp:.2f}): "
                f"higher lipophilicity, but toxicity reduction comes from other structural changes."
            )

        # PSA interpretation
        delta_psa = cand_psa - seed_psa
        if delta_psa > 10:
            explanation.property_explanations.append(
                f"Polar surface area increased ({seed_psa:.1f}→{cand_psa:.1f} Å²): "
                f"enhanced polarity improves aqueous solubility and reduces passive membrane diffusion, "
                f"limiting exposure to intracellular targets."
            )
        elif delta_psa < -10:
            explanation.property_explanations.append(
                f"Polar surface area decreased ({seed_psa:.1f}→{cand_psa:.1f} Å²): "
                f"reduced polarity may alter distribution profile."
            )

        # H-bond donors
        if cand_hbd > seed_hbd:
            explanation.property_explanations.append(
                f"H-bond donors increased ({seed_hbd}→{cand_hbd}): "
                f"enhanced hydrogen bonding capacity promotes aqueous solubility "
                f"and may reduce non-specific binding to hydrophobic targets."
            )

        # MW interpretation
        delta_mw = cand_mw - seed_mw
        if abs(delta_mw) > 20:
            direction = "increased" if delta_mw > 0 else "decreased"
            explanation.property_explanations.append(
                f"Molecular weight {direction} ({seed_mw:.1f}→{cand_mw:.1f}): "
                f"structural modification alters absorption and distribution profile."
            )

        # ── 4. MCS / scaffold preservation ────────────────────────
        try:
            mcs = rdFMCS.FindMCS(
                [seed_mol, cand_mol],
                timeout=3,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                ringMatchesRingOnly=True,
            )
            max_atoms = max(seed_mol.GetNumHeavyAtoms(), cand_mol.GetNumHeavyAtoms())
            if max_atoms > 0:
                ratio = mcs.numAtoms / max_atoms
                atoms_changed = max_atoms - mcs.numAtoms
                explanation.mcs_explanation = (
                    f"{mcs.numAtoms}/{max_atoms} heavy atoms shared "
                    f"({ratio:.0%} scaffold preserved, {atoms_changed} atoms modified). "
                    f"{'Minimal' if atoms_changed <= 3 else 'Moderate' if atoms_changed <= 6 else 'Significant'} "
                    f"structural perturbation."
                )
        except Exception:
            pass

        # ── 5. Overall summary ─────────────────────────────────────
        delta_tox = seed_p_toxic - candidate_p_toxic
        explanation.overall_summary = self._generate_summary(
            explanation, delta_tox, source, seed_p_toxic, candidate_p_toxic
        )

        return explanation

    def _detect_groups(self, mol) -> Dict[str, int]:
        """Count functional groups in a molecule."""
        counts = {}
        for name, pattern in self._patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                counts[name] = len(matches)
        return counts

    def _generate_summary(
        self,
        explanation: DetoxExplanation,
        delta_tox: float,
        source: str,
        seed_p: float,
        cand_p: float,
    ) -> str:
        """Generate a one-paragraph summary of the detoxification."""
        parts = []

        # Opening
        reduction_pct = (delta_tox / seed_p * 100) if seed_p > 0 else 0
        parts.append(
            f"Toxicity reduced by {delta_tox:.3f} ({reduction_pct:.1f}% relative reduction)."
        )

        # Mechanism
        toxic_removed = [c for c in explanation.structural_changes
                        if c.change_type == "removed" and c.is_toxic_group]
        polar_added = [c for c in explanation.structural_changes
                      if c.change_type == "added" and not c.is_toxic_group]

        if toxic_removed:
            names = ", ".join(c.group_name for c in toxic_removed)
            parts.append(f"Primary mechanism: removal of toxic group(s) ({names}).")
        elif polar_added:
            names = ", ".join(c.group_name for c in polar_added[:3])
            parts.append(f"Primary mechanism: introduction of polar group(s) ({names}) "
                        f"reducing lipophilicity and membrane penetration.")
        else:
            parts.append("Primary mechanism: scaffold modification altering "
                        "electronic and steric properties.")

        # Source acknowledgment
        if source == "scaffold":
            parts.append("Generated via bioisosteric replacement of known toxicophore.")
        else:
            parts.append("Generated by PPO-trained IUPAC-GPT through learned structural optimization.")

        return " ".join(parts)
