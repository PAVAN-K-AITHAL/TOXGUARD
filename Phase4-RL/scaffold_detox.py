"""Scaffold-aware molecule detoxification via toxicophore replacement.

Instead of generating entirely new molecules (like IUPAC-GPT does),
this module identifies KNOWN toxic substructures (toxicophores) in the
seed molecule and replaces them with safe bioisosteric groups.

This preserves the molecular scaffold and gives high Tanimoto similarity.

Example:
    nitrobenzene (-NO2 on benzene)
    -> aniline      (-NH2)  Tanimoto ~0.76
    -> phenol       (-OH)   Tanimoto ~0.73
    -> toluene      (-CH3)  Tanimoto ~0.71
    -> benzoic acid (-COOH) Tanimoto ~0.65
    -> fluorobenzene(-F)    Tanimoto ~0.70

Usage:
    detox = ScaffoldDetox(predictor, resolver, validator)
    results = detox.detoxify_smiles("O=[N+]([O-])c1ccccc1")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.DataStructs import TanimotoSimilarity
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available — scaffold detox disabled")


# ──────────────────────────────────────────────────────────────────────
# Toxicophore definitions: SMARTS pattern -> list of safe replacements
# ──────────────────────────────────────────────────────────────────────
# Each entry: (name, SMARTS_pattern, [replacement_SMARTS, ...])
# Replacements are bioisosteric groups that preserve shape but reduce toxicity.

TOXICOPHORE_REPLACEMENTS = [
    # ── Nitro group (highly toxic — methemoglobinemia, mutagenicity) ──
    (
        "nitro",
        "[N+](=O)[O-]",
        [
            ("amino", "[NH2]"),           # -NO2 -> -NH2 (aniline)
            ("hydroxyl", "[OH]"),         # -NO2 -> -OH  (phenol)
            ("fluorine", "[F]"),          # -NO2 -> -F   (fluorobenzene)
            ("methyl", "[CH3]"),          # -NO2 -> -CH3 (toluene)
            ("carboxyl", "C(=O)[OH]"),    # -NO2 -> -COOH (benzoic acid)
            ("acetamido", "[NH]C(=O)C"),  # -NO2 -> -NHCOCH3 (acetanilide)
            ("cyano", "C#N"),             # -NO2 -> -CN  (benzonitrile)
            ("hydrogen", "[H]"),          # -NO2 -> -H   (benzene)
        ],
    ),
    # ── Aromatic amine (mutagenic, carcinogenic) ──
    (
        "aromatic_amine",
        "[c][NH2]",
        [
            ("acetamide", "[c][NH]C(=O)C"),   # ArNH2 -> ArNHCOCH3
            ("hydroxyl", "[c][OH]"),            # ArNH2 -> ArOH
            ("methyl", "[c][CH3]"),             # ArNH2 -> ArCH3
        ],
    ),
    # ── Aldehyde (reactive, toxic) ──
    (
        "aldehyde",
        "[CH]=O",
        [
            ("alcohol", "[CH2][OH]"),       # -CHO -> -CH2OH
            ("carboxyl", "C(=O)[OH]"),      # -CHO -> -COOH
            ("methyl", "[CH3]"),            # -CHO -> -CH3
        ],
    ),
    # ── Epoxide (electrophilic, mutagenic) ──
    (
        "epoxide",
        "C1OC1",
        [
            ("diol", "C(O)CO"),             # epoxide -> diol
        ],
    ),
    # ── Acyl halide (reactive, toxic) ──
    (
        "acyl_chloride",
        "C(=O)Cl",
        [
            ("ester", "C(=O)OC"),           # -COCl -> -COOCH3
            ("amide", "C(=O)N"),            # -COCl -> -CONH2
            ("carboxyl", "C(=O)[OH]"),      # -COCl -> -COOH
        ],
    ),
    # ── Michael acceptor (reactive, electrophilic) ──
    (
        "michael_acceptor",
        "C=CC(=O)",
        [
            ("saturated", "CCCC(=O)"),      # remove double bond
        ],
    ),
]


@dataclass
class ScaffoldCandidate:
    """A candidate from scaffold-based detoxification."""
    iupac_name: str = ""
    smiles: str = ""
    p_toxic: float = 1.0
    tanimoto: float = 0.0
    qed: float = 0.0
    sa_score: float = 0.0
    toxicophore_removed: str = ""
    replacement_type: str = ""
    reward: float = 0.0
    delta_toxicity: float = 0.0


class ScaffoldDetox:
    """Scaffold-aware detoxification via toxicophore replacement.

    Uses RDKit substructure search to find known toxic groups,
    then replaces them with safe bioisosteric alternatives.
    Preserves the molecular scaffold for high Tanimoto similarity.

    Args:
        toxguard_predictor: Phase 1 ToxGuardPredictor.
        name_resolver: NameResolver for SMILES -> IUPAC conversion.
        molecule_validator: MoleculeValidator for property computation.
    """

    def __init__(self, toxguard_predictor, name_resolver, molecule_validator):
        self.predictor = toxguard_predictor
        self.resolver = name_resolver
        self.validator = molecule_validator

    def find_toxicophores(self, smiles: str) -> List[Tuple[str, str]]:
        """Find known toxic substructures in a molecule.

        Returns:
            List of (toxicophore_name, SMARTS_pattern) found.
        """
        if not HAS_RDKIT:
            return []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        found = []
        for name, smarts, _ in TOXICOPHORE_REPLACEMENTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found.append((name, smarts))
                logger.debug(f"Found toxicophore: {name} ({smarts}) in {smiles}")

        return found

    def generate_replacements(
        self,
        smiles: str,
        seed_p_toxic: float = 0.5,
    ) -> List[ScaffoldCandidate]:
        """Generate all possible toxicophore replacements for a molecule.

        For each toxicophore found, tries every safe bioisostere
        and returns candidates with their properties.

        Args:
            smiles: SMILES of the seed molecule.
            seed_p_toxic: P(toxic) of the seed for delta computation.

        Returns:
            List of ScaffoldCandidate, sorted by reward (descending).
        """
        if not HAS_RDKIT:
            logger.warning("RDKit not available — cannot do scaffold detox")
            return []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid seed SMILES: {smiles}")
            return []

        # Compute seed fingerprint for Tanimoto
        seed_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        candidates = []

        for tox_name, tox_smarts, replacements in TOXICOPHORE_REPLACEMENTS:
            tox_pattern = Chem.MolFromSmarts(tox_smarts)
            if tox_pattern is None or not mol.HasSubstructMatch(tox_pattern):
                continue

            logger.info(f"Toxicophore found: {tox_name} -- trying {len(replacements)} replacements")

            for repl_name, repl_smarts in replacements:
                try:
                    candidate = self._try_replacement(
                        mol, seed_fp, seed_p_toxic,
                        tox_smarts, repl_smarts,
                        tox_name, repl_name,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                except Exception as e:
                    logger.debug(
                        f"Replacement {tox_name}->{repl_name} failed: {e}"
                    )

        # Sort by reward (descending)
        candidates.sort(key=lambda c: c.reward, reverse=True)

        return candidates

    def _try_replacement(
        self,
        mol,
        seed_fp,
        seed_p_toxic: float,
        tox_smarts: str,
        repl_smarts: str,
        tox_name: str,
        repl_name: str,
    ) -> Optional[ScaffoldCandidate]:
        """Try a single toxicophore replacement and score the result."""
        tox_pattern = Chem.MolFromSmarts(tox_smarts)
        repl_mol = Chem.MolFromSmiles(repl_smarts)

        if tox_pattern is None or repl_mol is None:
            return None

        # Handle hydrogen replacement specially
        if repl_smarts == "[H]":
            # Remove the toxicophore entirely
            new_mol = AllChem.DeleteSubstructs(mol, tox_pattern)
        else:
            # Replace toxicophore with bioisostere
            new_mol = AllChem.ReplaceSubstructs(
                mol, tox_pattern, repl_mol,
                replaceAll=True,
            )
            if isinstance(new_mol, tuple):
                new_mol = new_mol[0]

        if new_mol is None:
            return None

        # Sanitize
        try:
            Chem.SanitizeMol(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol)
            # Re-parse to ensure valid
            check = Chem.MolFromSmiles(new_smiles)
            if check is None:
                return None
            new_smiles = Chem.MolToSmiles(check)
        except Exception:
            return None

        # Compute Tanimoto similarity
        new_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(new_smiles), 2, nBits=2048
        )
        tanimoto = TanimotoSimilarity(seed_fp, new_fp)

        # Compute QED and SA
        validation = self.validator.validate_candidate(
            candidate_smiles=new_smiles,
            seed_smiles=Chem.MolToSmiles(mol),
        )
        if not validation.valid:
            return None

        # Get IUPAC name
        iupac_name = self.resolver.smiles_to_iupac(new_smiles)
        if not iupac_name:
            # Use SMILES as fallback name
            iupac_name = new_smiles

        # Score toxicity with Phase 1
        try:
            prediction = self.predictor.predict(
                iupac_name,
                return_attention=False,
                return_egnn_vector=False,
            )
            p_toxic = prediction.toxicity_score
        except Exception as e:
            logger.debug(f"Toxicity prediction failed for {iupac_name}: {e}")
            p_toxic = 0.5

        # Compute reward (balanced)
        detox_reward = 1.0 - p_toxic
        sim_reward = tanimoto
        qed_reward = validation.qed
        sa_reward = validation.sa_score

        reward = (
            0.30 * detox_reward
            + 0.30 * sim_reward
            + 0.20 * qed_reward
            + 0.15 * sa_reward
            + 0.5  # validity bonus
        )

        delta_tox = seed_p_toxic - p_toxic

        candidate = ScaffoldCandidate(
            iupac_name=iupac_name,
            smiles=new_smiles,
            p_toxic=p_toxic,
            tanimoto=tanimoto,
            qed=validation.qed,
            sa_score=validation.sa_score,
            toxicophore_removed=tox_name,
            replacement_type=repl_name,
            reward=reward,
            delta_toxicity=delta_tox,
        )

        logger.info(
            f"  Scaffold candidate: {tox_name}->{repl_name} | "
            f"{iupac_name} | P={p_toxic:.3f} Tan={tanimoto:.3f}"
        )

        return candidate
