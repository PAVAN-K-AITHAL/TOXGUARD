"""RDKit-based molecule validation and property scoring for Phase 4.

Validates generated IUPAC names (after SMILES conversion) and computes
multi-property scores for the PPO reward function:
    - Chemical validity (RDKit parse + valence check)
    - Tanimoto similarity (Morgan fingerprints)
    - QED (Quantitative Estimate of Drug-likeness)
    - SA Score (Synthetic Accessibility)
    - Lipinski's Rule of Five

Usage:
    validator = MoleculeValidator()
    result = validator.validate_candidate(
        candidate_smiles="c1ccc(N)cc1",
        seed_smiles="[O-][N+](=O)c1ccccc1",
    )
    print(result)
    # ValidationResult(valid=True, tanimoto=0.78, qed=0.62, sa=2.1, ...)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        Lipinski,
        QED,
        DataStructs,
    )
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — molecule validation will be limited")


# ──────────────────────────────────────────────────────────────────────
# SA Score (Synthetic Accessibility)
# ──────────────────────────────────────────────────────────────────────

def _compute_sa_score(mol) -> float:
    """Compute synthetic accessibility score (1=easy, 10=hard).

    Uses the Ertl-Schuffenhauer SA score algorithm.
    Returns normalized score in [0, 1] where 1 = easy to synthesize.
    """
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys
        # Try to import the SA score module from RDKit contrib
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        raw_score = sascorer.calculateScore(mol)
        # Normalize: raw is 1-10, convert to 0-1 where 1=easy
        return (10.0 - raw_score) / 9.0
    except (ImportError, Exception) as e:
        # Fallback: use a simple heuristic based on ring count and heteroatoms
        logger.debug(f"SA scorer not available, using heuristic: {e}")
        ring_count = Chem.rdMolDescriptors.CalcNumRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        # Simple heuristic: fewer rings and atoms = easier
        heuristic = max(0.0, 1.0 - (ring_count * 0.1 + heavy_atoms * 0.01))
        return max(0.0, min(1.0, heuristic))


# ──────────────────────────────────────────────────────────────────────
# Validation Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result from molecule validation pipeline."""

    # Basic validity
    valid: bool = False
    smiles: str = ""
    canonical_smiles: str = ""
    error: str = ""

    # Properties (only set if valid)
    molecular_weight: float = 0.0
    heavy_atom_count: int = 0
    num_rings: int = 0

    # Comparison to seed (only set if seed provided)
    tanimoto: float = 0.0

    # Drug-likeness scores
    qed: float = 0.0
    sa_score: float = 0.0  # normalized [0, 1], 1 = easy

    # Lipinski Rule of Five
    lipinski_violations: int = 0
    lipinski_details: Dict[str, bool] = field(default_factory=dict)

    # Toxicity (filled by external caller)
    toxicity_score: float = -1.0  # -1 = not computed
    is_less_toxic: bool = False

    def __repr__(self):
        if not self.valid:
            return f"ValidationResult(valid=False, error={self.error!r})"
        return (
            f"ValidationResult(valid=True, tanimoto={self.tanimoto:.3f}, "
            f"qed={self.qed:.3f}, sa={self.sa_score:.3f}, "
            f"MW={self.molecular_weight:.1f}, tox={self.toxicity_score:.3f})"
        )


# ──────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────

class MoleculeValidator:
    """RDKit-based validation and property scoring.

    Validates SMILES strings and computes multi-property scores
    for the PPO reward function.

    Args:
        fingerprint_radius: Morgan fingerprint radius (default 2).
        fingerprint_nbits: Morgan fingerprint bit length (default 2048).
    """

    def __init__(
        self,
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
    ):
        if not RDKIT_AVAILABLE:
            raise RuntimeError(
                "RDKit is required for molecule validation. "
                "Install with: pip install rdkit"
            )
        self.fp_radius = fingerprint_radius
        self.fp_nbits = fingerprint_nbits

    def parse_smiles(self, smiles: str) -> Optional[object]:
        """Parse SMILES string into RDKit Mol object.

        Returns None if SMILES is invalid or has valence errors.
        """
        if not smiles or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is not None:
                # Check for sanitization issues
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return None
            return mol
        except Exception:
            return None

    def canonicalize(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form."""
        mol = self.parse_smiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    def compute_fingerprint(self, mol):
        """Compute Morgan fingerprint for a molecule."""
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_radius, nBits=self.fp_nbits
        )

    def compute_tanimoto(self, mol_a, mol_b) -> float:
        """Compute Tanimoto similarity between two molecules.

        Args:
            mol_a: RDKit Mol object (or SMILES string)
            mol_b: RDKit Mol object (or SMILES string)

        Returns:
            Tanimoto coefficient in [0, 1].
        """
        if isinstance(mol_a, str):
            mol_a = self.parse_smiles(mol_a)
        if isinstance(mol_b, str):
            mol_b = self.parse_smiles(mol_b)
        if mol_a is None or mol_b is None:
            return 0.0

        fp_a = self.compute_fingerprint(mol_a)
        fp_b = self.compute_fingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)

    def compute_qed(self, mol) -> float:
        """Compute Quantitative Estimate of Drug-likeness (0-1, higher=better)."""
        try:
            return QED.qed(mol)
        except Exception:
            return 0.0

    def compute_sa(self, mol) -> float:
        """Compute Synthetic Accessibility score (0-1, higher=easier to synthesize)."""
        return _compute_sa_score(mol)

    def compute_lipinski(self, mol) -> Tuple[int, Dict[str, bool]]:
        """Check Lipinski's Rule of Five.

        Returns:
            (num_violations, details_dict)
        """
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            details = {
                "MW_le_500": mw <= 500,
                "LogP_le_5": logp <= 5,
                "HBD_le_5": hbd <= 5,
                "HBA_le_10": hba <= 10,
            }
            violations = sum(1 for v in details.values() if not v)
            return violations, details
        except Exception:
            return 4, {}

    def validate_candidate(
        self,
        candidate_smiles: str,
        seed_smiles: Optional[str] = None,
        min_tanimoto: float = 0.0,
    ) -> ValidationResult:
        """Full validation pipeline for a candidate molecule.

        Args:
            candidate_smiles: SMILES of the generated candidate.
            seed_smiles: SMILES of the original (seed) molecule for similarity.
            min_tanimoto: Minimum Tanimoto similarity threshold.

        Returns:
            ValidationResult with all computed properties.
        """
        result = ValidationResult(smiles=candidate_smiles)

        # Step 1: Parse SMILES
        mol = self.parse_smiles(candidate_smiles)
        if mol is None:
            result.error = "Invalid SMILES — RDKit parse failed"
            return result

        # Canonicalize
        result.canonical_smiles = Chem.MolToSmiles(mol)
        result.valid = True

        # Step 2: Basic properties
        result.molecular_weight = Descriptors.MolWt(mol)
        result.heavy_atom_count = mol.GetNumHeavyAtoms()
        result.num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)

        # Step 3: Tanimoto similarity to seed
        if seed_smiles:
            seed_mol = self.parse_smiles(seed_smiles)
            if seed_mol is not None:
                result.tanimoto = self.compute_tanimoto(mol, seed_mol)
                if result.tanimoto < min_tanimoto:
                    result.valid = False
                    result.error = (
                        f"Tanimoto {result.tanimoto:.3f} below threshold "
                        f"{min_tanimoto:.3f}"
                    )
                    return result

        # Step 4: Drug-likeness
        result.qed = self.compute_qed(mol)
        result.sa_score = self.compute_sa(mol)

        # Step 5: Lipinski
        result.lipinski_violations, result.lipinski_details = self.compute_lipinski(mol)

        return result

    def validate_batch(
        self,
        candidate_smiles_list: List[str],
        seed_smiles: Optional[str] = None,
        min_tanimoto: float = 0.0,
    ) -> List[ValidationResult]:
        """Validate a batch of candidate molecules.

        Args:
            candidate_smiles_list: List of SMILES strings.
            seed_smiles: Seed molecule SMILES for similarity calculation.
            min_tanimoto: Minimum Tanimoto threshold.

        Returns:
            List of ValidationResult objects.
        """
        return [
            self.validate_candidate(smi, seed_smiles, min_tanimoto)
            for smi in candidate_smiles_list
        ]
