"""Conversion efficiency estimation using Maximum Common Substructure (MCS).

Estimates how difficult it would be to synthetically convert the seed
molecule into the candidate molecule. Uses RDKit's MCS algorithm to
find the largest shared substructure.

High MCS ratio = most of the molecule stays the same = easy conversion
Low MCS ratio  = completely different scaffolds = hard conversion

Example:
    nitrobenzene -> aniline:     MCS = benzene ring (6/7 atoms) = 0.86 score
    nitrobenzene -> random drug: MCS = nothing shared           = 0.05 score
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@dataclass
class ConversionResult:
    """Result of conversion efficiency analysis."""
    mcs_ratio: float = 0.0          # Fraction of atoms in common substructure
    mcs_smarts: str = ""            # SMARTS of maximum common substructure
    mcs_num_atoms: int = 0          # Number of atoms in MCS
    mcs_num_bonds: int = 0          # Number of bonds in MCS
    seed_num_atoms: int = 0         # Total atoms in seed
    candidate_num_atoms: int = 0    # Total atoms in candidate
    conversion_score: float = 0.0   # Final 0-1 score
    estimated_steps: int = 0        # Rough estimate of synthetic steps

    def __repr__(self):
        return (
            f"ConversionResult(score={self.conversion_score:.3f}, "
            f"MCS_ratio={self.mcs_ratio:.3f}, "
            f"MCS_atoms={self.mcs_num_atoms}/{self.seed_num_atoms}, "
            f"est_steps={self.estimated_steps})"
        )


class ConversionEfficiency:
    """Estimate synthetic conversion efficiency between two molecules.

    Uses Maximum Common Substructure (MCS) to determine how much
    of the seed molecule is preserved in the candidate. Higher
    preservation means fewer synthetic steps needed.

    Scoring:
        MCS ratio > 0.8  -> score ~1.0 (trivial, 1-step conversion)
        MCS ratio 0.5-0.8 -> score 0.5-0.8 (moderate, 2-4 steps)
        MCS ratio < 0.3  -> score ~0.1 (hard, different scaffold)
    """

    def __init__(self, timeout_seconds: int = 5):
        """Initialize with MCS computation timeout.

        Args:
            timeout_seconds: Maximum time for MCS computation.
                Smaller molecules are fast, but complex ones can be slow.
        """
        self.timeout = timeout_seconds

    def compute(
        self,
        seed_smiles: str,
        candidate_smiles: str,
    ) -> ConversionResult:
        """Compute conversion efficiency between seed and candidate.

        Args:
            seed_smiles: SMILES of the original molecule.
            candidate_smiles: SMILES of the proposed detoxified molecule.

        Returns:
            ConversionResult with MCS analysis and score.
        """
        result = ConversionResult()

        if not HAS_RDKIT:
            result.conversion_score = 0.5
            return result

        seed_mol = Chem.MolFromSmiles(seed_smiles)
        cand_mol = Chem.MolFromSmiles(candidate_smiles)

        if seed_mol is None or cand_mol is None:
            result.conversion_score = 0.0
            return result

        result.seed_num_atoms = seed_mol.GetNumHeavyAtoms()
        result.candidate_num_atoms = cand_mol.GetNumHeavyAtoms()

        # Compute MCS
        try:
            mcs_result = rdFMCS.FindMCS(
                [seed_mol, cand_mol],
                timeout=self.timeout,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                matchValences=False,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
            )

            result.mcs_num_atoms = mcs_result.numAtoms
            result.mcs_num_bonds = mcs_result.numBonds
            result.mcs_smarts = mcs_result.smartsString

        except Exception as e:
            logger.warning(f"MCS computation failed: {e}")
            result.conversion_score = 0.5
            return result

        # Compute MCS ratio (fraction of seed preserved)
        max_atoms = max(result.seed_num_atoms, result.candidate_num_atoms)
        if max_atoms > 0:
            result.mcs_ratio = result.mcs_num_atoms / max_atoms
        else:
            result.mcs_ratio = 0.0

        # Convert MCS ratio to conversion score
        # Piecewise linear: emphasizes high-MCS candidates
        if result.mcs_ratio >= 0.8:
            result.conversion_score = 0.9 + 0.1 * (result.mcs_ratio - 0.8) / 0.2
        elif result.mcs_ratio >= 0.5:
            result.conversion_score = 0.5 + 0.4 * (result.mcs_ratio - 0.5) / 0.3
        elif result.mcs_ratio >= 0.2:
            result.conversion_score = 0.1 + 0.4 * (result.mcs_ratio - 0.2) / 0.3
        else:
            result.conversion_score = result.mcs_ratio * 0.5

        # Estimate synthetic steps based on atoms changed
        atoms_changed = max_atoms - result.mcs_num_atoms
        if atoms_changed <= 1:
            result.estimated_steps = 1
        elif atoms_changed <= 3:
            result.estimated_steps = 2
        elif atoms_changed <= 6:
            result.estimated_steps = 3
        elif atoms_changed <= 10:
            result.estimated_steps = 4
        else:
            result.estimated_steps = 5  # 5+ steps = complex synthesis

        return result
