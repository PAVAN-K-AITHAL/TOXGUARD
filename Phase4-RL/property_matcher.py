"""Chemical property comparison between seed and candidate molecules.

Compares physicochemical properties using RDKit descriptors to ensure
the detoxified molecule retains similar chemical behavior.

Properties compared:
    - Molecular Weight (MW)
    - LogP (lipophilicity)
    - Hydrogen Bond Donors (HBD)
    - Hydrogen Bond Acceptors (HBA)
    - Polar Surface Area (PSA)
    - Rotatable Bonds
    - Number of Rings

Returns a property_score (0-1) indicating how well properties are preserved.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@dataclass
class PropertyProfile:
    """Physicochemical property profile of a molecule."""
    mw: float = 0.0               # Molecular Weight
    logp: float = 0.0             # LogP (lipophilicity)
    hbd: int = 0                  # H-bond donors
    hba: int = 0                  # H-bond acceptors
    psa: float = 0.0              # Polar Surface Area
    rotatable_bonds: int = 0      # Number of rotatable bonds
    num_rings: int = 0            # Number of rings
    num_aromatic_rings: int = 0   # Number of aromatic rings
    smiles: str = ""

    def __repr__(self):
        return (
            f"PropertyProfile(MW={self.mw:.1f}, LogP={self.logp:.2f}, "
            f"HBD={self.hbd}, HBA={self.hba}, PSA={self.psa:.1f}, "
            f"RotBonds={self.rotatable_bonds}, Rings={self.num_rings})"
        )


@dataclass
class PropertyComparison:
    """Result of comparing two property profiles."""
    overall_score: float = 0.0     # 0-1 combined score
    mw_score: float = 0.0          # Individual MW similarity
    logp_score: float = 0.0
    hbd_score: float = 0.0
    hba_score: float = 0.0
    psa_score: float = 0.0
    rotbond_score: float = 0.0
    ring_score: float = 0.0
    seed_profile: Optional[PropertyProfile] = None
    candidate_profile: Optional[PropertyProfile] = None

    def __repr__(self):
        return (
            f"PropertyComparison(score={self.overall_score:.3f}, "
            f"MW={self.mw_score:.2f}, LogP={self.logp_score:.2f}, "
            f"HBD={self.hbd_score:.2f}, HBA={self.hba_score:.2f}, "
            f"PSA={self.psa_score:.2f})"
        )


class PropertyMatcher:
    """Compare chemical properties between seed and candidate molecules.

    Uses RDKit to compute physicochemical properties and score
    how well they match between the original and proposed molecule.

    Property preservation is critical because the detoxified molecule
    should behave similarly (solubility, absorption, etc.) to the
    original — just without the toxic effect.
    """

    # Tolerances for each property (within these ranges = perfect score)
    TOLERANCES = {
        "mw": 0.30,             # ±30% MW difference is acceptable
        "logp": 1.5,            # ±1.5 LogP units
        "hbd": 2,               # ±2 donors
        "hba": 3,               # ±3 acceptors
        "psa": 0.40,            # ±40% PSA difference
        "rotatable_bonds": 3,   # ±3 bonds
        "rings": 1,             # ±1 ring
    }

    # Weights for combining property scores
    WEIGHTS = {
        "mw": 0.20,
        "logp": 0.25,
        "hbd": 0.10,
        "hba": 0.10,
        "psa": 0.15,
        "rotatable_bonds": 0.10,
        "rings": 0.10,
    }

    @staticmethod
    def compute_profile(smiles: str) -> Optional[PropertyProfile]:
        """Compute physicochemical property profile for a molecule.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            PropertyProfile or None if computation fails.
        """
        if not HAS_RDKIT:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            profile = PropertyProfile(
                mw=Descriptors.MolWt(mol),
                logp=Descriptors.MolLogP(mol),
                hbd=Lipinski.NumHDonors(mol),
                hba=Lipinski.NumHAcceptors(mol),
                psa=Descriptors.TPSA(mol),
                rotatable_bonds=Lipinski.NumRotatableBonds(mol),
                num_rings=Lipinski.RingCount(mol),
                num_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
                smiles=smiles,
            )
            return profile
        except Exception as e:
            logger.warning(f"Property computation failed for {smiles}: {e}")
            return None

    def compare(
        self,
        seed_smiles: str,
        candidate_smiles: str,
    ) -> PropertyComparison:
        """Compare properties between seed and candidate molecules.

        Args:
            seed_smiles: SMILES of the original toxic molecule.
            candidate_smiles: SMILES of the proposed detoxified molecule.

        Returns:
            PropertyComparison with individual and overall scores.
        """
        result = PropertyComparison()

        seed_prof = self.compute_profile(seed_smiles)
        cand_prof = self.compute_profile(candidate_smiles)

        if seed_prof is None or cand_prof is None:
            result.overall_score = 0.5  # Neutral if can't compute
            return result

        result.seed_profile = seed_prof
        result.candidate_profile = cand_prof

        # MW score: ratio-based comparison
        if seed_prof.mw > 0:
            mw_ratio = abs(cand_prof.mw - seed_prof.mw) / seed_prof.mw
            result.mw_score = max(0, 1.0 - mw_ratio / self.TOLERANCES["mw"])
        else:
            result.mw_score = 0.5

        # LogP score: absolute difference
        logp_diff = abs(cand_prof.logp - seed_prof.logp)
        result.logp_score = max(0, 1.0 - logp_diff / self.TOLERANCES["logp"])

        # HBD score: absolute count difference
        hbd_diff = abs(cand_prof.hbd - seed_prof.hbd)
        result.hbd_score = max(0, 1.0 - hbd_diff / self.TOLERANCES["hbd"])

        # HBA score: absolute count difference
        hba_diff = abs(cand_prof.hba - seed_prof.hba)
        result.hba_score = max(0, 1.0 - hba_diff / self.TOLERANCES["hba"])

        # PSA score: ratio-based
        if seed_prof.psa > 0:
            psa_ratio = abs(cand_prof.psa - seed_prof.psa) / seed_prof.psa
            result.psa_score = max(0, 1.0 - psa_ratio / self.TOLERANCES["psa"])
        else:
            result.psa_score = 0.5

        # Rotatable bonds score
        rot_diff = abs(cand_prof.rotatable_bonds - seed_prof.rotatable_bonds)
        result.rotbond_score = max(
            0, 1.0 - rot_diff / self.TOLERANCES["rotatable_bonds"]
        )

        # Ring count score
        ring_diff = abs(cand_prof.num_rings - seed_prof.num_rings)
        result.ring_score = max(0, 1.0 - ring_diff / self.TOLERANCES["rings"])

        # Weighted combination
        result.overall_score = (
            self.WEIGHTS["mw"] * result.mw_score
            + self.WEIGHTS["logp"] * result.logp_score
            + self.WEIGHTS["hbd"] * result.hbd_score
            + self.WEIGHTS["hba"] * result.hba_score
            + self.WEIGHTS["psa"] * result.psa_score
            + self.WEIGHTS["rotatable_bonds"] * result.rotbond_score
            + self.WEIGHTS["rings"] * result.ring_score
        )

        return result

    def format_comparison(self, comparison: PropertyComparison) -> str:
        """Format a property comparison as a human-readable string."""
        if comparison.seed_profile is None or comparison.candidate_profile is None:
            return "Property comparison unavailable"

        s = comparison.seed_profile
        c = comparison.candidate_profile

        lines = [
            f"Property Preservation Score: {comparison.overall_score:.3f}",
            f"  {'Property':<20s} {'Seed':>10s} {'Candidate':>10s} {'Score':>8s}",
            f"  {'MW':<20s} {s.mw:>10.1f} {c.mw:>10.1f} {comparison.mw_score:>8.2f}",
            f"  {'LogP':<20s} {s.logp:>10.2f} {c.logp:>10.2f} {comparison.logp_score:>8.2f}",
            f"  {'HBD':<20s} {s.hbd:>10d} {c.hbd:>10d} {comparison.hbd_score:>8.2f}",
            f"  {'HBA':<20s} {s.hba:>10d} {c.hba:>10d} {comparison.hba_score:>8.2f}",
            f"  {'PSA':<20s} {s.psa:>10.1f} {c.psa:>10.1f} {comparison.psa_score:>8.2f}",
            f"  {'Rotatable Bonds':<20s} {s.rotatable_bonds:>10d} {c.rotatable_bonds:>10d} {comparison.rotbond_score:>8.2f}",
            f"  {'Rings':<20s} {s.num_rings:>10d} {c.num_rings:>10d} {comparison.ring_score:>8.2f}",
        ]
        return "\n".join(lines)
