"""Bidirectional IUPAC ↔ SMILES converter for Phase 4 RL pipeline.

Resolution cascade:
    IUPAC → SMILES:  py2opsin (OPSIN, ~95%, offline) → PubChem API → NCI CIR
    SMILES → IUPAC:  PubChem API → NCI CIR

Features:
    - In-memory LRU cache + optional disk cache for RL training loops
    - Batch resolution for PPO training batches
    - Thread-safe caching

Usage:
    resolver = NameResolver()
    smiles = resolver.iupac_to_smiles("nitrobenzene")        # "[O-][N+](=O)c1ccccc1"
    iupac = resolver.smiles_to_iupac("[O-][N+](=O)c1ccccc1") # "nitrobenzene"
"""

import csv
import hashlib
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Disk Cache
# ──────────────────────────────────────────────────────────────────────

class _DiskCache:
    """Simple CSV-backed disk cache for name resolution results.

    Used during RL training to avoid repeated API/OPSIN calls for the
    same generated IUPAC names across epochs.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        self._cache[row[0]] = row[1]
            logger.debug(f"Loaded {len(self._cache)} entries from disk cache")
        except Exception as e:
            logger.warning(f"Failed to load disk cache: {e}")

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def put(self, key: str, value: str):
        if self._cache.get(key) != value:
            self._cache[key] = value
            self._dirty = True

    def save(self):
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        try:
            with open(self.cache_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["input", "output"])
                for k, v in sorted(self._cache.items()):
                    writer.writerow([k, v])
            self._dirty = False
            logger.debug(f"Saved {len(self._cache)} entries to disk cache")
        except Exception as e:
            logger.warning(f"Failed to save disk cache: {e}")

    def __len__(self):
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────────
# OPSIN Resolver (py2opsin)
# ──────────────────────────────────────────────────────────────────────

def _resolve_opsin(iupac_name: str) -> Optional[str]:
    """Convert IUPAC name to SMILES using OPSIN (via py2opsin).

    OPSIN is a rule-based IUPAC parser with ~95% accuracy.
    Requires: pip install py2opsin (+ Java JRE installed).
    """
    try:
        from py2opsin import py2opsin
        result = py2opsin(iupac_name)
        if result and result.strip() and result.strip() != "":
            smiles = result.strip()
            # Validate it's not an error message
            if len(smiles) < 500 and not smiles.startswith("Error"):
                logger.debug(f"OPSIN resolved: {iupac_name} -> {smiles}")
                return smiles
    except ImportError:
        logger.debug("py2opsin not installed — skipping OPSIN resolver")
    except Exception as e:
        logger.debug(f"OPSIN failed for '{iupac_name}': {e}")
    return None


# ──────────────────────────────────────────────────────────────────────
# PubChem API Resolver
# ──────────────────────────────────────────────────────────────────────

def _resolve_pubchem_iupac_to_smiles(iupac_name: str) -> Optional[str]:
    """Convert IUPAC name to SMILES via PubChem REST API."""
    import requests

    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(iupac_name)}/property/CanonicalSMILES/JSON"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = props[0].get("CanonicalSMILES", "")
                if smiles:
                    logger.debug(f"PubChem resolved: {iupac_name} -> {smiles}")
                    return smiles
    except Exception as e:
        logger.debug(f"PubChem API failed for '{iupac_name}': {e}")
    return None


def _resolve_pubchem_smiles_to_iupac(smiles: str) -> Optional[str]:
    """Convert SMILES to IUPAC name via PubChem REST API."""
    import requests

    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{requests.utils.quote(smiles)}/property/IUPACName/JSON"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                iupac = props[0].get("IUPACName", "")
                if iupac:
                    logger.debug(f"PubChem resolved: {smiles} -> {iupac}")
                    return iupac
    except Exception as e:
        logger.debug(f"PubChem API failed for '{smiles}': {e}")
    return None


# ──────────────────────────────────────────────────────────────────────
# NCI CIR Resolver (fallback)
# ──────────────────────────────────────────────────────────────────────

def _resolve_nci_cir(identifier: str, target_repr: str = "smiles") -> Optional[str]:
    """Resolve chemical identifier via NCI Chemical Identifier Resolver.

    Args:
        identifier: IUPAC name or SMILES
        target_repr: "smiles" or "iupac_name"
    """
    import requests

    url = (
        f"https://cactus.nci.nih.gov/chemical/structure/"
        f"{requests.utils.quote(identifier)}/{target_repr}"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            result = resp.text.strip().split("\n")[0]
            if result and len(result) < 500:
                logger.debug(f"NCI CIR resolved: {identifier} -> {result}")
                return result
    except Exception as e:
        logger.debug(f"NCI CIR failed for '{identifier}': {e}")
    return None


# ──────────────────────────────────────────────────────────────────────
# Main Resolver Class
# ──────────────────────────────────────────────────────────────────────

class NameResolver:
    """Bidirectional IUPAC ↔ SMILES resolver with caching.

    Resolution cascade:
        IUPAC → SMILES:  OPSIN (offline) → PubChem API → NCI CIR
        SMILES → IUPAC:  PubChem API → NCI CIR

    Args:
        cache_dir: Directory for disk cache files. If None, disk caching
            is disabled (in-memory LRU cache still active).
        use_opsin: Whether to try py2opsin first (requires Java JRE).
        api_delay: Delay between API calls (seconds) for rate limiting.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_opsin: bool = True,
        api_delay: float = 0.1,
    ):
        self.use_opsin = use_opsin
        self.api_delay = api_delay

        # Disk caches
        self._i2s_cache = None  # IUPAC → SMILES
        self._s2i_cache = None  # SMILES → IUPAC
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._i2s_cache = _DiskCache(os.path.join(cache_dir, "iupac_to_smiles_cache.csv"))
            self._s2i_cache = _DiskCache(os.path.join(cache_dir, "smiles_to_iupac_cache.csv"))
            logger.info(
                f"Disk cache loaded: {len(self._i2s_cache)} IUPAC->SMILES, "
                f"{len(self._s2i_cache)} SMILES->IUPAC"
            )

        # Track OPSIN availability
        self._opsin_available = None

    def _check_opsin(self) -> bool:
        """Check if py2opsin is installed and Java is available."""
        if self._opsin_available is not None:
            return self._opsin_available
        try:
            from py2opsin import py2opsin
            # Quick test
            result = py2opsin("ethanol")
            self._opsin_available = bool(result and result.strip())
            if self._opsin_available:
                logger.info("OPSIN resolver available [OK]")
            else:
                logger.warning("py2opsin installed but OPSIN JAR failed — check Java JRE")
                self._opsin_available = False
        except ImportError:
            logger.info("py2opsin not installed — using API-based resolvers only")
            self._opsin_available = False
        except Exception as e:
            logger.warning(f"OPSIN check failed: {e}")
            self._opsin_available = False
        return self._opsin_available

    @lru_cache(maxsize=4096)
    def iupac_to_smiles(self, iupac_name: str) -> Optional[str]:
        """Convert IUPAC name to canonical SMILES.

        Cascade: OPSIN → PubChem → NCI CIR.
        Results are cached in-memory (LRU) and on-disk.

        Args:
            iupac_name: IUPAC systematic name.

        Returns:
            Canonical SMILES string, or None if resolution failed.
        """
        if not iupac_name or not iupac_name.strip():
            return None

        iupac_name = iupac_name.strip()

        # Check disk cache
        if self._i2s_cache:
            cached = self._i2s_cache.get(iupac_name)
            if cached is not None:
                return cached if cached != "" else None

        # Cascade resolution
        smiles = None

        # 1. OPSIN (fastest, offline, ~95% accuracy)
        if self.use_opsin and self._check_opsin():
            smiles = _resolve_opsin(iupac_name)

        # 2. PubChem API
        if smiles is None:
            smiles = _resolve_pubchem_iupac_to_smiles(iupac_name)
            if smiles is None and self.api_delay > 0:
                time.sleep(self.api_delay)

        # 3. NCI CIR (last resort)
        if smiles is None:
            smiles = _resolve_nci_cir(iupac_name, "smiles")

        # Validate with RDKit if available
        if smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"RDKit rejected SMILES '{smiles}' for '{iupac_name}'")
                    smiles = None
                else:
                    # Canonicalize
                    smiles = Chem.MolToSmiles(mol)
            except ImportError:
                pass  # RDKit not available — skip validation

        # Cache result
        if self._i2s_cache:
            self._i2s_cache.put(iupac_name, smiles or "")

        if smiles:
            logger.debug(f"Resolved: {iupac_name} -> {smiles}")
        else:
            logger.debug(f"Failed to resolve: {iupac_name}")

        return smiles

    @lru_cache(maxsize=4096)
    def smiles_to_iupac(self, smiles: str) -> Optional[str]:
        """Convert SMILES to IUPAC name.

        Cascade: PubChem → NCI CIR.

        Args:
            smiles: Canonical SMILES string.

        Returns:
            IUPAC name, or None if resolution failed.
        """
        if not smiles or not smiles.strip():
            return None

        smiles = smiles.strip()

        # Check disk cache
        if self._s2i_cache:
            cached = self._s2i_cache.get(smiles)
            if cached is not None:
                return cached if cached != "" else None

        # Cascade
        iupac = _resolve_pubchem_smiles_to_iupac(smiles)

        if iupac is None:
            if self.api_delay > 0:
                time.sleep(self.api_delay)
            iupac = _resolve_nci_cir(smiles, "iupac_name")

        # Cache
        if self._s2i_cache:
            self._s2i_cache.put(smiles, iupac or "")

        return iupac

    def resolve_batch(
        self,
        iupac_names: List[str],
        direction: str = "iupac_to_smiles",
    ) -> Dict[str, Optional[str]]:
        """Resolve a batch of names.

        Args:
            iupac_names: List of names to resolve.
            direction: "iupac_to_smiles" or "smiles_to_iupac".

        Returns:
            Dict mapping input → resolved output (or None).
        """
        resolver = (
            self.iupac_to_smiles if direction == "iupac_to_smiles"
            else self.smiles_to_iupac
        )
        results = {}
        for name in iupac_names:
            results[name] = resolver(name)
        return results

    def save_cache(self):
        """Flush disk caches to file."""
        if self._i2s_cache:
            self._i2s_cache.save()
        if self._s2i_cache:
            self._s2i_cache.save()

    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {
            "lru_iupac_to_smiles": self.iupac_to_smiles.cache_info()._asdict(),
            "lru_smiles_to_iupac": self.smiles_to_iupac.cache_info()._asdict(),
        }
        if self._i2s_cache:
            stats["disk_iupac_to_smiles"] = len(self._i2s_cache)
        if self._s2i_cache:
            stats["disk_smiles_to_iupac"] = len(self._s2i_cache)
        return stats
