"""PubChem safety data fetcher for on-demand toxicological information.

Fetches safety-relevant data from PubChem for molecules not in T3DB:
    - GHS hazard classification
    - Hazard statements (H-codes)
    - Precautionary statements (P-codes)
    - Pharmacology/toxicology summary
    - First aid measures
    - IUPAC name, SMILES, CAS number

Uses the pubchempy library (already in requirements.txt) and direct
PubChem PUG REST API for safety data sections.

Rate limiting: PubChem allows 5 requests/second for API access.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# PubChem PUG REST API base URL
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

# Rate limiting
REQUEST_DELAY = 0.25  # 4 requests/second (conservative)


class PubChemFetcher:
    """Fetch safety and hazard data from PubChem REST API.

    Provides toxicologically relevant information for any compound
    in PubChem (~100M compounds), complementing the T3DB knowledge base.

    Usage:
        fetcher = PubChemFetcher()
        records = fetcher.fetch_safety_data("nitrobenzene")
        for r in records:
            print(r["name"], r["ghs_classification"])
    """

    def __init__(self, timeout: int = 15, max_retries: int = 2):
        """
        Args:
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retry attempts on failure.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ToxGuard-RAG/0.1 (academic research)",
        })

    def lookup_identifiers(
        self,
        query: str,
        query_type: str = "name",
    ) -> dict:
        """Quick PubChem lookup for molecule identifiers.

        Returns CAS, SMILES, InChIKey, and other identifiers without
        fetching the full safety data (much faster).

        Args:
            query: Compound identifier (name, CAS, SMILES, or CID).
            query_type: Type of query ("name", "cas", "smiles", "cid").

        Returns:
            Dict with resolved fields; empty strings for unresolved.
        """
        result = {
            "name": "",
            "iupac_name": "",
            "smiles": "",
            "cas_number": "",
            "inchikey": "",
            "molecular_formula": "",
            "molecular_weight": "",
        }

        cid = self._resolve_to_cid(query, query_type)
        if not cid:
            return result

        props = self._get_compound_properties(cid)
        result.update({k: v for k, v in props.items() if v})
        return result

    def fetch_safety_data(
        self,
        query: str,
        query_type: str = "name",
    ) -> List[dict]:
        """Fetch comprehensive safety data for a compound.

        Args:
            query: Compound identifier (name, CAS, SMILES, or CID).
            query_type: Type of query ("name", "cas", "smiles", "cid").

        Returns:
            List of record dicts (usually 1), each containing:
                - cid, name, iupac_name, smiles, cas_number, inchikey
                - ghs_classification, hazard_statements
                - safety_measures, first_aid, acute_toxicity, pharmacology
        """
        # Step 1: Resolve to PubChem CID
        cid = self._resolve_to_cid(query, query_type)
        if not cid:
            logger.warning(f"Could not resolve '{query}' to PubChem CID")
            return []

        logger.info(f"Resolved '{query}' to PubChem CID {cid}")

        # Step 2: Get compound properties
        props = self._get_compound_properties(cid)
        time.sleep(REQUEST_DELAY)

        # Step 3: Get safety/hazard data from PUG View
        safety = self._get_safety_data(cid)
        time.sleep(REQUEST_DELAY)

        record = {
            "cid": cid,
            "name": props.get("name", query),
            "iupac_name": props.get("iupac_name", ""),
            "smiles": props.get("smiles", ""),
            "cas_number": props.get("cas_number", ""),
            "inchikey": props.get("inchikey", ""),
            "molecular_formula": props.get("molecular_formula", ""),
            "molecular_weight": props.get("molecular_weight", ""),
            "ghs_classification": safety.get("ghs_classification", ""),
            "hazard_statements": safety.get("hazard_statements", ""),
            "safety_measures": safety.get("safety_measures", ""),
            "first_aid": safety.get("first_aid", ""),
            "acute_toxicity": safety.get("acute_toxicity", ""),
            "pharmacology": safety.get("pharmacology", ""),
        }

        return [record] if any(
            record.get(k) for k in [
                "ghs_classification", "hazard_statements",
                "safety_measures", "first_aid", "acute_toxicity",
                "pharmacology",
            ]
        ) else [record]

    def _resolve_to_cid(self, query: str, query_type: str = "name") -> Optional[int]:
        """Resolve a compound identifier to PubChem CID."""
        try:
            if query_type == "cid":
                return int(query)

            namespace = {
                "name": "name",
                "cas": "name",  # PubChem handles CAS via name search
                "smiles": "smiles",
            }.get(query_type, "name")

            url = f"{PUBCHEM_BASE}/compound/{namespace}/{query}/cids/JSON"
            resp = self._request(url)
            if resp and "IdentifierList" in resp:
                cids = resp["IdentifierList"].get("CID", [])
                return cids[0] if cids else None

        except Exception as e:
            logger.debug(f"CID resolution failed for '{query}': {e}")

        return None

    def _get_compound_properties(self, cid: int) -> dict:
        """Get basic compound properties from PubChem."""
        props = {}

        try:
            url = (
                f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
                "IUPACName,IsomericSMILES,CanonicalSMILES,InChIKey,MolecularFormula,MolecularWeight/JSON"
            )
            resp = self._request(url)

            if resp and "PropertyTable" in resp:
                entries = resp["PropertyTable"].get("Properties", [])
                if entries:
                    entry = entries[0]
                    props["iupac_name"] = entry.get("IUPACName", "")
                    # PubChem API returns SMILES under varying key names:
                    #   CanonicalSMILES → "ConnectivitySMILES"
                    #   IsomericSMILES  → "SMILES"
                    # Try all possible keys for robustness
                    props["smiles"] = (
                        entry.get("CanonicalSMILES", "")
                        or entry.get("IsomericSMILES", "")
                        or entry.get("ConnectivitySMILES", "")
                        or entry.get("SMILES", "")
                    )
                    props["inchikey"] = entry.get("InChIKey", "")
                    props["molecular_formula"] = entry.get("MolecularFormula", "")
                    props["molecular_weight"] = str(entry.get("MolecularWeight", ""))

            # Get synonyms (common name + CAS)
            url_syn = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
            time.sleep(REQUEST_DELAY)
            resp_syn = self._request(url_syn)

            if resp_syn and "InformationList" in resp_syn:
                info = resp_syn["InformationList"].get("Information", [])
                if info:
                    synonyms = info[0].get("Synonym", [])
                    if synonyms:
                        props["name"] = synonyms[0]  # First synonym = common name

                    # Find CAS number (format: digits-digits-digit)
                    import re
                    for syn in synonyms[:50]:
                        if re.match(r"^\d{2,7}-\d{2}-\d$", syn):
                            props["cas_number"] = syn
                            break

        except Exception as e:
            logger.debug(f"Property fetch failed for CID {cid}: {e}")

        return props

    def _get_safety_data(self, cid: int) -> dict:
        """Get GHS hazard, safety, first aid, and toxicity data from PUG View."""
        safety = {}

        try:
            url = f"{PUBCHEM_VIEW}/{cid}/JSON"
            resp = self._request(url)

            if not resp or "Record" not in resp:
                return safety

            record = resp["Record"]
            sections = record.get("Section", [])

            for section in sections:
                heading = section.get("TOCHeading", "")

                # Safety and Hazards section
                if heading == "Safety and Hazards":
                    safety.update(self._parse_safety_section(section))

                # Pharmacology and Biochemistry
                elif heading == "Pharmacology and Biochemistry":
                    safety["pharmacology"] = self._extract_text_from_section(
                        section, max_length=2000
                    )

                # Toxicity — extract both general tox and acute LD50/LC50
                elif heading == "Toxicity":
                    tox_text = self._extract_text_from_section(
                        section, max_length=2000
                    )
                    if tox_text:
                        safety["pharmacology"] = (
                            safety.get("pharmacology", "") + "\n\n"
                            + f"TOXICITY DATA:\n{tox_text}"
                        ).strip()

                    # Extract acute toxicity data (LD50, LC50) specifically
                    acute_text = self._extract_acute_toxicity(section)
                    if acute_text:
                        safety["acute_toxicity"] = acute_text

                # First Aid Measures (sometimes a top-level section)
                elif "First Aid" in heading:
                    fa_text = self._extract_text_from_section(
                        section, max_length=1500
                    )
                    if fa_text:
                        safety["first_aid"] = fa_text

        except Exception as e:
            logger.debug(f"Safety data fetch failed for CID {cid}: {e}")

        return safety

    def _parse_safety_section(self, section: dict) -> dict:
        """Parse the Safety and Hazards section from PUG View."""
        result = {}

        subsections = section.get("Section", [])
        for sub in subsections:
            heading = sub.get("TOCHeading", "")

            if "GHS" in heading or "Hazards" in heading:
                ghs_parts = []
                hazard_parts = []

                for item in self._walk_section_items(sub):
                    text = item.get("StringWithMarkup", [{}])[0].get("String", "") if isinstance(item.get("StringWithMarkup"), list) else ""
                    if not text:
                        text = str(item.get("StringValue", ""))

                    if text:
                        if text.startswith("H") and text[1:4].isdigit():
                            hazard_parts.append(text)
                        elif text.startswith("P") and text[1:4].isdigit():
                            pass  # Precautionary statements
                        else:
                            ghs_parts.append(text)

                if ghs_parts:
                    result["ghs_classification"] = "\n".join(ghs_parts[:20])
                if hazard_parts:
                    result["hazard_statements"] = "\n".join(hazard_parts[:20])

            elif "First Aid" in heading:
                text = self._extract_text_from_section(sub, max_length=1500)
                if text:
                    result["first_aid"] = text

            elif "Safety" in heading or "Handling" in heading:
                text = self._extract_text_from_section(sub, max_length=1500)
                if text:
                    result["safety_measures"] = text

        return result

    def _extract_acute_toxicity(self, section: dict) -> str:
        """Extract LD50, LC50, and other dose-response data from Toxicity section."""
        acute_parts = []

        for sub in section.get("Section", []):
            heading = sub.get("TOCHeading", "")
            heading_lower = heading.lower()

            # Look for acute toxicity, LD50, LC50 subsections
            if any(kw in heading_lower for kw in [
                "acute", "ld50", "lc50", "lethal", "dose",
                "oral", "dermal", "inhalation", "toxicity value",
            ]):
                text = self._extract_text_from_section(sub, max_length=800)
                if text:
                    acute_parts.append(f"{heading}:\n{text}")

            # Recurse into nested subsections
            for nested in sub.get("Section", []):
                nested_heading = nested.get("TOCHeading", "")
                nh_lower = nested_heading.lower()
                if any(kw in nh_lower for kw in [
                    "acute", "ld50", "lc50", "lethal", "dose", "toxicity value",
                ]):
                    text = self._extract_text_from_section(
                        nested, max_length=800
                    )
                    if text:
                        acute_parts.append(f"{nested_heading}:\n{text}")

        combined = "\n\n".join(acute_parts)
        return combined[:2000] if combined else ""

    def _walk_section_items(self, section: dict):
        """Recursively walk section Information items."""
        for info in section.get("Information", []):
            value = info.get("Value", {})
            for item in value.get("StringWithMarkup", []):
                yield {"StringWithMarkup": [item]}

        for sub in section.get("Section", []):
            yield from self._walk_section_items(sub)

    def _extract_text_from_section(
        self, section: dict, max_length: int = 1500
    ) -> str:
        """Extract readable text from a PUG View section recursively."""
        texts = []

        for info in section.get("Information", []):
            value = info.get("Value", {})
            for item in value.get("StringWithMarkup", []):
                text = item.get("String", "")
                if text and len(text) > 10:
                    texts.append(text)

        for sub in section.get("Section", []):
            sub_text = self._extract_text_from_section(sub, max_length=500)
            if sub_text:
                heading = sub.get("TOCHeading", "")
                if heading:
                    texts.append(f"\n{heading}:\n{sub_text}")
                else:
                    texts.append(sub_text)

        combined = "\n".join(texts)
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."
        return combined

    def _request(self, url: str) -> Optional[dict]:
        """Make an HTTP request with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 503:
                    # Rate limited — back off
                    delay = 2 ** attempt
                    logger.debug(f"PubChem rate limited, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    logger.debug(
                        f"PubChem HTTP {resp.status_code} for {url}"
                    )
                    return None
            except requests.exceptions.Timeout:
                logger.debug(f"PubChem request timeout (attempt {attempt+1})")
            except Exception as e:
                logger.debug(f"PubChem request error: {e}")
                return None

        return None
