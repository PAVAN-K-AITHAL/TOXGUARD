"""
ToxGuard End-to-End Pipeline: Phase 1 → Phase 2 → Phase 3 → Phase 4 (optional)
==========================================================

Usage:
    python main.py "nitrobenzene"
    python main.py "nitrobenzene" "arsenic trioxide" "ethanol"
    python main.py "nitrobenzene" -v
    python main.py "nitrobenzene" -o results/profiles.json
    python main.py --no-phase1 "nitrobenzene"
"""

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
import io
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(PROJECT_ROOT, "Phase1-IUPACGPT")
PHASE2_DIR = os.path.join(PROJECT_ROOT, "Phase2-CoT")
PHASE3_DIR = os.path.join(PROJECT_ROOT, "Phase3-RAG")
PHASE4_DIR = os.path.join(PROJECT_ROOT, "Phase4-RL")

sys.path.insert(0, PHASE2_DIR)
sys.path.insert(1, PHASE1_DIR)
sys.path.insert(2, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "iupacGPT", "iupac-gpt", "checkpoints", "iupac")
SPM_PATH = os.path.join(PROJECT_ROOT, "iupacGPT", "iupac-gpt", "iupac_gpt", "iupac_spm.model")
OUTPUT_DIR = os.path.join(PHASE1_DIR, "iupacGPT_outputs")
CHROMA_DIR = os.path.join(PHASE3_DIR, "chroma_db")
LOG_FILE = os.path.join(PROJECT_ROOT, "toxguard_pipeline.log")


# ── Redirect ALL logging to file ─────────────────────────────────────
def _silence_all_logging():
    """Route every log message to a file so the console stays clean."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing console handlers
    root.handlers = []

    # File handler captures everything
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    # Null handler for all noisy libraries as extra safety
    for name in [
        "httpx", "httpcore", "urllib3", "requests", "numexpr",
        "chromadb", "sentence_transformers", "transformers",
        "groq", "huggingface_hub", "onnxruntime", "tqdm",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


# ── Suppress tqdm and other stdout noise ──────────────────────────────
class _SuppressStdout:
    """Context manager to suppress stdout noise from tqdm, model loading, etc."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# ── Import Phase 3 via importlib ──────────────────────────────────────
def _load_phase3_module(module_name: str):
    path = os.path.join(PHASE3_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(
        f"phase3_{module_name}", path,
        submodule_search_locations=[PHASE3_DIR],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, PHASE3_DIR)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(PHASE3_DIR)
    return mod


def get_lora_weights_path():
    pointer = os.path.join(OUTPUT_DIR, "last_run.txt")
    if os.path.exists(pointer):
        with open(pointer) as f:
            run_dir = f.read().strip()
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(PROJECT_ROOT, run_dir)
        lora_path = os.path.join(run_dir, "lora_weights.pt")
        if os.path.exists(lora_path):
            return lora_path
    return None


def load_phase1_predictor(device="cpu"):
    from iupacGPT_finetune.inference import ToxGuardPredictor
    from iupacGPT_finetune.lora import LoRAConfig

    lora_path = get_lora_weights_path()
    if lora_path is None:
        raise FileNotFoundError("No trained LoRA weights found.")

    lora_config = LoRAConfig(
        r=32, alpha=64.0, dropout=0.2,
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,
    )
    predictor = ToxGuardPredictor.from_checkpoint(
        checkpoint_dir=CHECKPOINT_DIR,
        lora_weights_path=lora_path,
        tokenizer_path=SPM_PATH,
        device=device,
        lora_config=lora_config,
    )
    return predictor


def load_phase2_analyzer(api_key=None, model="llama-3.3-70b-versatile"):
    from llm_client import GroqLLMClient
    from cot_analyzer import CoTAnalyzer
    llm = GroqLLMClient(api_key=api_key, model=model)
    return CoTAnalyzer(llm_client=llm)


# ══════════════════════════════════════════════════════════════════════
#  Professional Safety Data Sheet Formatter
# ══════════════════════════════════════════════════════════════════════

W = 78  # Width of the report


def _header_box(title: str) -> str:
    border = "═" * W
    return f"\n  {border}\n  ║  {title.center(W - 6)}  ║\n  {border}"


def _section_header(number: int, title: str) -> str:
    line = "─" * W
    return f"\n  {line}\n   {number}. {title}\n  {line}"


def _kv(key: str, value: str, indent: int = 6) -> str:
    pad = " " * indent
    return f"{pad}{key:<22s}: {value}"


def _wrap_text(text: str, width: int = 68, indent: int = 8) -> str:
    """Word-wrap text to given width with indentation."""
    if not text:
        return " " * indent + "[No data available]"

    import textwrap
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if not line.strip():
            wrapped.append("")
            continue
        w = textwrap.fill(line.strip(), width=width,
                          initial_indent=" " * indent,
                          subsequent_indent=" " * indent)
        wrapped.append(w)
    return "\n".join(wrapped)


def _risk_bar(score: float) -> str:
    """Visual risk indicator bar."""
    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    if score >= 0.80:
        level = "■ CRITICAL"
    elif score >= 0.65:
        level = "■ HIGH"
    elif score >= 0.50:
        level = "■ MODERATE"
    elif score >= 0.20:
        level = "■ LOW"
    else:
        level = "■ MINIMAL"
    return f"[{bar}] {score:.1%}  {level}"


def print_cot_report(cot_result):
    """Print full 7-section Chain-of-Thought analysis."""
    print(_header_box("PHASE 2: CHAIN-OF-THOUGHT ANALYSIS"))

    cot_sections = [
        (1, "STRUCTURAL ANALYSIS", cot_result.structural_analysis),
        (2, "TOXICOPHORE IDENTIFICATION", cot_result.toxicophore_identification),
        (3, "MECHANISM OF ACTION", cot_result.mechanism_of_action),
        (4, "BIOLOGICAL PATHWAYS", cot_result.biological_pathways),
        (5, "ORGAN TOXICITY", cot_result.organ_toxicity),
        (6, "CONFIDENCE ASSESSMENT", cot_result.confidence),
        (7, "VERDICT", cot_result.verdict),
    ]

    for num, title, content in cot_sections:
        print(_section_header(num, title))
        print(_wrap_text(content))

    if cot_result.functional_groups:
        groups = ", ".join(cot_result.functional_groups)
        print(f"\n      Identified Functional Groups: {groups}")
    print(f"      Confidence Level: {cot_result.confidence_level}")
    print(f"      LLM: {cot_result.llm_model}  |  Latency: {cot_result.llm_latency_ms:.0f}ms")
    border = "═" * W
    print(f"  {border}")


def print_safety_report(profile, cot_result=None, elapsed: float = 0.0):
    """Print a professionally formatted Safety Data Sheet."""

    name = profile.common_name or profile.iupac_name
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(_header_box("TOXGUARD SAFETY DATA SHEET"))

    # ── Identification ────────────────────────────────────────────
    print(_section_header(1, "IDENTIFICATION"))
    print(_kv("IUPAC Name", profile.iupac_name or "—"))
    if profile.common_name:
        print(_kv("Common Name", profile.common_name))
    print(_kv("CAS Number", profile.cas_number or "Not available"))
    print(_kv("SMILES", profile.smiles or "Not available"))
    print(_kv("InChIKey", profile.inchikey or "Not available"))
    print(_kv("Date Generated", now))
    print(_kv("Pipeline", "ToxGuard v0.1 (Phase 1 → 2 → 3)"))

    # ── Hazard Summary ────────────────────────────────────────────
    print(_section_header(2, "HAZARD SUMMARY"))
    print()
    toxic_label = "⚠  TOXIC" if profile.is_toxic else "✓  NON-TOXIC"
    print(f"      Classification : {toxic_label}")
    print(f"      Severity       : {profile.severity_label}")
    print(f"      Risk Score     : {_risk_bar(profile.toxicity_score)}")
    print()

    # Show CoT-derived groups and mechanism in the hazard summary
    groups = (cot_result.functional_groups if cot_result
              else profile.cot_functional_groups)
    mechanism = (cot_result.mechanism_of_action if cot_result
                 else profile.cot_mechanism)

    if groups:
        print(f"      Hazardous Groups: {', '.join(groups[:6])}")
    if mechanism:
        lines = mechanism.strip().split("\n")
        print(f"      Primary Mechanism:")
        for line in lines[:4]:
            print(f"        {textwrap.shorten(line.strip(), width=72, placeholder='…')}")
        if len(lines) > 4:
            print(f"        … ({len(lines) - 4} more lines)")
    print()

    # ── 9-Section Safety Profile ──────────────────────────────────
    section_map = [
        (3, "TOXICITY MECHANISM", profile.toxicity_mechanism),
        (4, "AFFECTED ORGANS & SYSTEMS", profile.affected_organs),
        (5, "SYMPTOMS OF EXPOSURE", profile.symptoms_of_exposure),
        (6, "DOSE-RESPONSE DATA", profile.dose_response),
        (7, "FIRST AID & EMERGENCY PROCEDURES", profile.first_aid),
        (8, "HANDLING & STORAGE PRECAUTIONS", profile.handling_precautions),
        (9, "REGULATORY CLASSIFICATION", profile.regulatory_classification),
        (10, "STRUCTURALLY RELATED COMPOUNDS", profile.related_compounds),
        (11, "REFERENCES & DATA SOURCES", profile.references),
        (12, "CONFIDENCE RATIONALE", profile.confidence_rationale),
    ]

    for num, title, content in section_map:
        print(_section_header(num, title))
        print(_wrap_text(content))

    # ── Metadata Footer ───────────────────────────────────────────
    border = "═" * W
    print(f"\n  {border}")
    print(f"   REPORT METADATA")
    print(f"  {'─' * W}")
    sources = ", ".join(profile.retrieval_sources) if profile.retrieval_sources else "None"
    print(_kv("Data Sources", f"{sources}  ({profile.num_retrieved_docs} documents)"))
    print(_kv("LLM Model", profile.llm_model))
    print(_kv("LLM Latency", f"{profile.llm_latency_ms:.0f} ms"))
    print(_kv("Total Pipeline Time", f"{elapsed:.1f} s"))

    sections_filled = sum(1 for s in profile._get_sections().values() if s)
    total_sections = len(profile._get_sections())
    print(_kv("Section Coverage", f"{sections_filled}/{total_sections}"))

    # Extract confidence level from the rationale for quick-scan
    rationale = profile.confidence_rationale.upper()
    if "HIGH" in rationale:
        conf = "HIGH"
    elif "MEDIUM" in rationale or "MODERATE" in rationale:
        conf = "MEDIUM"
    elif "LOW" in rationale:
        conf = "LOW"
    else:
        conf = "HIGH" if sections_filled >= 7 else "MEDIUM" if sections_filled >= 5 else "LOW"
    print(_kv("Confidence", conf))
    print(f"  {border}")
    print(f"      Generated by ToxGuard — For research use only.")
    print(f"      This report does not replace professional toxicological assessment.")
    print(f"  {border}\n")


def print_compact_summary(profile, cot_result=None, elapsed: float = 0.0):
    """Print a compact summary card for non-verbose mode."""
    name = profile.common_name or profile.iupac_name
    toxic_label = "TOXIC" if profile.is_toxic else "NON-TOXIC"
    sections_filled = sum(1 for s in profile._get_sections().values() if s)
    total_sections = len(profile._get_sections())

    print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  {name.upper():<50s} {toxic_label:>14s}  ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  Risk Score : {_risk_bar(profile.toxicity_score):<54s} ║
  ║  Severity   : {profile.severity_label:<54s} ║""")

    if profile.cot_functional_groups:
        groups = ", ".join(profile.cot_functional_groups[:5])
        print(f"  ║  Groups     : {groups:<54s} ║")

    if cot_result and cot_result.confidence_level:
        print(f"  ║  CoT Conf.  : {cot_result.confidence_level:<54s} ║")

    print(f"  ║  Sections   : {sections_filled}/{total_sections} filled  |  {profile.num_retrieved_docs} docs retrieved{' ' * 23} ║")

    sources = ", ".join(profile.retrieval_sources) if profile.retrieval_sources else "none"
    print(f"  ║  Sources    : {sources:<54s} ║")
    print(f"  ║  Time       : {elapsed:.1f}s  |  LLM: {profile.llm_latency_ms:.0f}ms{' ' * 35} ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ToxGuard: Full 3-Phase Toxicity Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("molecules", nargs="*", default=[])
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--no-phase1", action="store_true",
                        help="Skip Phase 1 model (use default score 0.5)")
    parser.add_argument("--no-pubchem", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full Safety Data Sheet report")
    parser.add_argument("--debug", action="store_true",
                        help="Show full Phase 1 + Phase 2 CoT + Phase 3 RAG (developer mode)")
    parser.add_argument("--detox", action="store_true",
                        help="Enable Phase 4: RL detoxification for toxic molecules")
    parser.add_argument("--policy-weights", type=str, default=None,
                        help="Path to trained PPO policy weights (Phase 4)")
    parser.add_argument("--delay", type=float, default=2.0)

    args = parser.parse_args()

    # ── Silence ALL logging — redirect to file ───────────────────
    _silence_all_logging()

    # ── Auto-detect CUDA ──────────────────────────────────────────
    if args.device == "auto":
        try:
            import torch as _t
            args.device = "cuda" if _t.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    if not args.molecules:
        args.molecules = ["nitrobenzene"]
        print("  No molecules specified — using 'nitrobenzene' as demo.\n")

    # ── Banner ───────────────────────────────────────────────────
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                                                                    ║
  ║            T O X G U A R D   P I P E L I N E   v 0 . 2            ║
  ║         ──────────────────────────────────────────────────         ║
  ║    Phase 1 (IUPACGPT) → Phase 2 (CoT) → Phase 3 (RAG) → Phase 4    ║
  ║                                                                    ║
  ╚══════════════════════════════════════════════════════════════════════╝
  """)

    print(f"  Molecules : {', '.join(args.molecules)}")
    print(f"  LLM       : {args.model}")
    print(f"  Phase 1   : {'Enabled' if not args.no_phase1 else 'Disabled'}")
    print(f"  Phase 4   : {'Enabled (RL Detox)' if args.detox else 'Disabled'}")
    print(f"  Log file  : {LOG_FILE}")
    print()

    # ── [1/3] Phase 1 ────────────────────────────────────────────
    predictor = None
    if not args.no_phase1:
        print("  [1/4] Loading Phase 1 model...", end="", flush=True)
        try:
            with _SuppressStdout():
                predictor = load_phase1_predictor(device=args.device)
            print(f"  ✓  (device={args.device})")
        except Exception as e:
            print(f"  ✗  ({e})")
            print("        → Continuing without Phase 1\n")
    else:
        print("  [1/4] Phase 1 skipped")

    # ── [2/3] Phase 2 ────────────────────────────────────────────
    print("  [2/4] Loading Phase 2 CoT analyzer...", end="", flush=True)
    with _SuppressStdout():
        analyzer = load_phase2_analyzer(api_key=args.api_key, model=args.model)
    print("  ✓")

    # ── [3/3] Phase 3 ────────────────────────────────────────────
    print("  [3/4] Loading Phase 3 RAG pipeline...", end="", flush=True)
    with _SuppressStdout():
        rag_mod = _load_phase3_module("rag_pipeline")
        RAGPipeline = rag_mod.RAGPipeline
        pipeline = RAGPipeline(
            vector_store_dir=CHROMA_DIR,
            groq_api_key=args.api_key,
            groq_model=args.model,
            predictor=predictor,
            cot_analyzer=analyzer,
            enable_pubchem=not args.no_pubchem,
        )
        store_count = pipeline.store.count()
    print(f"  ✓  ({store_count:,} docs indexed)")

    if store_count == 0:
        print("\n  ⚠  Vector store is empty! Run: python Phase3-RAG/ingest_t3db.py\n")

    # ── [4/4] Phase 4 (optional) ───────────────────────────────────
    detox_enabled = args.detox
    if detox_enabled:
        run_detox_script = os.path.join(PHASE4_DIR, "run_detox.py")
        if os.path.exists(run_detox_script):
            print("  [4/4] Phase 4 RL detox ready  ✓  (via run_detox.py)")
        else:
            print("  [4/4] Phase 4  ✗  (run_detox.py not found)")
            detox_enabled = False
    else:
        print("  [4/4] Phase 4 skipped (use --detox to enable)")

    print(f"\n  {'─' * 70}")
    print(f"  Starting analysis of {len(args.molecules)} molecule(s)...")
    print(f"  {'─' * 70}")

    # ── Process each molecule ────────────────────────────────────
    profiles = []
    for i, molecule in enumerate(args.molecules):
        print(f"\n  ▶ [{i+1}/{len(args.molecules)}] {molecule}")

        try:
            start = time.time()

            # ── Phase 1: Prediction ──────────────────────────────
            tox_score = 0.5
            severity = "Unknown"
            is_toxic = False
            top_tokens = []
            toxicophore_hits = []

            if predictor:
                print(f"    Phase 1: Predicting...", end="", flush=True)
                with _SuppressStdout():
                    pred = predictor.predict(
                        molecule,
                        return_attention=True,
                        attention_top_k=10,
                    )
                tox_score = pred.toxicity_score
                severity = pred.severity_label
                is_toxic = pred.is_toxic
                top_tokens = pred.top_tokens or []
                toxicophore_hits = pred.toxicophore_hits or []
                print(f"  P(toxic) = {tox_score:.3f}  |  {severity}")
            else:
                print(f"    Phase 1: Skipped (default score 0.5)")

            # ── Phase 2: CoT Reasoning ───────────────────────────
            print(f"    Phase 2: CoT reasoning...", end="", flush=True)
            cot_result = None
            func_groups = []
            cot_mechanism = ""
            cot_pathways = ""

            with _SuppressStdout():
                cot_result = analyzer.analyze_from_prediction(
                    iupac_name=molecule,
                    toxicity_score=tox_score,
                    severity_label=severity,
                    is_toxic=is_toxic,
                    top_tokens=top_tokens,
                    toxicophore_hits=toxicophore_hits,
                )

            func_groups = cot_result.functional_groups
            cot_mechanism = cot_result.mechanism_of_action
            cot_pathways = cot_result.biological_pathways
            groups_str = ", ".join(func_groups[:4]) if func_groups else "—"
            print(f"  Groups: {groups_str}")

            # ── Phase 3: RAG Safety Profile ──────────────────────
            print(f"    Phase 3: RAG synthesis...", end="", flush=True)
            with _SuppressStdout():
                profile = pipeline.generate_safety_profile(
                    iupac_name=molecule,
                    toxicity_score=tox_score,
                    severity_label=severity,
                    is_toxic=is_toxic,
                    functional_groups=func_groups,
                    cot_mechanism=cot_mechanism,
                    cot_pathways=cot_pathways,
                )
            elapsed = time.time() - start
            profiles.append(profile)

            sections_filled = sum(1 for s in profile._get_sections().values() if s)
            total_sections = len(profile._get_sections())
            print(f"  {sections_filled}/{total_sections} sections, {profile.num_retrieved_docs} docs")
            print(f"    ✓ Done in {elapsed:.1f}s")

            # ── Display results ──────────────────────────────────
            if args.debug:
                # Developer mode: show all 3 phases in full
                print(_header_box("PHASE 1: TOXGUARD PREDICTION"))
                print()
                print(f"      Molecule       : {molecule}")
                print(f"      Classification : {'⚠  TOXIC' if is_toxic else '✓  NON-TOXIC'}")
                print(f"      Severity       : {severity}")
                print(f"      Risk Score     : {_risk_bar(tox_score)}")
                if top_tokens:
                    toks = ", ".join(f"{t.get('token','?')} ({t.get('score',0):.2f})" for t in top_tokens[:5])
                    print(f"      Top Attention  : {toks}")
                if toxicophore_hits:
                    hits = ", ".join(f"{h.get('pattern','?')} in '{h.get('fragment','')}'"
                                     for h in toxicophore_hits[:3])
                    print(f"      Toxicophores   : {hits}")
                print(f"  {'═' * W}")

                if cot_result:
                    print_cot_report(cot_result)

                print_safety_report(profile, cot_result, elapsed)

            elif args.verbose:
                # Application mode: clean SDS report only
                print_safety_report(profile, cot_result, elapsed)

            else:
                # Compact summary card
                print_compact_summary(profile, cot_result, elapsed)

            # ── Phase 4: Detoxification (if toxic + detox enabled) ─
            if detox_enabled and is_toxic and tox_score >= 0.5:
                print(f"\n    Phase 4: RL detoxification (via run_detox.py)...")
                try:
                    lora_path = get_lora_weights_path()
                    p4_cmd = [
                        sys.executable, "-W", "ignore",
                        os.path.join(PHASE4_DIR, "run_detox.py"),
                        "detox", molecule,
                        "--checkpoint", CHECKPOINT_DIR,
                        "--lora", lora_path or "",
                        "--tokenizer", SPM_PATH,
                    ]
                    # Add policy weights if provided
                    if args.policy_weights and os.path.exists(args.policy_weights):
                        p4_cmd.extend(["--policy-weights", args.policy_weights])
                    # Add seed toxicity score
                    p4_cmd.extend(["--score", str(tox_score)])
                    # Add verbose flag
                    if args.verbose or args.debug:
                        p4_cmd.append("-v")

                    result = subprocess.run(
                        p4_cmd,
                        cwd=PROJECT_ROOT,
                        capture_output=False,
                        text=True,
                    )
                    if result.returncode != 0:
                        print(f"    ✗ Phase 4 exited with code {result.returncode}")
                except Exception as e:
                    print(f"    ✗ Phase 4 error: ({e})")

        except Exception as e:
            print(f"\n    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

        if i < len(args.molecules) - 1:
            time.sleep(args.delay)

    # ── Save ─────────────────────────────────────────────────────
    if args.output and profiles:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2)
        print(f"\n  ✓ Results saved to: {args.output}")

    # ── Final Summary ────────────────────────────────────────────
    if len(profiles) > 1 or not args.verbose:
        print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  PIPELINE COMPLETE                                                 ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  Analyzed  : {len(profiles)}/{len(args.molecules)} molecules{' ' * 49}║""")
        if profiles:
            toxic_count = sum(1 for p in profiles if p.is_toxic)
            total_s = len(profiles[0]._get_sections())
            avg_s = sum(sum(1 for s in p._get_sections().values() if s) for p in profiles) / len(profiles)
            avg_d = sum(p.num_retrieved_docs for p in profiles) / len(profiles)
            print(f"  ║  Toxic     : {toxic_count}  |  Non-toxic: {len(profiles) - toxic_count}{' ' * 40}║")
            print(f"  ║  Sections  : {avg_s:.1f}/{total_s} avg  |  Docs: {avg_d:.0f} avg{' ' * 31}║")
        print(f"  ║  Log file  : toxguard_pipeline.log{' ' * 33}║")
        print(f"  ╚══════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
