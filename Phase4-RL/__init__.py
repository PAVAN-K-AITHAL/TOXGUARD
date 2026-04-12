"""Phase 4: RL-Guided Molecule Detoxification + Agentic Workflow.

Given a toxic molecule, generates structurally similar but less toxic
variants using:
    1. IUPAC-GPT as the policy network (PPO with new LoRA adapters)
    2. ToxGuard Phase 1 as the reward model (frozen)
    3. RDKit for chemical validity checks
    4. An agentic loop for iterative refinement

Pipeline:
    Toxic IUPAC Name
        → IUPAC-GPT generates candidate IUPAC names (PPO-trained)
        → py2opsin converts to SMILES
        → RDKit validates + computes properties (Tanimoto, QED, SA)
        → ToxGuard Phase 1 scores toxicity
        → Multi-objective reward: detox + similarity + drug-likeness
        → Agent loop: adjust strategy if no valid less-toxic candidate found
        → Output: Detoxified molecule report

Usage:
    # Standalone
    python Phase4-RL/run_detox.py "nitrobenzene" --score 0.91 -v

    # Via main pipeline
    python main.py "nitrobenzene" -v --detox
"""

__version__ = "0.1.0"
