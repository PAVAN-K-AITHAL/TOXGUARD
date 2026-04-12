"""ToxGuard RAG Evaluation Framework.

4-pillar RAGAS-powered validation:
    Pillar 1: eval_retrieval     — IR metrics + RAGAS context metrics
    Pillar 2: eval_faithfulness  — RAGAS faithfulness + domain hard gates
    Pillar 3: eval_human         — Golden test set auto-scoring + annotation sheets
    Pillar 4: eval_phase4_retrieval — Phase 4 structural-analogy retrieval tests

Utilities:
    validate_rag    — Unified 4-pillar orchestrator
    evaluate_rag    — Legacy entry point with --validate flag
    feedback_loop   — Expert annotations → tuning recommendations

Usage:
    python -m Phase3-RAG.evaluation.validate_rag --all --no-llm
"""
