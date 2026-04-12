"""Phase 2: Chain-of-Thought Toxicity Reasoning.

Provides mechanistic explanations for ToxGuard toxicity predictions
using few-shot CoT prompting with LLMs.

Pipeline:
    1. ToxGuard Phase 1 → P(toxic) + attention scores + toxicophore hits
    2. Build structured CoT prompt with few-shot exemplars
    3. Query LLM (Groq/Llama-3.3-70b) for step-by-step reasoning
    4. Parse structured output → functional groups, mechanisms, pathways

Reference: CoTox (Park et al., BIBM 2025)
"""

__version__ = "0.1.0"

from .cot_analyzer import CoTAnalyzer, CoTResult
from .llm_client import GroqLLMClient, LLMClient

__all__ = [
    "CoTAnalyzer",
    "CoTResult",
    "GroqLLMClient",
    "LLMClient",
]
