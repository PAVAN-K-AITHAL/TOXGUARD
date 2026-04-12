"""LLM client wrapper for Phase 2 CoT reasoning.

Supports Groq API (primary) with extensible base class for other providers.
Uses llama-3.3-70b-versatile for production CoT reasoning
and llama-3.1-8b-instant for fast development/testing.

Usage:
    client = GroqLLMClient(api_key="gsk_...")
    response = client.generate(
        system_prompt="You are a toxicologist...",
        user_prompt="Analyze this molecule..."
    )
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str                         # Raw text response
    model: str                           # Model used
    usage: Dict[str, int] = field(       # Token usage stats
        default_factory=dict
    )
    latency_ms: float = 0.0             # Request latency in milliseconds
    finish_reason: str = ""              # "stop", "length", etc.

    def __repr__(self):
        tokens = self.usage.get("total_tokens", "?")
        return (f"LLMResponse(model={self.model!r}, "
                f"tokens={tokens}, latency={self.latency_ms:.0f}ms)")


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API is reachable."""
        ...


class GroqLLMClient(LLMClient):
    """Groq API client for fast LLM inference.

    Models:
        - llama-3.3-70b-versatile: Best for CoT reasoning (production)
        - llama-3.1-8b-instant: Fast, good for development/testing
        - mixtral-8x7b-32768: Alternative with 32K context

    Rate limits (free tier):
        - llama-3.3-70b: 6,000 req/day, 6,000 tokens/min
        - llama-3.1-8b: 14,400 req/day, 6,000 tokens/min
    """

    # Available models on Groq
    MODELS = {
        "70b": "llama-3.3-70b-versatile",
        "8b": "llama-3.1-8b-instant",
        "mixtral": "mixtral-8x7b-32768",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize Groq client.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model name or shorthand ("70b", "8b", "mixtral").
            max_retries: Max retry attempts on rate limit / transient errors.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter.\n"
                "Get your key at: https://console.groq.com/keys"
            )

        # Resolve model shorthand
        self.model = self.MODELS.get(model, model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Lazy import — only needed when actually using Groq
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "groq package not installed. Install with:\n"
                "  pip install groq"
            )

        logger.info(f"Groq client initialized with model: {self.model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response using Groq API.

        Args:
            system_prompt: System message (role/instructions).
            user_prompt: User message (the actual query).
            temperature: Sampling temperature (0.0-1.0). Lower = more
                deterministic. 0.3 recommended for CoT reasoning.
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    stream=False,
                )
                latency = (time.time() - start_time) * 1000

                choice = completion.choices[0]
                usage = {}
                if completion.usage:
                    usage = {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    }

                response = LLMResponse(
                    content=choice.message.content or "",
                    model=self.model,
                    usage=usage,
                    latency_ms=latency,
                    finish_reason=choice.finish_reason or "",
                )
                logger.debug(f"Groq response: {response}")
                return response

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(kw in error_str for kw in [
                    "rate_limit", "429", "503", "timeout", "connection",
                    "overloaded",
                ])
                if is_retryable and attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Groq API error (attempt {attempt+1}/{self.max_retries}): "
                        f"{e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Groq API error: {e}")
                    raise

    def is_available(self) -> bool:
        """Check if Groq API is reachable with a minimal request."""
        try:
            response = self.generate(
                system_prompt="Reply with exactly: OK",
                user_prompt="Hello",
                temperature=0.0,
                max_tokens=5,
            )
            return len(response.content) > 0
        except Exception as e:
            logger.warning(f"Groq API not available: {e}")
            return False

    def switch_model(self, model: str):
        """Switch to a different model (e.g., for dev vs. production).

        Args:
            model: Model name or shorthand ("70b", "8b", "mixtral").
        """
        self.model = self.MODELS.get(model, model)
        logger.info(f"Switched to model: {self.model}")
