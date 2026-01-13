"""OpenRouter LLM integration for answer generation."""
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """OpenRouter API wrapper for LLM calls."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model to use
            base_url: OpenRouter API base URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.client = None

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max_tokens
            temperature: Override temperature

        Returns:
            Generated text
        """
        if self.client is None:
            return "Error: LLM client not initialized."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {str(e)}"

    def answer_question(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Answer a question using retrieved context.

        Args:
            query: User question
            context_chunks: Retrieved context chunks

        Returns:
            Generated answer
        """
        from .prompts import build_rag_prompt

        prompt = build_rag_prompt(query, context_chunks)

        return self.generate(prompt)
