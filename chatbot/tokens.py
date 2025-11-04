"""
Token tracking utilities for LLM calls.
Provides callback handlers for both native token tracking (OpenAI, Anthropic)
and fallback estimation for other models.
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
import tiktoken
import logging

logger = logging.getLogger(__name__)


class TokenCountingCallback(AsyncCallbackHandler):
    """
    Async callback handler to track token usage from LLM calls.

    Supports:
    - Native token tracking from providers (OpenAI, Anthropic, etc.)
    - Fallback token estimation using tiktoken for models without native tracking
    """

    def __init__(self, model_id: str, provider_id: str = None):
        super().__init__()
        self.model_id = model_id
        self.provider_id = provider_id
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated = False
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load tokenizer for fallback estimation."""
        if self._tokenizer is None:
            try:
                # Try to get encoding for the specific model
                self._tokenizer = tiktoken.encoding_for_model(self.model_id)
            except KeyError:
                # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
                logger.warning(f"No tokenizer found for {self.model_id}, using cl100k_base")
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using tiktoken."""
        if not text:
            return 0
        try:
            tokenizer = self._get_tokenizer()
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            # Very rough fallback: ~4 chars per token
            return len(text) // 4

    def _estimate_message_tokens(self, messages: List[Any]) -> int:
        """Estimate tokens for a list of messages."""
        total = 0
        for message in messages:
            if hasattr(message, 'content'):
                total += self._estimate_tokens(str(message.content))
            elif isinstance(message, dict) and 'content' in message:
                total += self._estimate_tokens(str(message['content']))
            elif isinstance(message, str):
                total += self._estimate_tokens(message)
        return total

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts - estimate prompt tokens if needed."""
        # This is called for older LangChain interfaces
        if prompts:
            estimated_prompt_tokens = sum(self._estimate_tokens(p) for p in prompts)
            logger.debug(f"Estimated prompt tokens from on_llm_start: {estimated_prompt_tokens}")

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any
    ) -> None:
        """Called when chat model starts - can be used for estimation."""
        # Store input for potential fallback estimation
        self._input_messages = messages

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Called when LLM ends - extract token usage from response metadata.
        If not available, fall back to estimation.
        """
        try:
            # Log the full response structure for debugging
            logger.debug(f"LLM Response structure - llm_output keys: {response.llm_output.keys() if response.llm_output else 'None'}")
            logger.debug(f"LLM Response - generations count: {len(response.generations) if response.generations else 0}")

            # Try to extract token usage from response metadata
            if response.llm_output and 'token_usage' in response.llm_output:
                token_usage = response.llm_output['token_usage']
                self.prompt_tokens = token_usage.get('prompt_tokens', 0)
                self.completion_tokens = token_usage.get('completion_tokens', 0)
                self.total_tokens = token_usage.get('total_tokens', 0)
                self.estimated = False
                logger.debug(f"Native token tracking (token_usage): {self.total_tokens} total tokens")

            # Check in usage_metadata (newer LangChain format)
            elif response.llm_output and 'usage_metadata' in response.llm_output:
                usage = response.llm_output['usage_metadata']
                self.prompt_tokens = usage.get('input_tokens', 0)
                self.completion_tokens = usage.get('output_tokens', 0)
                self.total_tokens = usage.get('total_tokens', 0)
                self.estimated = False
                logger.debug(f"Native token tracking (usage_metadata): {self.total_tokens} total tokens")

            # Check for response_metadata (common in AWS Bedrock)
            elif response.llm_output and 'response_metadata' in response.llm_output:
                metadata = response.llm_output['response_metadata']
                logger.debug(f"Found response_metadata: {metadata}")

                # AWS Bedrock format
                if 'usage' in metadata:
                    usage = metadata['usage']
                    self.prompt_tokens = usage.get('inputTokens', usage.get('prompt_tokens', 0))
                    self.completion_tokens = usage.get('outputTokens', usage.get('completion_tokens', 0))
                    self.total_tokens = usage.get('totalTokens', self.prompt_tokens + self.completion_tokens)
                    self.estimated = False
                    logger.info(f"Native token tracking (Bedrock): {self.total_tokens} total tokens (prompt: {self.prompt_tokens}, completion: {self.completion_tokens})")
                else:
                    # Fallback to estimation
                    self._estimate_from_response(response)

            # Check generations for token info (Anthropic format)
            elif response.generations and len(response.generations) > 0:
                gen = response.generations[0][0]
                logger.debug(f"Generation attributes: {dir(gen)}")

                # Check message.usage_metadata
                if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                    usage = gen.message.usage_metadata
                    logger.debug(f"Found usage_metadata on message: {usage}")

                    # usage_metadata can be a dict or an object
                    if isinstance(usage, dict):
                        self.prompt_tokens = usage.get('input_tokens', 0)
                        self.completion_tokens = usage.get('output_tokens', 0)
                        self.total_tokens = usage.get('total_tokens', 0)
                    else:
                        self.prompt_tokens = getattr(usage, 'input_tokens', 0)
                        self.completion_tokens = getattr(usage, 'output_tokens', 0)
                        self.total_tokens = getattr(usage, 'total_tokens', 0)

                    self.estimated = False
                    logger.info(f"Native token tracking (generation metadata): {self.total_tokens} total tokens (prompt: {self.prompt_tokens}, completion: {self.completion_tokens})")

                # Check message.response_metadata
                elif hasattr(gen, 'message') and hasattr(gen.message, 'response_metadata'):
                    metadata = gen.message.response_metadata
                    logger.debug(f"Found response_metadata on message: {metadata}")
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        self.prompt_tokens = usage.get('inputTokens', usage.get('input_tokens', 0))
                        self.completion_tokens = usage.get('outputTokens', usage.get('output_tokens', 0))
                        self.total_tokens = usage.get('totalTokens', self.prompt_tokens + self.completion_tokens)
                        self.estimated = False
                        logger.info(f"Native token tracking (message response_metadata): {self.total_tokens} total tokens")
                    else:
                        self._estimate_from_response(response)
                else:
                    # Fallback to estimation
                    self._estimate_from_response(response)
            else:
                # Fallback to estimation
                self._estimate_from_response(response)

        except Exception as e:
            logger.warning(f"Error extracting token usage, falling back to estimation: {e}", exc_info=True)
            self._estimate_from_response(response)

    def _estimate_from_response(self, response: LLMResult):
        """Estimate tokens when native tracking is unavailable."""
        self.estimated = True

        # Estimate prompt tokens from input messages if available
        if hasattr(self, '_input_messages') and self._input_messages:
            self.prompt_tokens = sum(
                self._estimate_message_tokens(msg_list)
                for msg_list in self._input_messages
            )

        # Estimate completion tokens from response
        completion_text = ""
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, 'text'):
                        completion_text += gen.text
                    elif hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                        completion_text += str(gen.message.content)

        self.completion_tokens = self._estimate_tokens(completion_text)
        self.total_tokens = self.prompt_tokens + self.completion_tokens

        logger.debug(
            f"Estimated tokens for {self.model_id}: "
            f"{self.prompt_tokens} prompt + {self.completion_tokens} completion = {self.total_tokens} total"
        )

    def get_token_counts(self) -> Dict[str, Any]:
        """Return the collected token counts."""
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'estimated': self.estimated
        }

    def reset(self):
        """Reset token counts for reuse."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated = False
        if hasattr(self, '_input_messages'):
            delattr(self, '_input_messages')
