"""Optional OpenAI-compatible hosted generation; local retrieval does not call this."""
from __future__ import annotations

import os

from dotenv import load_dotenv


class LLMConfigurationError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


def is_configured() -> bool:
    load_dotenv()
    return bool(os.getenv("OPENAI_API_KEY", "").strip() and os.getenv("OPENAI_MODEL", "").strip())


def generate_answer(messages: list[dict[str, str]]) -> str:
    """Make exactly one provider request, only when explicitly selected by user.

    No key is logged and the exception messages avoid echoing provider payloads.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        raise LLMConfigurationError(
            "Set OPENAI_API_KEY and OPENAI_MODEL in a local .env file to use LLM answers."
        )
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        timeout=40.0,
        max_retries=1,
    )
    try:
        completion = client.chat.completions.create(model=model, messages=messages)
    except RateLimitError as exc:
        raise LLMProviderError("LLM provider rate limit reached. Try again later.") from exc
    except APITimeoutError as exc:
        raise LLMProviderError("LLM provider request timed out.") from exc
    except APIConnectionError as exc:
        raise LLMProviderError("Cannot connect to the configured LLM provider.") from exc
    except APIStatusError as exc:
        raise LLMProviderError(f"LLM provider returned HTTP {exc.status_code}; check credentials, model and endpoint.") from exc
    if not completion.choices:
        raise LLMProviderError("LLM provider returned no response choices.")
    return completion.choices[0].message.content or ""
