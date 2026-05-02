"""Grounded generator. Routes Claude vs GPT based on model name prefix."""
from __future__ import annotations

from functools import lru_cache

import anthropic
from openai import OpenAI

from config import settings


@lru_cache(maxsize=1)
def _anthropic() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def generate(
    query_text: str,
    contexts: list[str],
    *,
    model: str,
    prompt_template: str,
) -> tuple[str, dict[str, int]]:
    """Return (answer_text, usage_dict). usage_dict has input_tokens/output_tokens for cost tracking."""
    prompt = prompt_template.format(
        retrieved_context="\n\n".join(contexts),
        query_text=query_text,
    )

    if model.startswith("claude"):
        resp = _anthropic().messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return resp.content[0].text, usage

    resp = _openai().chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
    }
    return resp.choices[0].message.content or "", usage
