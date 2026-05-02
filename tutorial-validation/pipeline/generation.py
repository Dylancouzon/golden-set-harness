"""Grounded generator. Routes Claude vs GPT based on model name prefix.

Both providers are needed because the demo lets the presenter swap generators
in the sidebar. Returns (answer, usage) where usage is a dict of token counts
that the cost meter aggregates.
"""
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
    """Run a grounded completion.

    Returns:
        (answer_text, usage_dict). usage_dict has input_tokens / output_tokens
        and is used by the demo's cost meter.
    """
    prompt = prompt_template.format(
        retrieved_context="\n\n".join(contexts),
        query_text=query_text,
    )

    # Anthropic branch — straightforward, the modern Claude SDK is consistent
    # across model versions.
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

    # OpenAI branch — has a parameter-naming quirk:
    #   - Legacy chat models (gpt-4o, gpt-4-turbo, ...) accept `max_tokens`.
    #   - Reasoning-class models (gpt-5+, o-series) reject `max_tokens` and
    #     require `max_completion_tokens` instead.
    # LangChain and DeepEval translate this internally for their wrapped
    # calls, but the raw OpenAI client used here does not. We try the legacy
    # name first and retry with the new name on the specific BadRequestError.
    # See results/tutorial_gaps.md for the full context.
    try:
        resp = _openai().chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            resp = _openai().chat.completions.create(
                model=model,
                max_completion_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
    }
    return resp.choices[0].message.content or "", usage
