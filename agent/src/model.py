"""Build the agent's chat models: per-site routing, provider-aware structured output.

Each model site resolves its spec as ``<provider>:<model>`` (or ``<provider>/<model>``)
in this order: site env var -> group env var -> ``COPILOTKIT_AGENT_MODEL`` -> code default.
Providers: ``openai``, ``anthropic``, ``google`` (``gemini``, ``google-gemini``),
``xai`` (``grok``). An OpenAI-compatible provider is ``openai:<its model id>`` plus
``OPENAI_BASE_URL``, which the OpenAI client reads itself.
"""

import json
import logging
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "gemini": "google",
    "google-gemini": "google",
    "xai": "xai",
    "grok": "xai",
}
# Our provider ids -> LangChain's init_chat_model provider ids.
_LANGCHAIN_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google_genai",
    "xai": "xai",
}
_SPEC = re.compile(r"^([A-Za-z0-9-]+)[:/](.+)$")

DEFAULT_SPEC = "openai:gpt-5.4"
GLOBAL_ENV = "COPILOTKIT_AGENT_MODEL"
# site -> env vars checked in order (before COPILOTKIT_AGENT_MODEL, then the default).
SITE_ENV: dict[str, tuple[str, ...]] = {
    "security_review": ("SECURITY_MODEL", "REVIEWER_MODEL"),
    "tests_review": ("TESTS_MODEL", "REVIEWER_MODEL"),
    # REGRESSION_MODEL is a deprecated alias of CORRECTNESS_MODEL (the regression reviewer was merged in).
    "correctness_review": ("CORRECTNESS_MODEL", "REGRESSION_MODEL", "REVIEWER_MODEL"),
    "summarize": ("SUMMARY_MODEL",),
    "chat": ("CHAT_MODEL",),
}


def parse_agent_model(value: str, source: str = GLOBAL_ENV) -> tuple[str, str]:
    """Split ``<provider>:<model>`` into (provider, model); the model id is kept unchanged."""
    match = _SPEC.match(value.strip())
    provider = match and _PROVIDER_ALIASES.get(match.group(1).lower())
    if not match or not provider:
        raise ValueError(
            f'{source}="{value}" is not <provider>:<model> '
            "with provider openai, anthropic, google or xai"
        )
    return provider, match.group(2)


_WARNED: set[str] = set()


def _warn_legacy_env() -> None:
    """One-time warnings for env vars from the removed lint / renamed regression reviewers."""
    notes = {
        "LINT_MODEL": "LINT_MODEL is ignored: the lint reviewer was removed (leave lint to CI linters).",
        "REGRESSION_MODEL": "REGRESSION_MODEL is deprecated: use CORRECTNESS_MODEL "
        "(the regression reviewer is now the correctness reviewer).",
    }
    for name, note in notes.items():
        if name not in _WARNED and (os.getenv(name) or "").strip():
            _WARNED.add(name)
            logger.warning(note)


def resolve_model_spec(site: str, default_spec: str = DEFAULT_SPEC) -> str:
    """The spec string this site uses (validated); raises ValueError naming the env var."""
    _warn_legacy_env()
    for name in (*SITE_ENV.get(site, ()), GLOBAL_ENV):
        value = (os.getenv(name) or "").strip()
        if value:
            parse_agent_model(value, name)
            return value
    parse_agent_model(default_spec, f"default model for {site}")
    return default_spec


def create_chat_model(
    default_spec: str,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    site: str | None = None,
    **openai_kwargs,
) -> BaseChatModel:
    """Build a chat model.

    Args:
        default_spec: The default, e.g. ``"openai:gpt-5.4"``, used when no env var applies.
        timeout: Request timeout in seconds (all providers).
        max_retries: Client retry count (all providers).
        site: Model site (key of ``SITE_ENV``) for per-site routing. Without it only
            ``COPILOTKIT_AGENT_MODEL`` overrides the default.
        **openai_kwargs: Extra ChatOpenAI options, applied only for OpenAI.
    """
    spec = resolve_model_spec(site, default_spec) if site else None
    if spec is None:
        spec = (os.getenv(GLOBAL_ENV) or "").strip() or default_spec
    provider, model = parse_agent_model(spec, GLOBAL_ENV if not site else f"model for {site}")
    kwargs = dict(openai_kwargs) if provider == "openai" else {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    # No temperature/top_p/top_k: current Anthropic models reject them (400).
    return init_chat_model(
        model, model_provider=_LANGCHAIN_PROVIDERS[provider], **kwargs
    )


def structured(model: BaseChatModel, schema, provider: str):
    """``model.with_structured_output(schema)`` using the provider's safe method.

    Anthropic's current models reject forced ``tool_choice`` (LangChain's default
    ``function_calling``), so use native ``output_config.format``. xAI uses its native
    structured outputs too. OpenAI and Google keep LangChain's defaults.

    ``include_raw=True`` makes the runnable return ``{"raw": AIMessage, "parsed", "parsing_error"}``
    so token usage (``raw.usage_metadata``) is readable on every provider; unpack it with
    ``unpack_structured``.
    """
    if provider in ("anthropic", "xai"):
        return model.with_structured_output(schema, method="json_schema", include_raw=True)
    return model.with_structured_output(schema, include_raw=True)


# ---------------------------------------------------------------- token usage and cost

# (provider, model-id prefix) -> USD per 1M tokens. Only prices known for certain; every other
# model has no price (cost is null) unless the user supplies ``MODEL_PRICES``.
PRICES: dict[tuple[str, str], dict[str, float]] = {
    ("anthropic", "claude-opus-5-5"): {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    ("anthropic", "claude-sonnet-5-5"): {"input": 2.0, "output": 10.0, "cache_read": 0.20},
    ("anthropic", "claude-haiku-5-5"): {"input": 0.10, "output": 0.50},
}
_BAD_PRICES_WARNED = False


def _user_prices() -> dict[tuple[str, str], dict[str, float]]:
    """``MODEL_PRICES`` env JSON: ``{"openai:gpt-5.4": {"input": x, "output": y, "cache_read": z}}``."""
    global _BAD_PRICES_WARNED
    raw = (os.getenv("MODEL_PRICES") or "").strip()
    if not raw:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    try:
        for spec, price in json.loads(raw).items():
            key = parse_agent_model(spec, "MODEL_PRICES")
            out[key] = {k: float(price[k]) for k in ("input", "output", "cache_read") if k in price}
            if "input" not in out[key] or "output" not in out[key]:
                raise ValueError(f"{spec} needs input and output")
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        if not _BAD_PRICES_WARNED:
            _BAD_PRICES_WARNED = True
            logger.warning("MODEL_PRICES ignored: %s", e)
        return {}
    return out


def price_for(spec: str | None) -> dict[str, float] | None:
    """Per-1M-token prices for a ``<provider>:<model>`` spec, or None when unknown."""
    if not spec:
        return None
    try:
        provider, model = parse_agent_model(spec)
    except ValueError:
        return None
    user = _user_prices()
    if (provider, model) in user:
        return user[(provider, model)]
    matches = [k for k in PRICES if k[0] == provider and model.startswith(k[1])]
    return PRICES[max(matches, key=lambda k: len(k[1]))] if matches else None


def usage_from_message(msg) -> dict[str, int]:
    """``{"input_tokens", "output_tokens", "cached_tokens"}`` from a message's ``usage_metadata``.

    LangChain normalizes every provider to this shape; ``input_tokens`` already includes
    cached tokens (``input_token_details.cache_read``). Missing metadata yields zeros.
    """
    um = getattr(msg, "usage_metadata", None) or {}
    details = um.get("input_token_details") or {}
    return {
        "input_tokens": int(um.get("input_tokens") or 0),
        "output_tokens": int(um.get("output_tokens") or 0),
        "cached_tokens": int(details.get("cache_read") or 0),
    }


def cost_usd(spec: str | None, usage: dict[str, int]) -> float | None:
    """USD cost of one call; cached tokens bill at ``cache_read`` (else the input price). None if unpriced."""
    price = price_for(spec)
    if price is None:
        return None
    cached = min(usage["cached_tokens"], usage["input_tokens"])
    cache_price = price.get("cache_read", price["input"])
    return ((usage["input_tokens"] - cached) * price["input"] + cached * cache_price
            + usage["output_tokens"] * price["output"]) / 1_000_000


def usage_record(spec: str | None, msg, files: int | None = None) -> dict:
    """A ``node_usage`` entry for one LLM call (``files`` only for reviewers)."""
    usage = usage_from_message(msg)
    cost = cost_usd(spec, usage)
    rec: dict = {**usage, "cost_usd": None if cost is None else round(cost, 9)}
    if files is not None:
        rec["files"] = files
    return rec


class StructuredOutputError(ValueError):
    """Structured output failed to parse; ``raw`` is the model message (its usage was still spent)."""

    def __init__(self, message: str, raw=None):
        super().__init__(message)
        self.raw = raw


def unpack_structured(result, schema):
    """``(parsed, raw_message | None)`` from a ``structured()`` result.

    Handles the ``include_raw`` dict (raising on a parse failure) and a bare parsed object/dict.
    """
    if isinstance(result, dict) and "raw" in result and "parsed" in result:
        parsed = result["parsed"]
        raw = result["raw"]
        if parsed is None:
            err = result.get("parsing_error")
            raise StructuredOutputError(f"structured output could not be parsed: {err}", raw) from err
    else:
        parsed, raw = result, None
    if isinstance(parsed, dict):
        try:
            parsed = schema.model_validate(parsed)
        except ValueError as e:
            raise StructuredOutputError(f"structured output failed validation: {e}", raw) from e
    return parsed, raw
