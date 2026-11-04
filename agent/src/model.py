"""Build the agent's chat models: per-site routing, provider-aware structured output.

Each model site resolves its spec as ``<provider>:<model>`` (or ``<provider>/<model>``)
in this order: site env var -> group env var -> ``COPILOTKIT_AGENT_MODEL`` -> code default.
Providers: ``openai``, ``anthropic``, ``google`` (``gemini``, ``google-gemini``),
``xai`` (``grok``). An OpenAI-compatible provider is ``openai:<its model id>`` plus
``OPENAI_BASE_URL``, which the OpenAI client reads itself.
"""

import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

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
    "lint_review": ("LINT_MODEL", "REVIEWER_MODEL"),
    "security_review": ("SECURITY_MODEL", "REVIEWER_MODEL"),
    "tests_review": ("TESTS_MODEL", "REVIEWER_MODEL"),
    "regression_review": ("REGRESSION_MODEL", "REVIEWER_MODEL"),
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


def resolve_model_spec(site: str, default_spec: str = DEFAULT_SPEC) -> str:
    """The spec string this site uses (validated); raises ValueError naming the env var."""
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
    """
    if provider in ("anthropic", "xai"):
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)
