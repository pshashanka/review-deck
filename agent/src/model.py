"""Build the agent's chat model from ``COPILOTKIT_AGENT_MODEL``.

``COPILOTKIT_AGENT_MODEL=<provider>:<model>`` (or ``<provider>/<model>``)
overrides EVERY model site in this agent. When it is unset, each site falls
back to its own default. Providers: ``openai``, ``anthropic``, ``google``
(``gemini`` and ``google-gemini`` are aliases of ``google``). An
OpenAI-compatible provider is ``openai:<its model id>`` plus
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
}
# Our provider ids -> LangChain's init_chat_model provider ids.
_LANGCHAIN_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google_genai",
}
_SPEC = re.compile(r"^([A-Za-z0-9-]+)[:/](.+)$")


def parse_agent_model(value: str) -> tuple[str, str]:
    """Split ``<provider>:<model>`` into (provider, model); the model id is kept unchanged."""
    match = _SPEC.match(value.strip())
    provider = match and _PROVIDER_ALIASES.get(match.group(1).lower())
    if not match or not provider:
        raise ValueError(
            f'COPILOTKIT_AGENT_MODEL="{value}" is not <provider>:<model> '
            "with provider openai, anthropic or google"
        )
    return provider, match.group(2)


def create_chat_model(
    default_spec: str,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    **openai_kwargs,
) -> BaseChatModel:
    """Build this site's chat model.

    Args:
        default_spec: This site's default, e.g. ``"openai:gpt-5.4"``.
        timeout: Request timeout in seconds (all providers).
        max_retries: Client retry count (all providers).
        **openai_kwargs: Extra ChatOpenAI options, applied only for OpenAI.
    """
    provider, model = parse_agent_model(
        (os.getenv("COPILOTKIT_AGENT_MODEL") or "").strip() or default_spec
    )
    kwargs = dict(openai_kwargs) if provider == "openai" else {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return init_chat_model(
        model, model_provider=_LANGCHAIN_PROVIDERS[provider], **kwargs
    )
