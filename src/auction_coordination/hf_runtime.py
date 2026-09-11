"""Hugging Face adapter for bounded Agent 11 structured work products.

Provider-specific request shaping remains outside the auction control plane. Each
live call sends the exact Pydantic JSON Schema for ModelWorkDraft and the returned
content still passes through the same local JSON parsing, evidence validation, and
WorkProduct validation used by the application.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import (
    HuggingFaceChatClient,
    ModelConfigurationError,
    ModelWorkDraft,
    StructuredModelError,
)

_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[^\s,;]+")
_HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9_-]{6,}\b")
_SUPPORTED_TASKS = {
    "market_research",
    "financial_analysis",
    "operating_risk",
    "evidence_verification",
    "executive_synthesis",
}


class ModelInvocationError(RuntimeError):
    """Provider/network/auth failure distinct from invalid model content."""


def _safe_provider_error(exc: Exception, *, token: str) -> str:
    """Return a bounded provider diagnostic without leaking credentials."""

    raw = " ".join(str(exc).split())
    if token:
        raw = raw.replace(token, "[redacted]")
    raw = _HF_TOKEN_PATTERN.sub("[redacted]", raw)
    raw = _BEARER_PATTERN.sub("Bearer [redacted]", raw)
    raw = raw[:500]

    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    parts = ["Hugging Face provider request failed"]
    if status is not None:
        parts.append(f"status={status}")
    if code:
        parts.append(f"code={code}")
    if raw:
        parts.append(raw)
    return ": ".join(parts)


@dataclass(frozen=True)
class HuggingFaceStructuredChatClient(HuggingFaceChatClient):
    """Hugging Face client hardened for strict Agent 11 JSON work drafts."""

    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if task_name not in _SUPPORTED_TASKS:
            raise StructuredModelError(
                f"no provider JSON schema registered for bounded task: {task_name}"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ModelConfigurationError(
                "openai package is required for live Hugging Face LLM mode"
            ) from exc

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": ModelWorkDraft.__name__,
                "schema": ModelWorkDraft.model_json_schema(),
                "strict": True,
            },
        }
        bounded_system_prompt = (
            f"{system_prompt}\n\n"
            "Output-size requirement: keep the summary under 1200 characters, use no more "
            "than 6 concise findings, and cite only supplied evidence IDs."
        )

        client = OpenAI(base_url=self.base_url, api_key=self.token)
        request: dict[str, object] = {
            "model": self.model_id,
            "messages": (
                {"role": "system", "content": bounded_system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": response_format,
        }

        try:
            completion = client.chat.completions.create(**request)
        except Exception as exc:  # provider/network/auth failures fail closed
            raise ModelInvocationError(
                _safe_provider_error(exc, token=self.token)
            ) from exc

        choice = completion.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise StructuredModelError(
                f"model structured response was truncated at max_tokens={self.max_tokens}"
            )
        if finish_reason not in {None, "stop"}:
            raise StructuredModelError(
                "model structured response ended unexpectedly: "
                f"finish_reason={finish_reason}"
            )

        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise StructuredModelError("model returned an empty chat response")
        return content
