from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from auction_coordination.hf_runtime import (
    HuggingFaceStructuredChatClient,
    ModelInvocationError,
)
from auction_coordination.llm import StructuredModelError


class RecordingOpenAI:
    last_request: dict[str, object] | None = None

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        type(self).last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"title":"Review","summary":"Synthetic summary",'
                            '"findings":["Finding"],'
                            '"evidence_ids":["evidence.market.1"]}'
                        )
                    ),
                )
            ]
        )


class TruncatedOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"title":"cut off'),
                )
            ]
        )


class FailingOpenAI:
    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        del kwargs
        raise RuntimeError("provider rejected Bearer hf_secret_runtime_token")


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, client_type: type) -> None:
    module = ModuleType("openai")
    module.OpenAI = client_type
    monkeypatch.setitem(sys.modules, "openai", module)


def test_provider_request_uses_strict_model_work_draft_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="example/model",
        token="hf_example",
    )

    content = client.complete_json(
        task_name="market_research",
        system_prompt="system",
        user_prompt="user",
    )
    assert "Synthetic summary" in content

    request = RecordingOpenAI.last_request
    assert request is not None
    response_format = request["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["name"] == "ModelWorkDraft"
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert {"title", "summary", "findings", "evidence_ids"} == set(
        schema["properties"]
    )
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 1800
    assert "response_format" in request


def test_all_five_work_capabilities_are_registered_for_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    client = HuggingFaceStructuredChatClient(
        model_id="example/model",
        token="hf_example",
    )
    for task_name in (
        "market_research",
        "financial_analysis",
        "operating_risk",
        "evidence_verification",
        "executive_synthesis",
    ):
        client.complete_json(
            task_name=task_name,
            system_prompt="system",
            user_prompt="user",
        )


def test_unknown_task_fails_before_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, RecordingOpenAI)
    RecordingOpenAI.last_request = None
    client = HuggingFaceStructuredChatClient(
        model_id="example/model",
        token="hf_example",
    )

    with pytest.raises(StructuredModelError, match="no provider JSON schema registered"):
        client.complete_json(
            task_name="auction_scoring",
            system_prompt="system",
            user_prompt="user",
        )

    assert RecordingOpenAI.last_request is None


def test_length_finish_reason_fails_closed_before_local_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, TruncatedOpenAI)
    client = HuggingFaceStructuredChatClient(
        model_id="example/model",
        token="hf_example",
    )

    with pytest.raises(StructuredModelError, match="truncated at max_tokens=1800"):
        client.complete_json(
            task_name="financial_analysis",
            system_prompt="system",
            user_prompt="user",
        )


def test_provider_failure_redacts_token_and_has_distinct_failure_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, FailingOpenAI)
    client = HuggingFaceStructuredChatClient(
        model_id="example/model",
        token="hf_secret_runtime_token",
    )

    with pytest.raises(ModelInvocationError) as exc_info:
        client.complete_json(
            task_name="operating_risk",
            system_prompt="system",
            user_prompt="user",
        )

    message = str(exc_info.value)
    assert "Hugging Face provider request failed" in message
    assert "hf_secret_runtime_token" not in message
    assert "Bearer [redacted]" in message
