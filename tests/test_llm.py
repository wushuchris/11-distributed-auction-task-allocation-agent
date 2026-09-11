import json
from datetime import datetime, timezone

import pytest

from auction_coordination.execution import DeterministicTaskExecutor
from auction_coordination.llm import (
    HF_DEFAULT_BASE_URL,
    HuggingFaceChatClient,
    ModelConfigurationError,
    build_llm_handlers,
)
from auction_coordination.runtime import DeterministicMissionRuntime, TaskRunStatus
from auction_coordination.scenario import (
    build_default_registry,
    default_bid_settings,
    default_task_blueprints,
)


STARTED_AT = datetime(2026, 9, 11, 16, 0, tzinfo=timezone.utc)


class ValidFakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def complete_json(self, *, task_name: str, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        self.calls.append((task_name, payload))
        evidence = payload["evidence"]
        evidence_ids = [item["evidence_id"] for item in evidence]
        assert evidence_ids
        assert "already been awarded" in system_prompt
        return json.dumps(
            {
                "title": f"LLM {task_name} review",
                "summary": f"Bounded synthetic {task_name} analysis.",
                "findings": [f"Synthetic finding for {task_name}."],
                "evidence_ids": evidence_ids[:2],
            }
        )


class InventedEvidenceModel:
    def complete_json(self, *, task_name: str, system_prompt: str, user_prompt: str) -> str:
        del task_name, system_prompt, user_prompt
        return json.dumps(
            {
                "title": "Unsupported evidence draft",
                "summary": "This draft cites an evidence ID the application never supplied.",
                "findings": ["Unsupported synthetic claim."],
                "evidence_ids": ["evidence.invented.999"],
            }
        )


class ExtraControlFieldModel:
    def complete_json(self, *, task_name: str, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        del task_name, system_prompt
        evidence_id = payload["evidence"][0]["evidence_id"]
        return json.dumps(
            {
                "title": "Attempted control-plane mutation",
                "summary": "The model tried to return an application-owned field.",
                "findings": ["The strict schema should reject this output."],
                "evidence_ids": [evidence_id],
                "awarded_agent_id": "peer.mosaic",
            }
        )


class MalformedJsonModel:
    def complete_json(self, *, task_name: str, system_prompt: str, user_prompt: str) -> str:
        del task_name, system_prompt, user_prompt
        return "not-json"


def _runtime(model) -> DeterministicMissionRuntime:
    registry = build_default_registry()
    executor = DeterministicTaskExecutor(
        registry,
        handlers=build_llm_handlers(model),
    )
    return DeterministicMissionRuntime(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
        executor=executor,
    )


def test_hugging_face_client_from_env_requires_runtime_token() -> None:
    with pytest.raises(ModelConfigurationError, match="HF_TOKEN"):
        HuggingFaceChatClient.from_env(
            {"MODEL_ID": "example/model", "HF_BASE_URL": HF_DEFAULT_BASE_URL}
        )


def test_hugging_face_client_from_env_requires_model_id() -> None:
    with pytest.raises(ModelConfigurationError, match="MODEL_ID"):
        HuggingFaceChatClient.from_env(
            {"HF_TOKEN": "hf_example", "HF_BASE_URL": HF_DEFAULT_BASE_URL}
        )


def test_hugging_face_client_from_env_requires_https() -> None:
    with pytest.raises(ModelConfigurationError, match="https"):
        HuggingFaceChatClient.from_env(
            {
                "HF_TOKEN": "hf_example",
                "MODEL_ID": "example/model",
                "HF_BASE_URL": "http://unsafe.example",
            }
        )


def test_hugging_face_client_from_env_accepts_safe_configuration() -> None:
    client = HuggingFaceChatClient.from_env(
        {
            "HF_TOKEN": "hf_example",
            "MODEL_ID": "example/model",
            "HF_BASE_URL": HF_DEFAULT_BASE_URL,
        }
    )
    assert client.model_id == "example/model"
    assert client.token == "hf_example"
    assert client.base_url == HF_DEFAULT_BASE_URL


def test_valid_fake_model_completes_full_five_task_mission() -> None:
    model = ValidFakeModel()
    result = _runtime(model).run(started_at=STARTED_AT)

    assert result.metrics.mission_success is True
    assert result.metrics.tasks_completed == 5
    assert result.metrics.total_messages == 50
    assert result.metrics.total_synthetic_execution_cost == 347.0
    assert result.metrics.allocation_efficiency == 1.0
    assert [item.metrics.winning_peer for item in result.task_results] == [
        "peer.atlas",
        "peer.ledger",
        "peer.sentinel",
        "peer.veritas",
        "peer.quill",
    ]
    assert [call[0] for call in model.calls] == [
        "market_research",
        "financial_analysis",
        "operating_risk",
        "evidence_verification",
        "executive_synthesis",
    ]


def test_downstream_llm_tasks_receive_validated_upstream_products() -> None:
    model = ValidFakeModel()
    result = _runtime(model).run(started_at=STARTED_AT)
    assert result.metrics.mission_success

    calls = {task_name: payload for task_name, payload in model.calls}
    verify_inputs = calls["evidence_verification"]["validated_upstream_work_products"]
    synthesis_inputs = calls["executive_synthesis"]["validated_upstream_work_products"]

    assert len(verify_inputs) == 3
    assert len(synthesis_inputs) == 4
    assert {item["capability"] for item in verify_inputs} == {
        "market_research",
        "financial_analysis",
        "operating_risk",
    }
    assert "evidence_verification" in {
        item["capability"] for item in synthesis_inputs
    }


def test_invented_evidence_fails_closed_and_exhausts_bounded_reauction() -> None:
    result = _runtime(InventedEvidenceModel()).run(started_at=STARTED_AT)

    first = result.task_results[0]
    assert first.status is TaskRunStatus.ESCALATED
    assert first.metrics.auction_rounds == 2
    assert first.metrics.reauction_count == 1
    assert first.metrics.execution_failures == 2
    assert [failure.failure_class.value for failure in first.failures] == [
        "invalid_output",
        "invalid_output",
    ]
    assert result.metrics.mission_success is False
    assert result.metrics.tasks_attempted == 1


def test_extra_control_plane_field_is_rejected_by_strict_model_schema() -> None:
    result = _runtime(ExtraControlFieldModel()).run(started_at=STARTED_AT)
    first = result.task_results[0]

    assert first.status is TaskRunStatus.ESCALATED
    assert all(failure.failure_class.value == "invalid_output" for failure in first.failures)
    assert "ModelWorkDraft" in first.failures[0].detail


def test_malformed_json_is_invalid_output_not_control_plane_failure() -> None:
    result = _runtime(MalformedJsonModel()).run(started_at=STARTED_AT)
    first = result.task_results[0]

    assert first.status is TaskRunStatus.ESCALATED
    assert first.failures[0].failure_class.value == "invalid_output"
    assert "not valid JSON" in first.failures[0].detail
