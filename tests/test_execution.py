"""Tests for deterministic Agent 11 task execution and work products."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from auction_coordination.execution import (
    DeterministicTaskExecutor,
    ExecutionError,
    HandlerOutput,
)
from auction_coordination.models import (
    FailureClass,
    TaskAnnouncement,
    TaskAward,
    WorkProduct,
)
from auction_coordination.reauction import BoundedReauctionPolicy, RecoveryAction
from auction_coordination.scenario import build_default_registry

NOW = datetime(2026, 9, 11, 22, 20, tzinfo=timezone.utc)

SPECIALISTS = {
    "market_research": "peer.atlas",
    "financial_analysis": "peer.ledger",
    "operating_risk": "peer.sentinel",
    "evidence_verification": "peer.veritas",
    "executive_synthesis": "peer.quill",
}


def announcement(
    capability: str,
    *,
    task_id: str | None = None,
    required_input_ids: tuple[str, ...] = (),
) -> TaskAnnouncement:
    suffix = capability.replace("_", ".")
    return TaskAnnouncement(
        task_id=task_id or f"task.{suffix}",
        auction_id=f"auction.{suffix}.1",
        auction_round=1,
        requested_capability=capability,
        summary=f"Execute synthetic {capability} diligence work.",
        minimum_capability_score=0.70,
        maximum_cost=100.0,
        required_input_ids=required_input_ids,
        max_auction_rounds=2,
        opened_at=NOW,
    )


def award_for(task: TaskAnnouncement, agent_id: str) -> TaskAward:
    return TaskAward(
        award_id=f"award:{task.task_id}:{agent_id}",
        auction_id=task.auction_id,
        task_id=task.task_id,
        auction_round=task.auction_round,
        winning_bid_id=f"bid:{task.task_id}:{agent_id}",
        awarded_agent_id=agent_id,
        winning_score=0.80,
        awarded_at=NOW,
    )


def upstream_product(product_id: str, title: str) -> WorkProduct:
    return WorkProduct(
        work_product_id=product_id,
        award_id=f"award.{product_id}",
        auction_id=f"auction.{product_id}",
        task_id=f"task.{product_id}",
        auction_round=1,
        agent_id="peer.mosaic",
        capability="market_research",
        title=title,
        summary="Synthetic upstream work product for deterministic execution testing.",
        findings=("Synthetic upstream finding.",),
        evidence_ids=(f"evidence.{product_id}",),
        completed_at=NOW,
    )


@pytest.mark.parametrize("capability,agent_id", SPECIALISTS.items())
def test_default_executor_supports_all_five_due_diligence_capabilities(
    capability: str,
    agent_id: str,
) -> None:
    registry = build_default_registry()
    task = announcement(capability)
    award = award_for(task, agent_id)

    product = DeterministicTaskExecutor(registry).execute(
        announcement=task,
        award=award,
        completed_at=NOW,
    )

    assert product.award_id == award.award_id
    assert product.auction_id == task.auction_id
    assert product.task_id == task.task_id
    assert product.auction_round == task.auction_round
    assert product.agent_id == agent_id
    assert product.capability == capability
    assert product.findings
    assert product.evidence_ids


def test_work_product_id_is_deterministic_for_same_award_and_task() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = award_for(task, "peer.atlas")
    executor = DeterministicTaskExecutor(registry)

    first = executor.execute(announcement=task, award=award, completed_at=NOW)
    second = executor.execute(announcement=task, award=award, completed_at=NOW)

    assert first.work_product_id == second.work_product_id
    assert first.work_product_id.startswith("work:")


def test_executor_rejects_award_lineage_mismatch() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = TaskAward(
        award_id="award:wrong-task",
        auction_id=task.auction_id,
        task_id="task.wrong",
        auction_round=task.auction_round,
        winning_bid_id="bid:wrong-task",
        awarded_agent_id="peer.atlas",
        winning_score=0.80,
        awarded_at=NOW,
    )

    with pytest.raises(ExecutionError) as exc_info:
        DeterministicTaskExecutor(registry).execute(
            announcement=task,
            award=award,
            completed_at=NOW,
        )

    assert exc_info.value.failure_class is FailureClass.POLICY_REJECTION
    assert "task_id" in str(exc_info.value)


def test_missing_required_input_is_dependency_failure() -> None:
    registry = build_default_registry()
    task = announcement(
        "executive_synthesis",
        required_input_ids=("work.market", "work.finance"),
    )
    award = award_for(task, "peer.quill")

    with pytest.raises(ExecutionError) as exc_info:
        DeterministicTaskExecutor(registry).execute(
            announcement=task,
            award=award,
            completed_at=NOW,
            available_inputs=(upstream_product("work.market", "Market Review"),),
        )

    assert exc_info.value.failure_class is FailureClass.DEPENDENCY_FAILURE
    assert "work.finance" in str(exc_info.value)


def test_executive_synthesis_consumes_required_inputs_in_declared_order() -> None:
    registry = build_default_registry()
    task = announcement(
        "executive_synthesis",
        required_input_ids=("work.market", "work.finance"),
    )
    award = award_for(task, "peer.quill")
    market = upstream_product("work.market", "Market Review")
    finance = upstream_product("work.finance", "Financial Review")

    product = DeterministicTaskExecutor(registry).execute(
        announcement=task,
        award=award,
        completed_at=NOW,
        available_inputs=(finance, market),
    )

    assert "Integrated 2 validated upstream work products." in product.summary
    assert "Market Review, Financial Review" in product.summary


def test_explicit_empty_handler_registry_fails_closed() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = award_for(task, "peer.atlas")

    with pytest.raises(ExecutionError) as exc_info:
        DeterministicTaskExecutor(registry, handlers={}).execute(
            announcement=task,
            award=award,
            completed_at=NOW,
        )

    assert exc_info.value.failure_class is FailureClass.POLICY_REJECTION
    assert "no approved handler" in str(exc_info.value)


def test_handler_exception_is_contained_as_execution_error() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = award_for(task, "peer.atlas")

    def exploding_handler(_context):
        raise RuntimeError("synthetic handler failure")

    executor = DeterministicTaskExecutor(
        registry,
        handlers={"market_research": exploding_handler},
    )

    with pytest.raises(ExecutionError) as exc_info:
        executor.execute(announcement=task, award=award, completed_at=NOW)

    assert exc_info.value.failure_class is FailureClass.EXECUTION_ERROR
    assert "RuntimeError" in str(exc_info.value)


def test_invalid_handler_content_is_rejected_before_publication() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = award_for(task, "peer.atlas")

    def malformed_handler(_context):
        return HandlerOutput(
            title="Malformed synthetic result",
            summary="This handler deliberately emits duplicate findings for validation testing.",
            findings=("duplicate finding", "duplicate finding"),
            evidence_ids=("evidence.bad.1",),
        )

    executor = DeterministicTaskExecutor(
        registry,
        handlers={"market_research": malformed_handler},
    )

    with pytest.raises(ExecutionError) as exc_info:
        executor.execute(announcement=task, award=award, completed_at=NOW)

    assert exc_info.value.failure_class is FailureClass.INVALID_OUTPUT
    assert "WorkProduct validation" in str(exc_info.value)


def test_execution_failure_can_feed_bounded_reauction_policy() -> None:
    registry = build_default_registry()
    task = announcement("market_research")
    award = award_for(task, "peer.atlas")

    def malformed_handler(_context):
        return HandlerOutput(
            title="Malformed synthetic result",
            summary="Invalid output is converted into a typed task failure.",
            findings=(),
            evidence_ids=("evidence.bad.1",),
        )

    executor = DeterministicTaskExecutor(
        registry,
        handlers={"market_research": malformed_handler},
    )

    with pytest.raises(ExecutionError) as exc_info:
        executor.execute(announcement=task, award=award, completed_at=NOW)

    recovery = BoundedReauctionPolicy()
    failure = recovery.record_failure(
        award=award,
        failure_class=exc_info.value.failure_class,
        detail=str(exc_info.value),
        failed_at=NOW,
    )
    plan = recovery.plan(
        announcement=task,
        award=award,
        failure=failure,
        reopened_at=NOW,
    )

    assert failure.failure_class is FailureClass.INVALID_OUTPUT
    assert plan.action is RecoveryAction.REAUCTION
    assert plan.excluded_bidder_ids == ("peer.atlas",)
    assert plan.next_announcement is not None
    assert plan.next_announcement.auction_round == 2


def test_work_product_contract_rejects_duplicate_evidence_and_naive_time() -> None:
    base = dict(
        work_product_id="work.contract.test",
        award_id="award.contract.test",
        auction_id="auction.contract.test",
        task_id="task.contract.test",
        auction_round=1,
        agent_id="peer.atlas",
        capability="market_research",
        title="Synthetic Contract Test",
        summary="Synthetic work product used to exercise strict output validation.",
        findings=("Unique finding.",),
        evidence_ids=("evidence.contract.1", "evidence.contract.1"),
        completed_at=NOW,
    )

    with pytest.raises(ValidationError, match="evidence_ids"):
        WorkProduct(**base)

    base["evidence_ids"] = ("evidence.contract.1",)
    base["completed_at"] = datetime(2026, 9, 11, 22, 20)
    with pytest.raises(ValidationError, match="timezone"):
        WorkProduct(**base)
