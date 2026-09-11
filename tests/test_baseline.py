"""Tests for the centralized Agent-8-style allocation baseline."""

from datetime import datetime, timezone

from auction_coordination.baseline import (
    CentralEventType,
    CentralizedAllocationBaseline,
    compare_allocation_models,
)
from auction_coordination.runtime import DeterministicMissionRuntime, TaskRunStatus
from auction_coordination.scenario import (
    build_default_registry,
    default_bid_settings,
    default_task_blueprints,
)

NOW = datetime(2026, 9, 11, 22, 40, tzinfo=timezone.utc)

EXPECTED_WORKERS = (
    "peer.atlas",
    "peer.ledger",
    "peer.sentinel",
    "peer.veritas",
    "peer.quill",
)


def centralized(*, failures=frozenset()):
    registry = build_default_registry()
    return CentralizedAllocationBaseline(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
        failure_injections=failures,
    )


def distributed(*, failures=frozenset()):
    registry = build_default_registry()
    return DeterministicMissionRuntime(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
        failure_injections=failures,
    )


def test_clean_centralized_baseline_completes_same_five_tasks() -> None:
    result = centralized().run(started_at=NOW)

    assert result.metrics.mission_success is True
    assert result.metrics.tasks_attempted == 5
    assert result.metrics.tasks_completed == 5
    assert result.metrics.tasks_reassigned == 0
    assert result.metrics.escalated_tasks == 0
    assert result.metrics.execution_failures == 0
    assert len(result.work_products) == 5

    workers = tuple(task.metrics.winning_peer for task in result.task_results)
    assert workers == EXPECTED_WORKERS
    assert all(task.status is TaskRunStatus.COMPLETED for task in result.task_results)


def test_clean_centralized_baseline_has_same_candidate_quality_and_cost() -> None:
    result = centralized().run(started_at=NOW)

    assert tuple(task.metrics.valid_candidates for task in result.task_results) == (4, 4, 4, 4, 2)
    assert tuple(task.metrics.winning_estimated_cost for task in result.task_results) == (
        62.0,
        70.0,
        68.0,
        72.0,
        75.0,
    )
    assert result.metrics.total_synthetic_execution_cost == 347.0
    assert result.metrics.allocation_efficiency == 1.0


def test_clean_centralized_baseline_uses_only_assignment_and_result_messages() -> None:
    result = centralized().run(started_at=NOW)

    assert result.metrics.total_messages == 10
    assert result.metrics.average_messages_per_task == 2.0
    assert tuple(event.event_type for event in result.events) == (
        CentralEventType.TASK_ASSIGNMENT,
        CentralEventType.TASK_RESULT,
        CentralEventType.TASK_ASSIGNMENT,
        CentralEventType.TASK_RESULT,
        CentralEventType.TASK_ASSIGNMENT,
        CentralEventType.TASK_RESULT,
        CentralEventType.TASK_ASSIGNMENT,
        CentralEventType.TASK_RESULT,
        CentralEventType.TASK_ASSIGNMENT,
        CentralEventType.TASK_RESULT,
    )


def test_clean_model_comparison_is_quality_equivalent_but_message_heavier_distributed() -> None:
    distributed_result = distributed().run(started_at=NOW)
    centralized_result = centralized().run(started_at=NOW)

    comparison = compare_allocation_models(
        distributed=distributed_result,
        centralized=centralized_result,
    )

    assert comparison.distributed_success is True
    assert comparison.centralized_success is True
    assert comparison.same_completed_workers is True
    assert comparison.distributed_messages == 50
    assert comparison.centralized_messages == 10
    assert comparison.message_delta == 40
    assert comparison.message_multiplier == 5.0
    assert comparison.distributed_execution_cost == 347.0
    assert comparison.centralized_execution_cost == 347.0
    assert comparison.execution_cost_delta == 0.0
    assert comparison.distributed_allocation_efficiency == 1.0
    assert comparison.centralized_allocation_efficiency == 1.0


def test_centralized_failure_reassigns_to_next_best_worker_with_same_bound() -> None:
    result = centralized(failures=frozenset({("task.market", 1)})).run(started_at=NOW)

    market = result.task_results[0]
    assert result.metrics.mission_success is True
    assert result.metrics.tasks_reassigned == 1
    assert result.metrics.execution_failures == 1
    assert result.metrics.total_messages == 12
    assert result.metrics.total_synthetic_execution_cost == 413.0
    assert market.metrics.attempts == 2
    assert market.metrics.reassignments == 1
    assert market.metrics.execution_failures == 1
    assert market.metrics.winning_peer == "peer.veritas"
    assert market.metrics.execution_cost == 128.0
    assert market.failures[0].agent_id == "peer.atlas"


def test_failure_comparison_preserves_outcome_but_distributed_uses_more_messages() -> None:
    failures = frozenset({("task.market", 1)})
    distributed_result = distributed(failures=failures).run(started_at=NOW)
    centralized_result = centralized(failures=failures).run(started_at=NOW)

    comparison = compare_allocation_models(
        distributed=distributed_result,
        centralized=centralized_result,
    )

    assert comparison.same_completed_workers is True
    assert comparison.distributed_success is True
    assert comparison.centralized_success is True
    assert comparison.distributed_messages == 60
    assert comparison.centralized_messages == 12
    assert comparison.message_delta == 48
    assert comparison.message_multiplier == 5.0
    assert comparison.distributed_execution_cost == 413.0
    assert comparison.centralized_execution_cost == 413.0
    assert comparison.execution_cost_delta == 0.0


def test_centralized_second_failure_escalates_and_stops_dependencies() -> None:
    failures = frozenset({("task.market", 1), ("task.market", 2)})
    result = centralized(failures=failures).run(started_at=NOW)

    assert result.metrics.mission_success is False
    assert result.metrics.tasks_attempted == 1
    assert result.metrics.tasks_completed == 0
    assert result.metrics.escalated_tasks == 1
    assert result.metrics.execution_failures == 2
    assert result.metrics.total_messages == 5
    assert result.metrics.total_synthetic_execution_cost == 128.0
    assert result.task_results[0].status is TaskRunStatus.ESCALATED
    assert result.events[-1].event_type is CentralEventType.ESCALATION
