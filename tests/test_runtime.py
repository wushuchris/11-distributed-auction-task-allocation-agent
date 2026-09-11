"""End-to-end tests for the deterministic Agent 11 mission runtime."""

from datetime import datetime, timezone

import pytest

from auction_coordination.runtime import (
    DeterministicMissionRuntime,
    MissionEventType,
    TaskRunStatus,
)
from auction_coordination.scenario import (
    TaskBlueprint,
    build_default_registry,
    default_bid_settings,
    default_task_blueprints,
)

START = datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc)


def build_runtime(
    *, failure_injections: frozenset[tuple[str, int]] = frozenset()
) -> DeterministicMissionRuntime:
    registry = build_default_registry()
    return DeterministicMissionRuntime(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
        failure_injections=failure_injections,
    )


def test_default_blueprint_has_five_dependency_ordered_tasks() -> None:
    blueprints = default_task_blueprints()

    assert [task.task_id for task in blueprints] == [
        "task.market",
        "task.finance",
        "task.risk",
        "task.verify",
        "task.synthesis",
    ]
    assert blueprints[3].dependency_task_ids == (
        "task.market",
        "task.finance",
        "task.risk",
    )
    assert blueprints[4].dependency_task_ids == (
        "task.market",
        "task.finance",
        "task.risk",
        "task.verify",
    )


def test_default_mission_completes_all_five_tasks_with_expected_winners() -> None:
    result = build_runtime().run(started_at=START)

    assert result.metrics.mission_success
    assert result.metrics.tasks_attempted == 5
    assert result.metrics.tasks_completed == 5
    assert [task.status for task in result.task_results] == [TaskRunStatus.COMPLETED] * 5
    assert [task.metrics.winning_peer for task in result.task_results] == [
        "peer.atlas",
        "peer.ledger",
        "peer.sentinel",
        "peer.veritas",
        "peer.quill",
    ]
    assert len(result.work_products) == 5


def test_default_mission_metrics_are_deterministic_and_auditable() -> None:
    result = build_runtime().run(started_at=START)
    metrics = result.metrics

    assert metrics.tasks_reauctioned == 0
    assert metrics.escalated_tasks == 0
    assert metrics.failed_auctions == 0
    assert metrics.execution_failures == 0
    assert metrics.total_bids == 18
    assert metrics.total_abstentions == 12
    assert metrics.rejected_responses == 0
    assert metrics.total_messages == 50
    assert metrics.average_bids_per_task == pytest.approx(3.6)
    assert metrics.average_messages_per_task == pytest.approx(10.0)
    assert metrics.total_synthetic_execution_cost == pytest.approx(347.0)
    assert metrics.allocation_efficiency == pytest.approx(1.0)
    assert all(task.metrics.allocation_efficiency == 1.0 for task in result.task_results)


def test_default_winning_margins_and_costs_match_settlement_evidence() -> None:
    result = build_runtime().run(started_at=START)
    by_task = {task.blueprint.task_id: task.metrics for task in result.task_results}

    assert by_task["task.market"].winning_score == pytest.approx(0.8385)
    assert by_task["task.market"].winning_margin == pytest.approx(0.081)
    assert by_task["task.market"].winning_estimated_cost == pytest.approx(62.0)

    assert by_task["task.finance"].winning_score == pytest.approx(0.8345)
    assert by_task["task.finance"].winning_margin == pytest.approx(0.1215)
    assert by_task["task.finance"].winning_estimated_cost == pytest.approx(70.0)

    assert by_task["task.risk"].winning_score == pytest.approx(0.8195)
    assert by_task["task.risk"].winning_margin == pytest.approx(0.08)
    assert by_task["task.risk"].winning_estimated_cost == pytest.approx(68.0)

    assert by_task["task.verify"].winning_score == pytest.approx(0.872)
    assert by_task["task.verify"].winning_margin == pytest.approx(0.1375)
    assert by_task["task.verify"].winning_estimated_cost == pytest.approx(72.0)

    assert by_task["task.synthesis"].winning_score == pytest.approx(0.852)
    assert by_task["task.synthesis"].winning_margin == pytest.approx(0.144)
    assert by_task["task.synthesis"].winning_estimated_cost == pytest.approx(75.0)


def test_downstream_announcements_reference_validated_upstream_products() -> None:
    result = build_runtime().run(started_at=START)
    products = {product.task_id: product for product in result.work_products}
    tasks = {task.blueprint.task_id: task for task in result.task_results}

    assert tasks["task.verify"].final_announcement.required_input_ids == (
        products["task.market"].work_product_id,
        products["task.finance"].work_product_id,
        products["task.risk"].work_product_id,
    )
    assert tasks["task.synthesis"].final_announcement.required_input_ids == (
        products["task.market"].work_product_id,
        products["task.finance"].work_product_id,
        products["task.risk"].work_product_id,
        products["task.verify"].work_product_id,
    )
    assert "Integrated 4 validated upstream work products." in products["task.synthesis"].summary


def test_event_log_is_append_only_ordered_protocol_traffic() -> None:
    result = build_runtime().run(started_at=START)

    assert [event.sequence for event in result.events] == list(range(1, 51))
    assert result.events[0].event_type is MissionEventType.TASK_ANNOUNCEMENT
    assert result.events[-1].event_type is MissionEventType.TASK_RESULT
    assert sum(event.event_type is MissionEventType.TASK_AWARD for event in result.events) == 5
    assert sum(event.event_type is MissionEventType.TASK_RESULT for event in result.events) == 5


def test_one_injected_failure_reauctions_market_task_and_then_recovers() -> None:
    result = build_runtime(
        failure_injections=frozenset({("task.market", 1)})
    ).run(started_at=START)
    market = result.task_results[0]

    assert result.metrics.mission_success
    assert result.metrics.tasks_reauctioned == 1
    assert result.metrics.execution_failures == 1
    assert result.metrics.total_messages == 60
    assert result.metrics.total_bids == 21
    assert result.metrics.total_abstentions == 14
    assert result.metrics.rejected_responses == 1
    assert result.metrics.total_synthetic_execution_cost == pytest.approx(413.0)
    assert result.metrics.allocation_efficiency == pytest.approx(1.0)

    assert market.status is TaskRunStatus.COMPLETED
    assert market.metrics.auction_rounds == 2
    assert market.metrics.reauction_count == 1
    assert market.metrics.execution_failures == 1
    assert market.metrics.rejected_responses == 1
    assert market.metrics.messages == 20
    assert market.metrics.bids == 7
    assert market.metrics.abstentions == 4
    assert market.metrics.execution_cost == pytest.approx(128.0)
    assert market.metrics.winning_peer == "peer.veritas"
    assert market.final_announcement.auction_round == 2
    assert market.final_award is not None
    assert market.final_award.awarded_agent_id == "peer.veritas"

    rejected_atlas = [
        event
        for event in result.events
        if event.task_id == "task.market"
        and event.auction_round == 2
        and event.actor_id == "peer.atlas"
        and event.event_type is MissionEventType.BID
    ]
    assert len(rejected_atlas) == 1
    assert rejected_atlas[0].accepted is False
    assert "excluded_bidder" in rejected_atlas[0].detail


def test_second_failure_escalates_and_stops_dependent_mission_progress() -> None:
    result = build_runtime(
        failure_injections=frozenset(
            {
                ("task.market", 1),
                ("task.market", 2),
            }
        )
    ).run(started_at=START)
    market = result.task_results[0]

    assert not result.metrics.mission_success
    assert result.metrics.tasks_attempted == 1
    assert result.metrics.tasks_completed == 0
    assert result.metrics.tasks_reauctioned == 1
    assert result.metrics.escalated_tasks == 1
    assert result.metrics.execution_failures == 2
    assert market.status is TaskRunStatus.ESCALATED
    assert market.metrics.auction_rounds == 2
    assert market.metrics.messages == 21
    assert market.metrics.execution_cost == pytest.approx(128.0)
    assert len(result.work_products) == 0
    assert result.events[-1].event_type is MissionEventType.ESCALATION


def test_runtime_rejects_dependency_graph_that_is_not_topologically_ordered() -> None:
    registry = build_default_registry()
    bad = (
        TaskBlueprint(
            task_id="task.child",
            auction_id="auction.child",
            capability="market_research",
            summary="Synthetic child task.",
            dependency_task_ids=("task.parent",),
        ),
        TaskBlueprint(
            task_id="task.parent",
            auction_id="auction.parent",
            capability="market_research",
            summary="Synthetic parent task.",
        ),
    )

    with pytest.raises(ValueError, match="depends on unavailable prior tasks"):
        DeterministicMissionRuntime(
            registry=registry,
            bid_settings=default_bid_settings(),
            task_blueprints=bad,
        )


def test_runtime_rejects_naive_start_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        build_runtime().run(started_at=datetime(2026, 9, 11, 23, 0))
