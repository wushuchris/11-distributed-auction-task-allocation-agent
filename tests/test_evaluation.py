"""Regression tests for the Agent 11 deterministic stress evaluation harness."""

from datetime import datetime, timezone

import pytest

from auction_coordination.evaluation import (
    CapabilityStressOverride,
    PeerStressOverride,
    StressScenario,
    TaskStressOverride,
    default_stress_scenarios,
    run_stress_evaluation,
)

NOW = datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc)


def _report():
    return run_stress_evaluation(started_at=NOW)


def _outcome(report, name: str):
    return next(outcome for outcome in report.outcomes if outcome.scenario.name == name)


def _task(result, task_id: str):
    return next(item for item in result.task_results if item.blueprint.task_id == task_id)


def test_default_stress_matrix_contains_nine_distinct_controlled_scenarios() -> None:
    scenarios = default_stress_scenarios()

    assert len(scenarios) == 9
    assert [scenario.name for scenario in scenarios] == [
        "control",
        "cost_pressure",
        "capability_scarcity",
        "low_confidence",
        "availability_shock",
        "exact_tie",
        "no_valid_bid",
        "single_failure_recovery",
        "retry_exhaustion",
    ]


def test_clean_control_preserves_known_distributed_and_centralized_baselines() -> None:
    report = _report()
    outcome = _outcome(report, "control")

    assert outcome.distributed.metrics.mission_success is True
    assert outcome.centralized.metrics.mission_success is True
    assert outcome.distributed.metrics.total_messages == 50
    assert outcome.centralized.metrics.total_messages == 10
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 347.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 347.0
    assert outcome.comparison.message_multiplier == 5.0
    assert outcome.comparison.same_completed_workers is True


def test_cost_pressure_reduces_market_competition_without_changing_winner() -> None:
    report = _report()
    outcome = _outcome(report, "cost_pressure")
    market = _task(outcome.distributed, "task.market")

    assert outcome.distributed.metrics.mission_success is True
    assert outcome.centralized.metrics.mission_success is True
    assert market.metrics.bids == 2
    assert market.metrics.abstentions == 4
    assert market.metrics.winning_peer == "peer.atlas"
    assert market.metrics.winning_estimated_cost == 62.0
    assert outcome.comparison.same_completed_workers is True


def test_capability_scarcity_leaves_one_market_specialist() -> None:
    report = _report()
    outcome = _outcome(report, "capability_scarcity")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert distributed_market.metrics.eligible_peers == 1
    assert distributed_market.metrics.bids == 1
    assert distributed_market.metrics.winning_peer == "peer.atlas"
    assert centralized_market.metrics.valid_candidates == 1
    assert centralized_market.metrics.winning_peer == "peer.atlas"


def test_low_confidence_forces_atlas_to_abstain_and_veritas_wins_market() -> None:
    report = _report()
    outcome = _outcome(report, "low_confidence")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert distributed_market.metrics.bids == 3
    assert distributed_market.metrics.abstentions == 3
    assert distributed_market.metrics.winning_peer == "peer.veritas"
    assert centralized_market.metrics.winning_peer == "peer.veritas"
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 351.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 351.0


def test_availability_shock_reallocates_market_away_from_atlas() -> None:
    report = _report()
    outcome = _outcome(report, "availability_shock")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert distributed_market.metrics.winning_peer == "peer.veritas"
    assert centralized_market.metrics.winning_peer == "peer.veritas"
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 351.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 351.0


def test_exact_tie_uses_lexical_agent_id_after_equal_score_capability_and_cost() -> None:
    report = _report()
    outcome = _outcome(report, "exact_tie")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert distributed_market.metrics.winning_peer == "peer.atlas"
    assert distributed_market.metrics.winning_score == 0.795
    assert distributed_market.metrics.second_place_score == 0.795
    assert distributed_market.metrics.winning_margin == 0.0
    assert centralized_market.metrics.winning_peer == "peer.atlas"
    assert centralized_market.metrics.winning_margin == 0.0
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 355.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 355.0


def test_no_valid_bid_escalates_immediately_in_both_architectures() -> None:
    report = _report()
    outcome = _outcome(report, "no_valid_bid")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert outcome.distributed.metrics.mission_success is False
    assert outcome.centralized.metrics.mission_success is False
    assert outcome.distributed.metrics.tasks_attempted == 1
    assert outcome.centralized.metrics.tasks_attempted == 1
    assert distributed_market.metrics.no_valid_bid_rounds == 1
    assert distributed_market.metrics.bids == 0
    assert distributed_market.metrics.abstentions == 6
    assert centralized_market.metrics.valid_candidates == 0
    assert outcome.distributed.metrics.total_messages == 9
    assert outcome.centralized.metrics.total_messages == 1
    assert outcome.comparison.message_multiplier == 9.0


def test_single_failure_recovery_matches_prior_reauction_baseline() -> None:
    report = _report()
    outcome = _outcome(report, "single_failure_recovery")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert outcome.distributed.metrics.mission_success is True
    assert outcome.centralized.metrics.mission_success is True
    assert distributed_market.metrics.reauction_count == 1
    assert distributed_market.metrics.rejected_responses == 1
    assert distributed_market.metrics.winning_peer == "peer.veritas"
    assert centralized_market.metrics.reassignments == 1
    assert centralized_market.metrics.winning_peer == "peer.veritas"
    assert outcome.distributed.metrics.total_messages == 60
    assert outcome.centralized.metrics.total_messages == 12
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 413.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 413.0


def test_retry_exhaustion_stops_after_two_failed_attempts() -> None:
    report = _report()
    outcome = _outcome(report, "retry_exhaustion")
    distributed_market = _task(outcome.distributed, "task.market")
    centralized_market = _task(outcome.centralized, "task.market")

    assert outcome.distributed.metrics.mission_success is False
    assert outcome.centralized.metrics.mission_success is False
    assert outcome.distributed.metrics.tasks_attempted == 1
    assert outcome.centralized.metrics.tasks_attempted == 1
    assert distributed_market.metrics.auction_rounds == 2
    assert distributed_market.metrics.execution_failures == 2
    assert centralized_market.metrics.attempts == 2
    assert centralized_market.metrics.execution_failures == 2
    assert outcome.distributed.metrics.total_messages == 21
    assert outcome.centralized.metrics.total_messages == 5
    assert outcome.distributed.metrics.total_synthetic_execution_cost == 128.0
    assert outcome.centralized.metrics.total_synthetic_execution_cost == 128.0


def test_default_suite_aggregates_controlled_architecture_tradeoffs() -> None:
    report = _report()
    metrics = report.metrics

    assert metrics.scenario_count == 9
    assert metrics.both_succeeded == 7
    assert metrics.both_failed == 2
    assert metrics.success_disagreements == 0
    assert metrics.worker_disagreements == 0
    assert metrics.distributed_total_messages == 390
    assert metrics.centralized_total_messages == 78
    assert metrics.message_multiplier == 5.0
    assert metrics.distributed_total_execution_cost == 2639.0
    assert metrics.centralized_total_execution_cost == 2639.0
    assert metrics.execution_cost_delta == 0.0
    assert metrics.distributed_mean_allocation_efficiency == 1.0
    assert metrics.centralized_mean_allocation_efficiency == 1.0


def test_custom_stress_scenario_is_applied_equally_to_both_models() -> None:
    scenario = StressScenario(
        name="custom_market_constraint",
        description="Custom regression scenario.",
        peer_overrides=(
            PeerStressOverride(
                agent_id="peer.atlas",
                confidence_overrides=(("market_research", 0.50),),
            ),
        ),
        capability_overrides=(
            CapabilityStressOverride(
                agent_id="peer.veritas",
                capability="market_research",
                score=0.85,
            ),
        ),
        task_overrides=(
            TaskStressOverride(task_id="task.market", maximum_cost=80.0),
        ),
    )

    report = run_stress_evaluation(started_at=NOW, scenarios=(scenario,))
    outcome = report.outcomes[0]

    assert outcome.distributed.metrics.mission_success is True
    assert outcome.centralized.metrics.mission_success is True
    assert _task(outcome.distributed, "task.market").metrics.winning_peer == "peer.veritas"
    assert _task(outcome.centralized, "task.market").metrics.winning_peer == "peer.veritas"


def test_stress_evaluation_rejects_unknown_override_targets() -> None:
    scenario = StressScenario(
        name="bad_override",
        description="Invalid target should fail closed.",
        task_overrides=(TaskStressOverride(task_id="task.unknown", maximum_cost=50.0),),
    )

    with pytest.raises(ValueError, match="unknown task stress override"):
        run_stress_evaluation(started_at=NOW, scenarios=(scenario,))
