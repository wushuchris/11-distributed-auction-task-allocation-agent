"""Deterministic stress evaluation for Agent 11 allocation architectures.

The harness perturbs only shared mission inputs, then runs the existing
distributed-auction runtime and centralized baseline unchanged. This keeps the
comparison controlled: both architectures see the same registry, economics,
task constraints, and explicit execution failures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from statistics import mean

from .baseline import (
    AllocationComparison,
    CentralMissionResult,
    CentralizedAllocationBaseline,
    compare_allocation_models,
)
from .bidding import PeerBidSettings
from .models import CapabilityProfile, RegisteredCapability
from .registry import PeerRegistry
from .runtime import DeterministicMissionRuntime, MissionResult
from .scenario import (
    TaskBlueprint,
    default_bid_settings,
    default_peer_profiles,
    default_task_blueprints,
)


@dataclass(frozen=True)
class PeerStressOverride:
    """Patch task-local economics for one peer without changing qualification."""

    agent_id: str
    availability: float | None = None
    confidence_overrides: tuple[tuple[str, float], ...] = ()
    cost_overrides: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.availability is not None and not 0.0 <= self.availability <= 1.0:
            raise ValueError("stress availability must be between 0.0 and 1.0")
        if len({key for key, _ in self.confidence_overrides}) != len(
            self.confidence_overrides
        ):
            raise ValueError("confidence stress overrides must have unique capabilities")
        if len({key for key, _ in self.cost_overrides}) != len(self.cost_overrides):
            raise ValueError("cost stress overrides must have unique capabilities")
        for capability, value in self.confidence_overrides:
            if not capability or not 0.0 <= value <= 1.0:
                raise ValueError("stress confidence values must be within [0, 1]")
        for capability, value in self.cost_overrides:
            if not capability or value <= 0.0:
                raise ValueError("stress cost values must be positive")


@dataclass(frozen=True)
class CapabilityStressOverride:
    """Patch application-owned registered capability for one peer/capability."""

    agent_id: str
    capability: str
    score: float

    def __post_init__(self) -> None:
        if not self.agent_id or not self.capability:
            raise ValueError("capability stress override identifiers must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("capability stress score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class TaskStressOverride:
    """Patch task admission/economic constraints for one synthetic task."""

    task_id: str
    minimum_capability_score: float | None = None
    maximum_cost: float | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task stress override requires task_id")
        if (
            self.minimum_capability_score is not None
            and not 0.0 <= self.minimum_capability_score <= 1.0
        ):
            raise ValueError("minimum capability stress value must be within [0, 1]")
        if self.maximum_cost is not None and self.maximum_cost <= 0.0:
            raise ValueError("maximum cost stress value must be positive")


@dataclass(frozen=True)
class StressScenario:
    """One controlled perturbation applied equally to both allocation models."""

    name: str
    description: str
    peer_overrides: tuple[PeerStressOverride, ...] = ()
    capability_overrides: tuple[CapabilityStressOverride, ...] = ()
    task_overrides: tuple[TaskStressOverride, ...] = ()
    failure_injections: frozenset[tuple[str, int]] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ValueError("stress scenario requires name and description")
        peer_ids = [item.agent_id for item in self.peer_overrides]
        if len(peer_ids) != len(set(peer_ids)):
            raise ValueError("stress scenario cannot override the same peer twice")
        capability_keys = [
            (item.agent_id, item.capability) for item in self.capability_overrides
        ]
        if len(capability_keys) != len(set(capability_keys)):
            raise ValueError("stress scenario capability overrides must be unique")
        task_ids = [item.task_id for item in self.task_overrides]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("stress scenario cannot override the same task twice")
        for task_id, auction_round in self.failure_injections:
            if not task_id or auction_round < 1:
                raise ValueError("failure injection requires task_id and positive round")


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario: StressScenario
    distributed: MissionResult
    centralized: CentralMissionResult
    comparison: AllocationComparison


@dataclass(frozen=True)
class StressSuiteMetrics:
    scenario_count: int
    both_succeeded: int
    both_failed: int
    success_disagreements: int
    worker_disagreements: int
    distributed_total_messages: int
    centralized_total_messages: int
    message_multiplier: float | None
    distributed_total_execution_cost: float
    centralized_total_execution_cost: float
    execution_cost_delta: float
    distributed_mean_allocation_efficiency: float | None
    centralized_mean_allocation_efficiency: float | None


@dataclass(frozen=True)
class StressEvaluationReport:
    outcomes: tuple[ScenarioEvaluation, ...]
    metrics: StressSuiteMetrics


def default_stress_scenarios() -> tuple[StressScenario, ...]:
    """Return the public-safe deterministic Agent 11 evaluation matrix."""

    return (
        StressScenario(
            name="control",
            description="Unmodified five-task mission used as the clean control.",
        ),
        StressScenario(
            name="cost_pressure",
            description=(
                "Tighten the market-research budget to 65 so only lower-cost valid "
                "competitors remain."
            ),
            task_overrides=(
                TaskStressOverride(task_id="task.market", maximum_cost=65.0),
            ),
        ),
        StressScenario(
            name="capability_scarcity",
            description=(
                "Raise the market-research capability floor to 0.90 so a single "
                "registered specialist remains eligible."
            ),
            task_overrides=(
                TaskStressOverride(
                    task_id="task.market",
                    minimum_capability_score=0.90,
                ),
            ),
        ),
        StressScenario(
            name="low_confidence",
            description=(
                "Drop Atlas market confidence below the local bidding threshold to "
                "force allocation to another qualified peer."
            ),
            peer_overrides=(
                PeerStressOverride(
                    agent_id="peer.atlas",
                    confidence_overrides=(("market_research", 0.50),),
                ),
            ),
        ),
        StressScenario(
            name="availability_shock",
            description=(
                "Set Atlas availability to zero and require the market to allocate "
                "around a temporarily unavailable leading specialist."
            ),
            peer_overrides=(
                PeerStressOverride(agent_id="peer.atlas", availability=0.0),
            ),
        ),
        StressScenario(
            name="exact_tie",
            description=(
                "Create two exactly equal market candidates so the deterministic "
                "lexical agent-id tie-break is exercised."
            ),
            peer_overrides=(
                PeerStressOverride(
                    agent_id="peer.atlas",
                    availability=0.80,
                    confidence_overrides=(("market_research", 0.80),),
                    cost_overrides=(("market_research", 70.0),),
                ),
                PeerStressOverride(
                    agent_id="peer.veritas",
                    availability=0.80,
                    confidence_overrides=(("market_research", 0.80),),
                    cost_overrides=(("market_research", 70.0),),
                ),
            ),
            capability_overrides=(
                CapabilityStressOverride(
                    agent_id="peer.atlas",
                    capability="market_research",
                    score=0.90,
                ),
                CapabilityStressOverride(
                    agent_id="peer.veritas",
                    capability="market_research",
                    score=0.90,
                ),
            ),
            task_overrides=(
                TaskStressOverride(
                    task_id="task.market",
                    minimum_capability_score=0.90,
                ),
            ),
        ),
        StressScenario(
            name="no_valid_bid",
            description=(
                "Reduce the market-research budget below every qualified peer's "
                "declared cost so the task must escalate without an award."
            ),
            task_overrides=(
                TaskStressOverride(task_id="task.market", maximum_cost=50.0),
            ),
        ),
        StressScenario(
            name="single_failure_recovery",
            description=(
                "Inject an execution failure into the first market allocation and "
                "exercise bounded reauction/reassignment."
            ),
            failure_injections=frozenset({("task.market", 1)}),
        ),
        StressScenario(
            name="retry_exhaustion",
            description=(
                "Fail both allowed market attempts so each architecture must stop "
                "and escalate rather than continue dependent work."
            ),
            failure_injections=frozenset(
                {("task.market", 1), ("task.market", 2)}
            ),
        ),
    )


def run_stress_evaluation(
    *,
    started_at: datetime,
    scenarios: tuple[StressScenario, ...] | None = None,
) -> StressEvaluationReport:
    """Run each controlled scenario through both allocation architectures."""

    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must include timezone information")

    selected = default_stress_scenarios() if scenarios is None else scenarios
    if not selected:
        raise ValueError("stress evaluation requires at least one scenario")
    names = [scenario.name for scenario in selected]
    if len(names) != len(set(names)):
        raise ValueError("stress scenario names must be unique")

    outcomes: list[ScenarioEvaluation] = []
    for scenario in selected:
        registry, bid_settings, task_blueprints = _materialize_scenario(scenario)
        distributed = DeterministicMissionRuntime(
            registry=registry,
            bid_settings=bid_settings,
            task_blueprints=task_blueprints,
            failure_injections=scenario.failure_injections,
        ).run(started_at=started_at)
        centralized = CentralizedAllocationBaseline(
            registry=registry,
            bid_settings=bid_settings,
            task_blueprints=task_blueprints,
            failure_injections=scenario.failure_injections,
        ).run(started_at=started_at)
        outcomes.append(
            ScenarioEvaluation(
                scenario=scenario,
                distributed=distributed,
                centralized=centralized,
                comparison=compare_allocation_models(
                    distributed=distributed,
                    centralized=centralized,
                ),
            )
        )

    result_tuple = tuple(outcomes)
    return StressEvaluationReport(
        outcomes=result_tuple,
        metrics=_suite_metrics(result_tuple),
    )


def _materialize_scenario(
    scenario: StressScenario,
) -> tuple[PeerRegistry, dict[str, PeerBidSettings], tuple[TaskBlueprint, ...]]:
    profiles = _stressed_profiles(scenario)
    registry = PeerRegistry(profiles)
    bid_settings = _stressed_bid_settings(scenario)
    blueprints = _stressed_task_blueprints(scenario)
    known_task_ids = {blueprint.task_id for blueprint in blueprints}
    max_rounds = {
        blueprint.task_id: blueprint.max_auction_rounds for blueprint in blueprints
    }
    for task_id, auction_round in scenario.failure_injections:
        if task_id not in known_task_ids:
            raise ValueError(f"unknown failure-injection task: {task_id}")
        if auction_round > max_rounds[task_id]:
            raise ValueError(
                f"failure-injection round {auction_round} exceeds task limit for {task_id}"
            )
    return registry, bid_settings, blueprints


def _stressed_profiles(scenario: StressScenario) -> tuple[CapabilityProfile, ...]:
    overrides = {
        (item.agent_id, item.capability): item.score
        for item in scenario.capability_overrides
    }
    seen: set[tuple[str, str]] = set()
    profiles: list[CapabilityProfile] = []
    for profile in default_peer_profiles():
        capabilities: list[RegisteredCapability] = []
        for registered in profile.capabilities:
            key = (profile.agent_id, registered.capability)
            score = overrides.get(key, registered.score)
            if key in overrides:
                seen.add(key)
            capabilities.append(
                RegisteredCapability(capability=registered.capability, score=score)
            )
        profiles.append(
            CapabilityProfile(
                agent_id=profile.agent_id,
                display_name=profile.display_name,
                capabilities=tuple(capabilities),
            )
        )
    unknown = set(overrides) - seen
    if unknown:
        rendered = ", ".join(f"{agent}:{capability}" for agent, capability in sorted(unknown))
        raise ValueError(f"unknown capability stress override(s): {rendered}")
    return tuple(profiles)


def _stressed_bid_settings(scenario: StressScenario) -> dict[str, PeerBidSettings]:
    settings = default_bid_settings()
    overrides = {item.agent_id: item for item in scenario.peer_overrides}
    unknown_agents = set(overrides) - set(settings)
    if unknown_agents:
        raise ValueError(
            "unknown peer stress override(s): " + ", ".join(sorted(unknown_agents))
        )

    result: dict[str, PeerBidSettings] = {}
    for agent_id, base in settings.items():
        override = overrides.get(agent_id)
        if override is None:
            result[agent_id] = base
            continue

        confidence = dict(base.confidence_by_capability)
        cost = dict(base.cost_by_capability)
        for capability, value in override.confidence_overrides:
            if capability not in confidence:
                raise ValueError(
                    f"{agent_id} has no confidence input for capability {capability}"
                )
            confidence[capability] = value
        for capability, value in override.cost_overrides:
            if capability not in cost:
                raise ValueError(f"{agent_id} has no cost input for capability {capability}")
            cost[capability] = value

        result[agent_id] = PeerBidSettings(
            agent_id=agent_id,
            availability=(
                base.availability if override.availability is None else override.availability
            ),
            confidence_by_capability=confidence,
            cost_by_capability=cost,
        )
    return result


def _stressed_task_blueprints(scenario: StressScenario) -> tuple[TaskBlueprint, ...]:
    blueprints = default_task_blueprints()
    overrides = {item.task_id: item for item in scenario.task_overrides}
    known = {blueprint.task_id for blueprint in blueprints}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError("unknown task stress override(s): " + ", ".join(sorted(unknown)))

    result: list[TaskBlueprint] = []
    for blueprint in blueprints:
        override = overrides.get(blueprint.task_id)
        if override is None:
            result.append(blueprint)
            continue
        result.append(
            replace(
                blueprint,
                minimum_capability_score=(
                    blueprint.minimum_capability_score
                    if override.minimum_capability_score is None
                    else override.minimum_capability_score
                ),
                maximum_cost=(
                    blueprint.maximum_cost
                    if override.maximum_cost is None
                    else override.maximum_cost
                ),
            )
        )
    return tuple(result)


def _suite_metrics(
    outcomes: tuple[ScenarioEvaluation, ...],
) -> StressSuiteMetrics:
    distributed_messages = sum(
        outcome.distributed.metrics.total_messages for outcome in outcomes
    )
    centralized_messages = sum(
        outcome.centralized.metrics.total_messages for outcome in outcomes
    )
    distributed_cost = round(
        sum(
            outcome.distributed.metrics.total_synthetic_execution_cost
            for outcome in outcomes
        ),
        12,
    )
    centralized_cost = round(
        sum(
            outcome.centralized.metrics.total_synthetic_execution_cost
            for outcome in outcomes
        ),
        12,
    )
    distributed_efficiencies = [
        outcome.distributed.metrics.allocation_efficiency
        for outcome in outcomes
        if outcome.distributed.metrics.allocation_efficiency is not None
    ]
    centralized_efficiencies = [
        outcome.centralized.metrics.allocation_efficiency
        for outcome in outcomes
        if outcome.centralized.metrics.allocation_efficiency is not None
    ]

    both_succeeded = sum(
        outcome.distributed.metrics.mission_success
        and outcome.centralized.metrics.mission_success
        for outcome in outcomes
    )
    both_failed = sum(
        not outcome.distributed.metrics.mission_success
        and not outcome.centralized.metrics.mission_success
        for outcome in outcomes
    )
    success_disagreements = sum(
        outcome.distributed.metrics.mission_success
        != outcome.centralized.metrics.mission_success
        for outcome in outcomes
    )
    worker_disagreements = sum(
        not outcome.comparison.same_completed_workers for outcome in outcomes
    )

    return StressSuiteMetrics(
        scenario_count=len(outcomes),
        both_succeeded=both_succeeded,
        both_failed=both_failed,
        success_disagreements=success_disagreements,
        worker_disagreements=worker_disagreements,
        distributed_total_messages=distributed_messages,
        centralized_total_messages=centralized_messages,
        message_multiplier=(
            round(distributed_messages / centralized_messages, 12)
            if centralized_messages > 0
            else None
        ),
        distributed_total_execution_cost=distributed_cost,
        centralized_total_execution_cost=centralized_cost,
        execution_cost_delta=round(distributed_cost - centralized_cost, 12),
        distributed_mean_allocation_efficiency=(
            round(mean(distributed_efficiencies), 12)
            if distributed_efficiencies
            else None
        ),
        centralized_mean_allocation_efficiency=(
            round(mean(centralized_efficiencies), 12)
            if centralized_efficiencies
            else None
        ),
    )
