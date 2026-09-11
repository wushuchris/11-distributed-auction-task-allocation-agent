"""Centralized Agent-8-style allocation baseline for Agent 11 evaluation.

The baseline intentionally uses the same peer registry, local eligibility inputs,
45/25/20/10 utility function, tie-break order, task graph, executor, and bounded
two-attempt failure limit as the distributed auction runtime.

Only the coordination mechanism changes: one central allocator has global
visibility and directly assigns the highest-ranked valid worker. Peers do not
exchange BID or ABSTAIN messages and no auction-round protocol is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean

from .bidding import BidAction, LocalBidPolicy, PeerBidSettings
from .execution import DeterministicTaskExecutor, ExecutionError
from .models import FailureClass, TaskAnnouncement, TaskAward, TaskFailure, WorkProduct
from .registry import PeerRegistry
from .runtime import MissionResult, TaskRunStatus
from .scenario import TaskBlueprint
from .settlement import BidScorecard, DeterministicSettlement


class CentralEventType(str, Enum):
    """Protocol-relevant events emitted by the centralized baseline."""

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TASK_FAILURE = "task_failure"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class CentralEvent:
    sequence: int
    event_type: CentralEventType
    task_id: str
    attempt: int
    occurred_at: datetime
    actor_id: str | None
    detail: str


@dataclass(frozen=True)
class CentralTaskMetrics:
    task_id: str
    capability: str
    status: TaskRunStatus
    valid_candidates: int
    attempts: int
    reassignments: int
    execution_failures: int
    messages: int
    winning_peer: str | None
    winning_score: float | None
    second_place_score: float | None
    winning_margin: float | None
    winning_estimated_cost: float | None
    execution_cost: float
    allocation_efficiency: float | None


@dataclass(frozen=True)
class CentralTaskResult:
    blueprint: TaskBlueprint
    status: TaskRunStatus
    final_announcement: TaskAnnouncement
    final_award: TaskAward | None
    work_product: WorkProduct | None
    failures: tuple[TaskFailure, ...]
    metrics: CentralTaskMetrics


@dataclass(frozen=True)
class CentralMissionMetrics:
    mission_success: bool
    tasks_attempted: int
    tasks_completed: int
    tasks_reassigned: int
    escalated_tasks: int
    execution_failures: int
    total_messages: int
    average_messages_per_task: float
    total_synthetic_execution_cost: float
    allocation_efficiency: float | None


@dataclass(frozen=True)
class CentralMissionResult:
    task_results: tuple[CentralTaskResult, ...]
    work_products: tuple[WorkProduct, ...]
    events: tuple[CentralEvent, ...]
    metrics: CentralMissionMetrics


@dataclass(frozen=True)
class AllocationComparison:
    """Side-by-side clean-run comparison of distributed and centralized control."""

    distributed_success: bool
    centralized_success: bool
    distributed_messages: int
    centralized_messages: int
    message_delta: int
    message_multiplier: float | None
    distributed_execution_cost: float
    centralized_execution_cost: float
    execution_cost_delta: float
    distributed_allocation_efficiency: float | None
    centralized_allocation_efficiency: float | None
    same_completed_workers: bool


class CentralizedAllocationBaseline:
    """Directly assign the best valid worker using global state.

    Candidate eligibility is derived from the same LocalBidPolicy used by
    distributed peers, but those decisions remain internal to the central
    allocator and are never transmitted as market messages. Candidate utility
    is computed by the same DeterministicSettlement score function.
    """

    def __init__(
        self,
        *,
        registry: PeerRegistry,
        bid_settings: dict[str, PeerBidSettings],
        task_blueprints: tuple[TaskBlueprint, ...],
        executor: DeterministicTaskExecutor | None = None,
        failure_injections: frozenset[tuple[str, int]] = frozenset(),
    ) -> None:
        self._registry = registry
        self._bid_settings = dict(bid_settings)
        self._task_blueprints = task_blueprints
        self._executor = executor or DeterministicTaskExecutor(registry)
        self._failure_injections = failure_injections
        self._eligibility = LocalBidPolicy(registry)
        self._utility = DeterministicSettlement(registry)
        self._validate_configuration()

    def run(self, *, started_at: datetime) -> CentralMissionResult:
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must include timezone information")

        sequence = 0
        clock_offset = 0
        events: list[CentralEvent] = []
        task_results: list[CentralTaskResult] = []
        completed: dict[str, WorkProduct] = {}

        def tick() -> datetime:
            nonlocal clock_offset
            value = started_at + timedelta(seconds=clock_offset)
            clock_offset += 1
            return value

        def emit(
            event_type: CentralEventType,
            *,
            task_id: str,
            attempt: int,
            detail: str,
            actor_id: str | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            events.append(
                CentralEvent(
                    sequence=sequence,
                    event_type=event_type,
                    task_id=task_id,
                    attempt=attempt,
                    occurred_at=tick(),
                    actor_id=actor_id,
                    detail=detail,
                )
            )

        for blueprint in self._task_blueprints:
            required_input_ids = tuple(
                completed[task_id].work_product_id
                for task_id in blueprint.dependency_task_ids
            )
            result = self._run_task(
                blueprint=blueprint,
                required_input_ids=required_input_ids,
                available_inputs=tuple(completed.values()),
                tick=tick,
                emit=emit,
                task_event_start=len(events),
                events=events,
            )
            task_results.append(result)

            if result.work_product is not None:
                completed[blueprint.task_id] = result.work_product
            if result.status is TaskRunStatus.ESCALATED:
                break

        metrics = self._mission_metrics(tuple(task_results))
        work_products = tuple(
            completed[blueprint.task_id]
            for blueprint in self._task_blueprints
            if blueprint.task_id in completed
        )
        return CentralMissionResult(
            task_results=tuple(task_results),
            work_products=work_products,
            events=tuple(events),
            metrics=metrics,
        )

    def _run_task(
        self,
        *,
        blueprint: TaskBlueprint,
        required_input_ids: tuple[str, ...],
        available_inputs: tuple[WorkProduct, ...],
        tick,
        emit,
        task_event_start: int,
        events: list[CentralEvent],
    ) -> CentralTaskResult:
        excluded: set[str] = set()
        failures: list[TaskFailure] = []
        execution_cost = 0.0
        execution_failures = 0
        last_announcement: TaskAnnouncement | None = None
        last_ranked: tuple[BidScorecard, ...] = ()
        last_award: TaskAward | None = None

        for attempt in range(1, blueprint.max_auction_rounds + 1):
            announcement = TaskAnnouncement(
                task_id=blueprint.task_id,
                auction_id=blueprint.auction_id,
                auction_round=attempt,
                requested_capability=blueprint.capability,
                summary=blueprint.summary,
                minimum_capability_score=blueprint.minimum_capability_score,
                maximum_cost=blueprint.maximum_cost,
                required_input_ids=required_input_ids,
                max_auction_rounds=blueprint.max_auction_rounds,
                opened_at=tick(),
            )
            last_announcement = announcement
            ranked = self._rank_candidates(
                announcement=announcement,
                excluded_agent_ids=frozenset(excluded),
                evaluated_at=tick(),
            )
            last_ranked = ranked

            if not ranked:
                emit(
                    CentralEventType.ESCALATION,
                    task_id=blueprint.task_id,
                    attempt=attempt,
                    actor_id="central.manager",
                    detail="no valid centralized assignment candidate remains",
                )
                return self._task_result(
                    blueprint=blueprint,
                    status=TaskRunStatus.ESCALATED,
                    announcement=announcement,
                    award=None,
                    work_product=None,
                    failures=tuple(failures),
                    ranked=ranked,
                    execution_cost=execution_cost,
                    execution_failures=execution_failures,
                    messages=len(events) - task_event_start,
                )

            winner = ranked[0]
            award = TaskAward(
                award_id=(
                    f"award:central:{announcement.auction_id}:"
                    f"{attempt}:{winner.bidder_id}"
                ),
                auction_id=announcement.auction_id,
                task_id=announcement.task_id,
                auction_round=attempt,
                winning_bid_id=f"central:{announcement.task_id}:{attempt}:{winner.bidder_id}",
                awarded_agent_id=winner.bidder_id,
                winning_score=winner.total_score,
                awarded_at=tick(),
            )
            last_award = award
            emit(
                CentralEventType.TASK_ASSIGNMENT,
                task_id=blueprint.task_id,
                attempt=attempt,
                actor_id="central.manager",
                detail=(
                    f"directly assigned {winner.bidder_id} at utility "
                    f"{winner.total_score:.4f}"
                ),
            )
            execution_cost += winner.estimated_cost

            try:
                if (blueprint.task_id, attempt) in self._failure_injections:
                    raise ExecutionError(
                        "synthetic centralized baseline failure injection",
                        failure_class=FailureClass.EXECUTION_ERROR,
                    )

                product = self._executor.execute(
                    announcement=announcement,
                    award=award,
                    completed_at=tick(),
                    available_inputs=available_inputs,
                )
            except ExecutionError as exc:
                execution_failures += 1
                failure = self._task_failure(
                    award=award,
                    failure_class=exc.failure_class,
                    detail=str(exc),
                    failed_at=tick(),
                )
                failures.append(failure)
                excluded.add(award.awarded_agent_id)
                emit(
                    CentralEventType.TASK_FAILURE,
                    task_id=blueprint.task_id,
                    attempt=attempt,
                    actor_id=award.awarded_agent_id,
                    detail=f"{exc.failure_class.value}: {exc}",
                )

                if attempt >= blueprint.max_auction_rounds:
                    emit(
                        CentralEventType.ESCALATION,
                        task_id=blueprint.task_id,
                        attempt=attempt,
                        actor_id="central.manager",
                        detail="centralized reassignment attempts exhausted",
                    )
                    return self._task_result(
                        blueprint=blueprint,
                        status=TaskRunStatus.ESCALATED,
                        announcement=announcement,
                        award=award,
                        work_product=None,
                        failures=tuple(failures),
                        ranked=ranked,
                        execution_cost=execution_cost,
                        execution_failures=execution_failures,
                        messages=len(events) - task_event_start,
                    )
                continue

            emit(
                CentralEventType.TASK_RESULT,
                task_id=blueprint.task_id,
                attempt=attempt,
                actor_id=award.awarded_agent_id,
                detail=f"validated work product {product.work_product_id}",
            )
            return self._task_result(
                blueprint=blueprint,
                status=TaskRunStatus.COMPLETED,
                announcement=announcement,
                award=award,
                work_product=product,
                failures=tuple(failures),
                ranked=ranked,
                execution_cost=execution_cost,
                execution_failures=execution_failures,
                messages=len(events) - task_event_start,
            )

        assert last_announcement is not None
        return self._task_result(
            blueprint=blueprint,
            status=TaskRunStatus.ESCALATED,
            announcement=last_announcement,
            award=last_award,
            work_product=None,
            failures=tuple(failures),
            ranked=last_ranked,
            execution_cost=execution_cost,
            execution_failures=execution_failures,
            messages=len(events) - task_event_start,
        )

    def _rank_candidates(
        self,
        *,
        announcement: TaskAnnouncement,
        excluded_agent_ids: frozenset[str],
        evaluated_at: datetime,
    ) -> tuple[BidScorecard, ...]:
        scorecards: list[BidScorecard] = []
        for agent_id in sorted(self._bid_settings):
            if agent_id in excluded_agent_ids:
                continue
            decision = self._eligibility.evaluate(
                announcement=announcement,
                settings=self._bid_settings[agent_id],
                submitted_at=evaluated_at,
            )
            if decision.action is not BidAction.BID or decision.bid is None:
                continue
            scorecards.append(
                self._utility.score_bid(
                    bid=decision.bid,
                    announcement=announcement,
                )
            )
        return self._utility.rank_scorecards(tuple(scorecards))

    @staticmethod
    def _task_failure(
        *,
        award: TaskAward,
        failure_class: FailureClass,
        detail: str,
        failed_at: datetime,
    ) -> TaskFailure:
        return TaskFailure(
            failure_id=(
                f"failure:central:{award.auction_id}:"
                f"{award.auction_round}:{award.awarded_agent_id}"
            ),
            award_id=award.award_id,
            auction_id=award.auction_id,
            task_id=award.task_id,
            auction_round=award.auction_round,
            agent_id=award.awarded_agent_id,
            failure_class=failure_class,
            detail=detail,
            failed_at=failed_at,
        )

    def _task_result(
        self,
        *,
        blueprint: TaskBlueprint,
        status: TaskRunStatus,
        announcement: TaskAnnouncement,
        award: TaskAward | None,
        work_product: WorkProduct | None,
        failures: tuple[TaskFailure, ...],
        ranked: tuple[BidScorecard, ...],
        execution_cost: float,
        execution_failures: int,
        messages: int,
    ) -> CentralTaskResult:
        winner = ranked[0] if ranked and award is not None else None
        second = ranked[1].total_score if len(ranked) > 1 else None
        margin = (
            winner.total_score - second
            if winner is not None and second is not None
            else None
        )
        efficiency = 1.0 if winner is not None else None
        attempts = announcement.auction_round
        return CentralTaskResult(
            blueprint=blueprint,
            status=status,
            final_announcement=announcement,
            final_award=award,
            work_product=work_product,
            failures=failures,
            metrics=CentralTaskMetrics(
                task_id=blueprint.task_id,
                capability=blueprint.capability,
                status=status,
                valid_candidates=len(ranked),
                attempts=attempts,
                reassignments=max(0, attempts - 1),
                execution_failures=execution_failures,
                messages=messages,
                winning_peer=winner.bidder_id if winner is not None else None,
                winning_score=winner.total_score if winner is not None else None,
                second_place_score=second,
                winning_margin=margin,
                winning_estimated_cost=(
                    winner.estimated_cost if winner is not None else None
                ),
                execution_cost=round(execution_cost, 12),
                allocation_efficiency=efficiency,
            ),
        )

    @staticmethod
    def _mission_metrics(
        task_results: tuple[CentralTaskResult, ...],
    ) -> CentralMissionMetrics:
        completed = sum(
            result.status is TaskRunStatus.COMPLETED for result in task_results
        )
        attempted = len(task_results)
        efficiencies = [
            result.metrics.allocation_efficiency
            for result in task_results
            if result.metrics.allocation_efficiency is not None
        ]
        total_messages = sum(result.metrics.messages for result in task_results)
        return CentralMissionMetrics(
            mission_success=bool(task_results) and completed == attempted == 5,
            tasks_attempted=attempted,
            tasks_completed=completed,
            tasks_reassigned=sum(
                result.metrics.reassignments > 0 for result in task_results
            ),
            escalated_tasks=sum(
                result.status is TaskRunStatus.ESCALATED for result in task_results
            ),
            execution_failures=sum(
                result.metrics.execution_failures for result in task_results
            ),
            total_messages=total_messages,
            average_messages_per_task=(
                round(total_messages / attempted, 12) if attempted else 0.0
            ),
            total_synthetic_execution_cost=round(
                sum(result.metrics.execution_cost for result in task_results), 12
            ),
            allocation_efficiency=(
                round(mean(efficiencies), 12) if efficiencies else None
            ),
        )

    def _validate_configuration(self) -> None:
        if not self._task_blueprints:
            raise ValueError("task_blueprints must not be empty")
        task_ids = [blueprint.task_id for blueprint in self._task_blueprints]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task blueprint IDs must be unique")
        known_tasks: set[str] = set()
        for blueprint in self._task_blueprints:
            missing = set(blueprint.dependency_task_ids) - known_tasks
            if missing:
                raise ValueError(
                    f"task {blueprint.task_id} depends on unavailable prior tasks: "
                    f"{', '.join(sorted(missing))}"
                )
            known_tasks.add(blueprint.task_id)

        registered_ids = {profile.agent_id for profile in self._registry.all_profiles()}
        configured_ids = set(self._bid_settings)
        if registered_ids != configured_ids:
            raise ValueError(
                "bid_settings must contain exactly the registered peer IDs"
            )


def compare_allocation_models(
    *,
    distributed: MissionResult,
    centralized: CentralMissionResult,
) -> AllocationComparison:
    """Compare successful-worker quality, execution cost, and coordination overhead."""

    distributed_workers = tuple(
        result.metrics.winning_peer
        for result in distributed.task_results
        if result.status is TaskRunStatus.COMPLETED
    )
    centralized_workers = tuple(
        result.metrics.winning_peer
        for result in centralized.task_results
        if result.status is TaskRunStatus.COMPLETED
    )
    central_messages = centralized.metrics.total_messages
    distributed_messages = distributed.metrics.total_messages
    multiplier = (
        round(distributed_messages / central_messages, 12)
        if central_messages > 0
        else None
    )
    return AllocationComparison(
        distributed_success=distributed.metrics.mission_success,
        centralized_success=centralized.metrics.mission_success,
        distributed_messages=distributed_messages,
        centralized_messages=central_messages,
        message_delta=distributed_messages - central_messages,
        message_multiplier=multiplier,
        distributed_execution_cost=distributed.metrics.total_synthetic_execution_cost,
        centralized_execution_cost=centralized.metrics.total_synthetic_execution_cost,
        execution_cost_delta=round(
            distributed.metrics.total_synthetic_execution_cost
            - centralized.metrics.total_synthetic_execution_cost,
            12,
        ),
        distributed_allocation_efficiency=distributed.metrics.allocation_efficiency,
        centralized_allocation_efficiency=centralized.metrics.allocation_efficiency,
        same_completed_workers=distributed_workers == centralized_workers,
    )
