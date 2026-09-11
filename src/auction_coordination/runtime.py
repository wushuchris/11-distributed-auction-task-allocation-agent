"""Deterministic end-to-end mission runtime for Agent 11.

The runtime advances the synthetic diligence task graph and records protocol
traffic. It never chooses a worker. Peers decide whether to compete, the
auction protocol admits responses, and deterministic settlement selects each
winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean

from .auction import AuctionRoundProtocol
from .bidding import BidAction, LocalBidPolicy, PeerBidSettings
from .execution import DeterministicTaskExecutor, ExecutionError
from .models import FailureClass, TaskAnnouncement, TaskAward, TaskFailure, WorkProduct
from .reauction import BoundedReauctionPolicy, RecoveryAction
from .registry import PeerRegistry
from .scenario import TaskBlueprint
from .settlement import DeterministicSettlement, SettlementResult


class MissionEventType(str, Enum):
    """Protocol-level traffic recorded by the deterministic mission runtime."""

    TASK_ANNOUNCEMENT = "task_announcement"
    REAUCTION_ANNOUNCEMENT = "reauction_announcement"
    BID = "bid"
    BID_ABSTENTION = "bid_abstention"
    AUCTION_CLOSED = "auction_closed"
    TASK_AWARD = "task_award"
    TASK_RESULT = "task_result"
    TASK_FAILURE = "task_failure"
    ESCALATION = "escalation"


class TaskRunStatus(str, Enum):
    COMPLETED = "completed"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class MissionEvent:
    sequence: int
    event_type: MissionEventType
    task_id: str
    auction_round: int
    occurred_at: datetime
    actor_id: str | None
    accepted: bool | None
    detail: str


@dataclass(frozen=True)
class TaskMetrics:
    task_id: str
    capability: str
    status: TaskRunStatus
    eligible_peers: int
    bids: int
    abstentions: int
    rejected_responses: int
    auction_rounds: int
    reauction_count: int
    no_valid_bid_rounds: int
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
class TaskRunResult:
    blueprint: TaskBlueprint
    status: TaskRunStatus
    final_announcement: TaskAnnouncement
    final_award: TaskAward | None
    work_product: WorkProduct | None
    failures: tuple[TaskFailure, ...]
    metrics: TaskMetrics


@dataclass(frozen=True)
class MissionMetrics:
    mission_success: bool
    tasks_attempted: int
    tasks_completed: int
    tasks_reauctioned: int
    escalated_tasks: int
    failed_auctions: int
    execution_failures: int
    total_bids: int
    total_abstentions: int
    rejected_responses: int
    total_messages: int
    average_bids_per_task: float
    average_messages_per_task: float
    total_synthetic_execution_cost: float
    allocation_efficiency: float | None


@dataclass(frozen=True)
class MissionResult:
    task_results: tuple[TaskRunResult, ...]
    work_products: tuple[WorkProduct, ...]
    events: tuple[MissionEvent, ...]
    metrics: MissionMetrics


class DeterministicMissionRuntime:
    """Run a bounded five-task distributed-auction mission deterministically."""

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
        self._bid_policy = LocalBidPolicy(registry)
        self._settlement = DeterministicSettlement(registry)
        self._reauction = BoundedReauctionPolicy()
        self._validate_configuration()

    def run(self, *, started_at: datetime) -> MissionResult:
        """Execute the configured mission in deterministic dependency order."""

        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("started_at must include timezone information")

        sequence = 0
        clock_offset = 0
        events: list[MissionEvent] = []
        task_results: list[TaskRunResult] = []
        completed: dict[str, WorkProduct] = {}

        def tick() -> datetime:
            nonlocal clock_offset
            value = started_at + timedelta(seconds=clock_offset)
            clock_offset += 1
            return value

        def emit(
            event_type: MissionEventType,
            *,
            task_id: str,
            auction_round: int,
            detail: str,
            actor_id: str | None = None,
            accepted: bool | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            events.append(
                MissionEvent(
                    sequence=sequence,
                    event_type=event_type,
                    task_id=task_id,
                    auction_round=auction_round,
                    occurred_at=tick(),
                    actor_id=actor_id,
                    accepted=accepted,
                    detail=detail,
                )
            )

        for blueprint in self._task_blueprints:
            required_input_ids = tuple(
                completed[task_id].work_product_id
                for task_id in blueprint.dependency_task_ids
            )
            announcement = TaskAnnouncement(
                task_id=blueprint.task_id,
                auction_id=blueprint.auction_id,
                auction_round=1,
                requested_capability=blueprint.capability,
                summary=blueprint.summary,
                minimum_capability_score=blueprint.minimum_capability_score,
                maximum_cost=blueprint.maximum_cost,
                required_input_ids=required_input_ids,
                max_auction_rounds=blueprint.max_auction_rounds,
                opened_at=tick(),
            )

            result = self._run_task(
                blueprint=blueprint,
                initial_announcement=announcement,
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
        return MissionResult(
            task_results=tuple(task_results),
            work_products=work_products,
            events=tuple(events),
            metrics=metrics,
        )

    def _run_task(
        self,
        *,
        blueprint: TaskBlueprint,
        initial_announcement: TaskAnnouncement,
        available_inputs: tuple[WorkProduct, ...],
        tick,
        emit,
        task_event_start: int,
        events: list[MissionEvent],
    ) -> TaskRunResult:
        announcement = initial_announcement
        protocol = AuctionRoundProtocol(announcement, self._registry)
        failures: list[TaskFailure] = []
        bids = 0
        abstentions = 0
        rejected = 0
        rounds = 0
        reauctions = 0
        no_valid_bid_rounds = 0
        execution_failures = 0
        execution_cost = 0.0
        efficiencies: list[float] = []
        final_settlement: SettlementResult | None = None
        final_award: TaskAward | None = None

        while True:
            rounds += 1
            announce_type = (
                MissionEventType.TASK_ANNOUNCEMENT
                if announcement.auction_round == 1
                else MissionEventType.REAUCTION_ANNOUNCEMENT
            )
            emit(
                announce_type,
                task_id=announcement.task_id,
                auction_round=announcement.auction_round,
                detail=(
                    f"opened auction round {announcement.auction_round} for "
                    f"{announcement.requested_capability}"
                ),
            )

            for agent_id in sorted(self._bid_settings):
                settings = self._bid_settings[agent_id]
                decision = self._bid_policy.evaluate(
                    announcement=announcement,
                    settings=settings,
                    submitted_at=tick(),
                )
                if decision.action is BidAction.BID:
                    assert decision.bid is not None
                    receipt = protocol.submit_bid(decision.bid)
                    event_type = MissionEventType.BID
                else:
                    assert decision.abstention is not None
                    receipt = protocol.submit_abstention(decision.abstention)
                    event_type = MissionEventType.BID_ABSTENTION

                emit(
                    event_type,
                    task_id=announcement.task_id,
                    auction_round=announcement.auction_round,
                    actor_id=agent_id,
                    accepted=receipt.accepted,
                    detail=f"{receipt.code.value}: {receipt.detail}",
                )
                if receipt.accepted and event_type is MissionEventType.BID:
                    bids += 1
                elif receipt.accepted:
                    abstentions += 1
                else:
                    rejected += 1

            protocol.close()
            emit(
                MissionEventType.AUCTION_CLOSED,
                task_id=announcement.task_id,
                auction_round=announcement.auction_round,
                detail=(
                    f"closed with {len(protocol.accepted_bids())} admitted bids and "
                    f"{len(protocol.accepted_abstentions())} admitted abstentions"
                ),
            )

            if not protocol.accepted_bids():
                no_valid_bid_rounds += 1
                emit(
                    MissionEventType.ESCALATION,
                    task_id=announcement.task_id,
                    auction_round=announcement.auction_round,
                    detail="no valid bids remained after protocol admission",
                )
                return self._task_result(
                    blueprint=blueprint,
                    status=TaskRunStatus.ESCALATED,
                    announcement=announcement,
                    final_award=None,
                    work_product=None,
                    failures=tuple(failures),
                    bids=bids,
                    abstentions=abstentions,
                    rejected=rejected,
                    rounds=rounds,
                    reauctions=reauctions,
                    no_valid_bid_rounds=no_valid_bid_rounds,
                    execution_failures=execution_failures,
                    execution_cost=execution_cost,
                    efficiencies=efficiencies,
                    final_settlement=None,
                    message_count=len(events) - task_event_start,
                )

            settlement = self._settlement.settle(protocol=protocol, awarded_at=tick())
            final_settlement = settlement
            final_award = settlement.award
            winner = settlement.winning_scorecard
            execution_cost += winner.estimated_cost
            best_valid_utility = max(card.total_score for card in settlement.ranked_scorecards)
            efficiencies.append(
                1.0 if best_valid_utility == 0.0 else winner.total_score / best_valid_utility
            )
            emit(
                MissionEventType.TASK_AWARD,
                task_id=announcement.task_id,
                auction_round=announcement.auction_round,
                actor_id=settlement.award.awarded_agent_id,
                detail=(
                    f"awarded at score {settlement.award.winning_score:.4f} and synthetic "
                    f"cost {winner.estimated_cost:.2f}"
                ),
            )

            try:
                if (announcement.task_id, announcement.auction_round) in self._failure_injections:
                    raise ExecutionError(
                        "injected deterministic execution failure",
                        failure_class=FailureClass.EXECUTION_ERROR,
                    )
                product = self._executor.execute(
                    announcement=announcement,
                    award=settlement.award,
                    completed_at=tick(),
                    available_inputs=available_inputs,
                )
            except ExecutionError as exc:
                execution_failures += 1
                failure = self._reauction.record_failure(
                    award=settlement.award,
                    failure_class=exc.failure_class,
                    detail=str(exc),
                    failed_at=tick(),
                )
                failures.append(failure)
                emit(
                    MissionEventType.TASK_FAILURE,
                    task_id=announcement.task_id,
                    auction_round=announcement.auction_round,
                    actor_id=settlement.award.awarded_agent_id,
                    detail=f"{failure.failure_class.value}: {failure.detail}",
                )
                plan = self._reauction.plan(
                    announcement=announcement,
                    award=settlement.award,
                    failure=failure,
                    reopened_at=tick(),
                )
                if plan.action is RecoveryAction.ESCALATE:
                    emit(
                        MissionEventType.ESCALATION,
                        task_id=announcement.task_id,
                        auction_round=announcement.auction_round,
                        detail=plan.reason,
                    )
                    return self._task_result(
                        blueprint=blueprint,
                        status=TaskRunStatus.ESCALATED,
                        announcement=announcement,
                        final_award=final_award,
                        work_product=None,
                        failures=tuple(failures),
                        bids=bids,
                        abstentions=abstentions,
                        rejected=rejected,
                        rounds=rounds,
                        reauctions=reauctions,
                        no_valid_bid_rounds=no_valid_bid_rounds,
                        execution_failures=execution_failures,
                        execution_cost=execution_cost,
                        efficiencies=efficiencies,
                        final_settlement=final_settlement,
                        message_count=len(events) - task_event_start,
                    )

                reauctions += 1
                protocol = self._reauction.open_reauction_round(
                    plan=plan,
                    registry=self._registry,
                )
                assert plan.next_announcement is not None
                announcement = plan.next_announcement
                continue

            emit(
                MissionEventType.TASK_RESULT,
                task_id=announcement.task_id,
                auction_round=announcement.auction_round,
                actor_id=product.agent_id,
                detail=f"validated work product {product.work_product_id}",
            )
            return self._task_result(
                blueprint=blueprint,
                status=TaskRunStatus.COMPLETED,
                announcement=announcement,
                final_award=final_award,
                work_product=product,
                failures=tuple(failures),
                bids=bids,
                abstentions=abstentions,
                rejected=rejected,
                rounds=rounds,
                reauctions=reauctions,
                no_valid_bid_rounds=no_valid_bid_rounds,
                execution_failures=execution_failures,
                execution_cost=execution_cost,
                efficiencies=efficiencies,
                final_settlement=final_settlement,
                message_count=len(events) - task_event_start,
            )

    def _task_result(
        self,
        *,
        blueprint: TaskBlueprint,
        status: TaskRunStatus,
        announcement: TaskAnnouncement,
        final_award: TaskAward | None,
        work_product: WorkProduct | None,
        failures: tuple[TaskFailure, ...],
        bids: int,
        abstentions: int,
        rejected: int,
        rounds: int,
        reauctions: int,
        no_valid_bid_rounds: int,
        execution_failures: int,
        execution_cost: float,
        efficiencies: list[float],
        final_settlement: SettlementResult | None,
        message_count: int,
    ) -> TaskRunResult:
        winning_score = None
        second_place_score = None
        winning_margin = None
        winning_cost = None
        winning_peer = None
        if final_settlement is not None:
            winning_score = final_settlement.award.winning_score
            second_place_score = final_settlement.second_place_score
            winning_peer = final_settlement.award.awarded_agent_id
            winning_cost = final_settlement.winning_scorecard.estimated_cost
            if second_place_score is not None:
                winning_margin = winning_score - second_place_score

        metrics = TaskMetrics(
            task_id=blueprint.task_id,
            capability=blueprint.capability,
            status=status,
            eligible_peers=len(
                self._registry.eligible_profiles(
                    blueprint.capability,
                    blueprint.minimum_capability_score,
                )
            ),
            bids=bids,
            abstentions=abstentions,
            rejected_responses=rejected,
            auction_rounds=rounds,
            reauction_count=reauctions,
            no_valid_bid_rounds=no_valid_bid_rounds,
            execution_failures=execution_failures,
            messages=message_count,
            winning_peer=winning_peer,
            winning_score=winning_score,
            second_place_score=second_place_score,
            winning_margin=winning_margin,
            winning_estimated_cost=winning_cost,
            execution_cost=round(execution_cost, 12),
            allocation_efficiency=(
                round(mean(efficiencies), 12) if efficiencies else None
            ),
        )
        return TaskRunResult(
            blueprint=blueprint,
            status=status,
            final_announcement=announcement,
            final_award=final_award,
            work_product=work_product,
            failures=failures,
            metrics=metrics,
        )

    def _mission_metrics(self, results: tuple[TaskRunResult, ...]) -> MissionMetrics:
        attempted = len(results)
        completed = sum(result.status is TaskRunStatus.COMPLETED for result in results)
        efficiencies = [
            result.metrics.allocation_efficiency
            for result in results
            if result.metrics.allocation_efficiency is not None
        ]
        mission_success = attempted == len(self._task_blueprints) and completed == attempted
        return MissionMetrics(
            mission_success=mission_success,
            tasks_attempted=attempted,
            tasks_completed=completed,
            tasks_reauctioned=sum(result.metrics.reauction_count > 0 for result in results),
            escalated_tasks=sum(result.status is TaskRunStatus.ESCALATED for result in results),
            failed_auctions=sum(result.metrics.no_valid_bid_rounds for result in results),
            execution_failures=sum(result.metrics.execution_failures for result in results),
            total_bids=sum(result.metrics.bids for result in results),
            total_abstentions=sum(result.metrics.abstentions for result in results),
            rejected_responses=sum(result.metrics.rejected_responses for result in results),
            total_messages=sum(result.metrics.messages for result in results),
            average_bids_per_task=(
                round(sum(result.metrics.bids for result in results) / attempted, 12)
                if attempted
                else 0.0
            ),
            average_messages_per_task=(
                round(sum(result.metrics.messages for result in results) / attempted, 12)
                if attempted
                else 0.0
            ),
            total_synthetic_execution_cost=round(
                sum(result.metrics.execution_cost for result in results), 12
            ),
            allocation_efficiency=(round(mean(efficiencies), 12) if efficiencies else None),
        )

    def _validate_configuration(self) -> None:
        if not self._task_blueprints:
            raise ValueError("mission requires at least one task blueprint")
        task_ids = [blueprint.task_id for blueprint in self._task_blueprints]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task blueprint IDs must be unique")

        seen: set[str] = set()
        for blueprint in self._task_blueprints:
            missing = [
                dependency
                for dependency in blueprint.dependency_task_ids
                if dependency not in seen
            ]
            if missing:
                raise ValueError(
                    f"task {blueprint.task_id} depends on unavailable prior tasks: "
                    f"{', '.join(missing)}"
                )
            seen.add(blueprint.task_id)

        registered_ids = {profile.agent_id for profile in self._registry.all_profiles()}
        if set(self._bid_settings) != registered_ids:
            raise ValueError("bid settings must cover exactly the registered peer identities")
