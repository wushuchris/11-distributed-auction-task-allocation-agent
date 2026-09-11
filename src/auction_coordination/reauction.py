"""Bounded task-failure recovery and reauction policy for Agent 11.

This module handles one narrow recovery case: an awarded worker explicitly
fails a task. The failed worker is excluded from the immediate retry, the
auction round increments once when allowed, and the task escalates when the
configured round limit has been exhausted.

It deliberately does not implement reputation, redundancy, compromised-agent
handling, broad replanning, or organizational self-healing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .auction import AuctionRoundProtocol
from .models import FailureClass, TaskAnnouncement, TaskAward, TaskFailure
from .registry import PeerRegistry


class ReauctionError(ValueError):
    """Raised when failure recovery inputs are inconsistent or unsafe."""


class RecoveryAction(str, Enum):
    """Bounded actions available after an awarded task fails."""

    REAUCTION = "reauction"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class ReauctionPlan:
    """Auditable recovery decision for one failed task award."""

    action: RecoveryAction
    failure: TaskFailure
    previous_award: TaskAward
    excluded_bidder_ids: tuple[str, ...]
    reason: str
    next_announcement: TaskAnnouncement | None = None

    def __post_init__(self) -> None:
        if self.action is RecoveryAction.REAUCTION and self.next_announcement is None:
            raise ReauctionError("REAUCTION action requires next_announcement")
        if self.action is RecoveryAction.ESCALATE and self.next_announcement is not None:
            raise ReauctionError("ESCALATE action cannot contain next_announcement")


class BoundedReauctionPolicy:
    """Create at most the next bounded auction round after explicit failure."""

    def record_failure(
        self,
        *,
        award: TaskAward,
        failure_class: FailureClass,
        detail: str,
        failed_at: datetime,
    ) -> TaskFailure:
        """Create a deterministic typed failure record for the awarded worker."""

        return TaskFailure(
            failure_id=(
                f"failure:{award.auction_id}:"
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

    def plan(
        self,
        *,
        announcement: TaskAnnouncement,
        award: TaskAward,
        failure: TaskFailure,
        reopened_at: datetime,
    ) -> ReauctionPlan:
        """Return either one fresh auction round or a terminal escalation."""

        self._validate_lineage(
            announcement=announcement,
            award=award,
            failure=failure,
        )

        excluded = (award.awarded_agent_id,)
        if announcement.auction_round >= announcement.max_auction_rounds:
            return ReauctionPlan(
                action=RecoveryAction.ESCALATE,
                failure=failure,
                previous_award=award,
                excluded_bidder_ids=excluded,
                reason=(
                    f"auction round {announcement.auction_round} exhausted the configured "
                    f"maximum of {announcement.max_auction_rounds}; human escalation required"
                ),
            )

        next_announcement = TaskAnnouncement(
            task_id=announcement.task_id,
            auction_id=announcement.auction_id,
            auction_round=announcement.auction_round + 1,
            requested_capability=announcement.requested_capability,
            summary=announcement.summary,
            minimum_capability_score=announcement.minimum_capability_score,
            maximum_cost=announcement.maximum_cost,
            required_input_ids=announcement.required_input_ids,
            max_auction_rounds=announcement.max_auction_rounds,
            opened_at=reopened_at,
        )
        return ReauctionPlan(
            action=RecoveryAction.REAUCTION,
            failure=failure,
            previous_award=award,
            excluded_bidder_ids=excluded,
            reason=(
                f"explicit failure from {award.awarded_agent_id}; opening bounded round "
                f"{next_announcement.auction_round} with immediate failed-winner exclusion"
            ),
            next_announcement=next_announcement,
        )

    def open_reauction_round(
        self,
        *,
        plan: ReauctionPlan,
        registry: PeerRegistry,
    ) -> AuctionRoundProtocol:
        """Open the planned retry with protocol-enforced bidder exclusions."""

        if plan.action is not RecoveryAction.REAUCTION:
            raise ReauctionError("cannot open an auction round from an escalation plan")
        if plan.next_announcement is None:
            raise ReauctionError("reauction plan is missing next_announcement")
        return AuctionRoundProtocol(
            plan.next_announcement,
            registry,
            excluded_bidder_ids=plan.excluded_bidder_ids,
        )

    @staticmethod
    def _validate_lineage(
        *,
        announcement: TaskAnnouncement,
        award: TaskAward,
        failure: TaskFailure,
    ) -> None:
        if award.auction_id != announcement.auction_id:
            raise ReauctionError("award auction_id does not match announcement")
        if award.task_id != announcement.task_id:
            raise ReauctionError("award task_id does not match announcement")
        if award.auction_round != announcement.auction_round:
            raise ReauctionError("award auction_round does not match announcement")

        if failure.award_id != award.award_id:
            raise ReauctionError("failure award_id does not match the failed award")
        if failure.auction_id != award.auction_id:
            raise ReauctionError("failure auction_id does not match the failed award")
        if failure.task_id != award.task_id:
            raise ReauctionError("failure task_id does not match the failed award")
        if failure.auction_round != award.auction_round:
            raise ReauctionError("failure auction_round does not match the failed award")
        if failure.agent_id != award.awarded_agent_id:
            raise ReauctionError("failure agent_id does not match the awarded worker")
