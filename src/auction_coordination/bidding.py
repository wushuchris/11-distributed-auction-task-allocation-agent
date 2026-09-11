"""Deterministic local BID / ABSTAIN policy for Agent 11 peers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping

from .models import AbstentionReason, Bid, BidAbstention, TaskAnnouncement
from .registry import PeerRegistry


class BidAction(str, Enum):
    """Bounded actions available to one peer during a task auction."""

    BID = "bid"
    ABSTAIN = "abstain"


class BiddingPolicyError(ValueError):
    """Raised when local bidding configuration is internally inconsistent."""


@dataclass(frozen=True)
class PeerBidSettings:
    """Task-local economic and availability inputs owned by one peer.

    Qualification is intentionally absent. Capability authority remains in the
    application-owned PeerRegistry rather than in peer bidding settings.
    """

    agent_id: str
    availability: float
    confidence_by_capability: Mapping[str, float]
    cost_by_capability: Mapping[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.availability <= 1.0:
            raise BiddingPolicyError("availability must be between 0.0 and 1.0")

        for capability, confidence in self.confidence_by_capability.items():
            if not capability:
                raise BiddingPolicyError("confidence capability names must not be empty")
            if not 0.0 <= confidence <= 1.0:
                raise BiddingPolicyError(
                    f"confidence for {capability} must be between 0.0 and 1.0"
                )

        for capability, cost in self.cost_by_capability.items():
            if not capability:
                raise BiddingPolicyError("cost capability names must not be empty")
            if cost <= 0.0:
                raise BiddingPolicyError(f"cost for {capability} must be positive")

    def confidence_for(self, capability: str) -> float | None:
        return self.confidence_by_capability.get(capability)

    def cost_for(self, capability: str) -> float | None:
        return self.cost_by_capability.get(capability)


@dataclass(frozen=True)
class BidDecision:
    """Auditable result of one peer evaluating one task announcement."""

    action: BidAction
    reason: str
    bid: Bid | None = None
    abstention: BidAbstention | None = None

    def __post_init__(self) -> None:
        has_bid = self.bid is not None
        has_abstention = self.abstention is not None
        if has_bid == has_abstention:
            raise BiddingPolicyError(
                "bid decision must contain exactly one of bid or abstention"
            )
        if self.action is BidAction.BID and not has_bid:
            raise BiddingPolicyError("BID action requires a bid")
        if self.action is BidAction.ABSTAIN and not has_abstention:
            raise BiddingPolicyError("ABSTAIN action requires an abstention")


class LocalBidPolicy:
    """Pure deterministic policy deciding whether one peer should compete.

    The policy does not compare peers, score bids, or select winners. It only
    determines whether the current peer should submit a valid bid or an
    explicit abstention for the announced task.
    """

    def __init__(
        self,
        registry: PeerRegistry,
        *,
        minimum_confidence: float = 0.60,
        minimum_availability: float = 0.10,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise BiddingPolicyError("minimum_confidence must be between 0.0 and 1.0")
        if not 0.0 <= minimum_availability <= 1.0:
            raise BiddingPolicyError("minimum_availability must be between 0.0 and 1.0")
        self._registry = registry
        self._minimum_confidence = minimum_confidence
        self._minimum_availability = minimum_availability

    def evaluate(
        self,
        *,
        announcement: TaskAnnouncement,
        settings: PeerBidSettings,
        submitted_at: datetime,
    ) -> BidDecision:
        """Return a deterministic BID or ABSTAIN decision for one peer."""

        profile = self._registry.require(settings.agent_id)
        capability = announcement.requested_capability
        capability_score = profile.score_for(capability)

        if capability_score is None:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.CAPABILITY_MISMATCH,
                detail=f"{profile.display_name} is not registered for {capability}.",
            )

        if capability_score < announcement.minimum_capability_score:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.CAPABILITY_MISMATCH,
                detail=(
                    f"Registered capability {capability_score:.2f} is below the task floor "
                    f"of {announcement.minimum_capability_score:.2f}."
                ),
            )

        if settings.availability < self._minimum_availability:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.UNAVAILABLE,
                detail=(
                    f"Availability {settings.availability:.2f} is below the local bidding "
                    f"threshold of {self._minimum_availability:.2f}."
                ),
            )

        confidence = settings.confidence_for(capability)
        estimated_cost = settings.cost_for(capability)
        if confidence is None or estimated_cost is None:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.LOCAL_POLICY,
                detail="Local bidding inputs are incomplete for the requested capability.",
            )

        if estimated_cost > announcement.maximum_cost:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.OVER_BUDGET,
                detail=(
                    f"Estimated cost {estimated_cost:.2f} exceeds the task maximum "
                    f"of {announcement.maximum_cost:.2f}."
                ),
            )

        if confidence < self._minimum_confidence:
            return self._abstain(
                announcement=announcement,
                settings=settings,
                submitted_at=submitted_at,
                reason=AbstentionReason.LOW_CONFIDENCE,
                detail=(
                    f"Confidence {confidence:.2f} is below the local bidding threshold "
                    f"of {self._minimum_confidence:.2f}."
                ),
            )

        bid = Bid(
            bid_id=self._bid_id(announcement, settings.agent_id),
            auction_id=announcement.auction_id,
            task_id=announcement.task_id,
            auction_round=announcement.auction_round,
            bidder_id=settings.agent_id,
            capability=capability,
            confidence=confidence,
            availability=settings.availability,
            estimated_cost=estimated_cost,
            submitted_at=submitted_at,
        )
        return BidDecision(
            action=BidAction.BID,
            reason="peer meets registered capability, availability, confidence, and cost gates",
            bid=bid,
        )

    def _abstain(
        self,
        *,
        announcement: TaskAnnouncement,
        settings: PeerBidSettings,
        submitted_at: datetime,
        reason: AbstentionReason,
        detail: str,
    ) -> BidDecision:
        abstention = BidAbstention(
            abstention_id=self._abstention_id(announcement, settings.agent_id),
            auction_id=announcement.auction_id,
            task_id=announcement.task_id,
            auction_round=announcement.auction_round,
            bidder_id=settings.agent_id,
            reason=reason,
            detail=detail,
            submitted_at=submitted_at,
        )
        return BidDecision(
            action=BidAction.ABSTAIN,
            reason=detail,
            abstention=abstention,
        )

    @staticmethod
    def _bid_id(announcement: TaskAnnouncement, agent_id: str) -> str:
        return f"bid:{announcement.auction_id}:{announcement.auction_round}:{agent_id}"

    @staticmethod
    def _abstention_id(announcement: TaskAnnouncement, agent_id: str) -> str:
        return f"abstain:{announcement.auction_id}:{announcement.auction_round}:{agent_id}"
