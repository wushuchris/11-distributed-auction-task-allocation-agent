"""Deterministic scoring and settlement for admitted Agent 11 bids.

Settlement operates only on bids already admitted by AuctionRoundProtocol. The
scoring policy is application-owned and contains no LLM or bidder-controlled
capability authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose

from .auction import AuctionRoundProtocol
from .models import Bid, TaskAnnouncement, TaskAward
from .registry import PeerRegistry


class SettlementError(ValueError):
    """Raised when a round cannot be settled safely and deterministically."""


@dataclass(frozen=True)
class ScoringWeights:
    """Application-owned utility weights for deterministic bid scoring."""

    capability: float = 0.45
    confidence: float = 0.25
    availability: float = 0.20
    cost_efficiency: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.capability,
            self.confidence,
            self.availability,
            self.cost_efficiency,
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise SettlementError("scoring weights must each be between 0.0 and 1.0")
        if not isclose(sum(values), 1.0, abs_tol=1e-12):
            raise SettlementError("scoring weights must sum to 1.0")


DEFAULT_SCORING_WEIGHTS = ScoringWeights()


@dataclass(frozen=True)
class BidScorecard:
    """Auditable utility decomposition for one admitted bid."""

    bid_id: str
    bidder_id: str
    capability_score: float
    confidence: float
    availability: float
    cost_efficiency: float
    estimated_cost: float
    total_score: float


@dataclass(frozen=True)
class SettlementResult:
    """Deterministic award plus full ordered scorecard evidence."""

    award: TaskAward
    ranked_scorecards: tuple[BidScorecard, ...]

    @property
    def winning_scorecard(self) -> BidScorecard:
        return self.ranked_scorecards[0]

    @property
    def second_place_score(self) -> float | None:
        if len(self.ranked_scorecards) < 2:
            return None
        return self.ranked_scorecards[1].total_score


class DeterministicSettlement:
    """Score admitted bids and deterministically select one winner.

    Tie-breaking order is intentionally explicit:
    1. higher total auction score
    2. higher registered capability score
    3. lower estimated cost
    4. lexical bidder_id
    """

    def __init__(
        self,
        registry: PeerRegistry,
        *,
        weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS,
    ) -> None:
        self._registry = registry
        self._weights = weights

    @property
    def weights(self) -> ScoringWeights:
        return self._weights

    def score_bid(
        self,
        *,
        bid: Bid,
        announcement: TaskAnnouncement,
    ) -> BidScorecard:
        """Return the deterministic scorecard for one already-admitted bid."""

        if bid.auction_id != announcement.auction_id:
            raise SettlementError("bid auction_id does not match announcement")
        if bid.task_id != announcement.task_id:
            raise SettlementError("bid task_id does not match announcement")
        if bid.auction_round != announcement.auction_round:
            raise SettlementError("bid auction_round does not match announcement")
        if bid.capability != announcement.requested_capability:
            raise SettlementError("bid capability does not match requested capability")

        capability_score = self._registry.capability_score(
            bid.bidder_id,
            announcement.requested_capability,
        )
        if capability_score is None:
            raise SettlementError(
                f"{bid.bidder_id} is not registered for {announcement.requested_capability}"
            )
        if capability_score < announcement.minimum_capability_score:
            raise SettlementError("bidder capability is below the task minimum")
        if bid.availability <= 0.0:
            raise SettlementError("bidder has no positive availability")
        if bid.estimated_cost > announcement.maximum_cost:
            raise SettlementError("bid cost exceeds the task maximum")

        cost_efficiency = 1.0 - (bid.estimated_cost / announcement.maximum_cost)
        total_score = (
            self._weights.capability * capability_score
            + self._weights.confidence * bid.confidence
            + self._weights.availability * bid.availability
            + self._weights.cost_efficiency * cost_efficiency
        )

        # Stable rounding makes score evidence easier to compare across logs and UIs
        # without changing the policy itself.
        return BidScorecard(
            bid_id=bid.bid_id,
            bidder_id=bid.bidder_id,
            capability_score=round(capability_score, 12),
            confidence=round(bid.confidence, 12),
            availability=round(bid.availability, 12),
            cost_efficiency=round(cost_efficiency, 12),
            estimated_cost=round(bid.estimated_cost, 12),
            total_score=round(total_score, 12),
        )

    def rank_scorecards(
        self,
        scorecards: tuple[BidScorecard, ...],
    ) -> tuple[BidScorecard, ...]:
        """Return scorecards in deterministic settlement order."""

        return tuple(
            sorted(
                scorecards,
                key=lambda card: (
                    -card.total_score,
                    -card.capability_score,
                    card.estimated_cost,
                    card.bidder_id,
                ),
            )
        )

    def settle(
        self,
        *,
        protocol: AuctionRoundProtocol,
        awarded_at: datetime,
    ) -> SettlementResult:
        """Settle one closed round using only its admitted bids."""

        if protocol.is_open:
            raise SettlementError("auction round must be closed before settlement")

        bids = protocol.accepted_bids()
        if not bids:
            raise SettlementError("auction round has no admitted bids to settle")

        announcement = protocol.announcement
        scorecards = tuple(
            self.score_bid(bid=bid, announcement=announcement) for bid in bids
        )
        ranked = self.rank_scorecards(scorecards)
        winner = ranked[0]

        award = TaskAward(
            award_id=(
                f"award:{announcement.auction_id}:"
                f"{announcement.auction_round}:{winner.bidder_id}"
            ),
            auction_id=announcement.auction_id,
            task_id=announcement.task_id,
            auction_round=announcement.auction_round,
            winning_bid_id=winner.bid_id,
            awarded_agent_id=winner.bidder_id,
            winning_score=winner.total_score,
            awarded_at=awarded_at,
        )
        return SettlementResult(award=award, ranked_scorecards=ranked)
