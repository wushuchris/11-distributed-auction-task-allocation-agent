"""Tests for deterministic Agent 11 bid scoring and tie-breaking."""

from datetime import datetime, timezone

import pytest

from auction_coordination.models import Bid, TaskAnnouncement
from auction_coordination.scenario import build_default_registry
from auction_coordination.settlement import (
    DEFAULT_SCORING_WEIGHTS,
    BidScorecard,
    DeterministicSettlement,
    ScoringWeights,
    SettlementError,
)

NOW = datetime(2026, 9, 11, 22, 15, tzinfo=timezone.utc)


def market_announcement() -> TaskAnnouncement:
    return TaskAnnouncement(
        task_id="task.market",
        auction_id="auction.market.1",
        auction_round=1,
        requested_capability="market_research",
        summary="Assess the fictional target market and competitive environment.",
        minimum_capability_score=0.70,
        maximum_cost=100.0,
        required_input_ids=(),
        max_auction_rounds=2,
        opened_at=NOW,
    )


def test_default_scoring_weights_match_design() -> None:
    assert DEFAULT_SCORING_WEIGHTS.capability == pytest.approx(0.45)
    assert DEFAULT_SCORING_WEIGHTS.confidence == pytest.approx(0.25)
    assert DEFAULT_SCORING_WEIGHTS.availability == pytest.approx(0.20)
    assert DEFAULT_SCORING_WEIGHTS.cost_efficiency == pytest.approx(0.10)


def test_scoring_weights_must_sum_to_one() -> None:
    with pytest.raises(SettlementError):
        ScoringWeights(
            capability=0.50,
            confidence=0.25,
            availability=0.20,
            cost_efficiency=0.10,
        )


def test_scoring_weights_reject_negative_component() -> None:
    with pytest.raises(SettlementError):
        ScoringWeights(
            capability=-0.01,
            confidence=0.51,
            availability=0.30,
            cost_efficiency=0.20,
        )


def test_score_bid_uses_registry_capability_and_task_budget() -> None:
    settlement = DeterministicSettlement(build_default_registry())
    bid = Bid(
        bid_id="bid:auction.market.1:1:peer.atlas",
        auction_id="auction.market.1",
        task_id="task.market",
        auction_round=1,
        bidder_id="peer.atlas",
        capability="market_research",
        confidence=0.91,
        availability=0.75,
        estimated_cost=62.0,
        submitted_at=NOW,
    )

    score = settlement.score_bid(bid=bid, announcement=market_announcement())

    assert score.capability_score == pytest.approx(0.94)
    assert score.cost_efficiency == pytest.approx(0.38)
    assert score.total_score == pytest.approx(
        0.45 * 0.94 + 0.25 * 0.91 + 0.20 * 0.75 + 0.10 * 0.38
    )


def card(
    bidder_id: str,
    *,
    total: float = 0.85,
    capability: float = 0.90,
    cost: float = 50.0,
) -> BidScorecard:
    return BidScorecard(
        bid_id=f"bid.{bidder_id}",
        bidder_id=bidder_id,
        capability_score=capability,
        confidence=0.80,
        availability=0.80,
        cost_efficiency=0.50,
        estimated_cost=cost,
        total_score=total,
    )


def test_rank_breaks_total_score_tie_with_higher_capability() -> None:
    settlement = DeterministicSettlement(build_default_registry())
    ranked = settlement.rank_scorecards(
        (
            card("peer.a", capability=0.80),
            card("peer.b", capability=0.90),
        )
    )

    assert ranked[0].bidder_id == "peer.b"


def test_rank_breaks_score_and_capability_tie_with_lower_cost() -> None:
    settlement = DeterministicSettlement(build_default_registry())
    ranked = settlement.rank_scorecards(
        (
            card("peer.a", cost=60.0),
            card("peer.b", cost=50.0),
        )
    )

    assert ranked[0].bidder_id == "peer.b"


def test_rank_breaks_remaining_tie_with_lexical_agent_id() -> None:
    settlement = DeterministicSettlement(build_default_registry())
    ranked = settlement.rank_scorecards((card("peer.zeta"), card("peer.alpha")))

    assert ranked[0].bidder_id == "peer.alpha"
