"""Integration tests for deterministic Agent 11 settlement."""

from datetime import datetime, timezone

import pytest

from auction_coordination.auction import AuctionRoundProtocol
from auction_coordination.bidding import BidAction, LocalBidPolicy
from auction_coordination.models import TaskAnnouncement
from auction_coordination.scenario import build_default_registry, default_bid_settings
from auction_coordination.settlement import DeterministicSettlement, SettlementError

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


def admitted_market_round() -> AuctionRoundProtocol:
    registry = build_default_registry()
    protocol = AuctionRoundProtocol(market_announcement(), registry)
    policy = LocalBidPolicy(registry)

    for settings in default_bid_settings().values():
        decision = policy.evaluate(
            announcement=protocol.announcement,
            settings=settings,
            submitted_at=NOW,
        )
        if decision.action is BidAction.BID:
            assert decision.bid is not None
            receipt = protocol.submit_bid(decision.bid)
        else:
            assert decision.abstention is not None
            receipt = protocol.submit_abstention(decision.abstention)
        assert receipt.accepted

    return protocol


def test_settlement_refuses_open_round() -> None:
    registry = build_default_registry()
    protocol = admitted_market_round()

    with pytest.raises(SettlementError, match="closed"):
        DeterministicSettlement(registry).settle(protocol=protocol, awarded_at=NOW)


def test_settlement_refuses_closed_round_without_bids() -> None:
    registry = build_default_registry()
    protocol = AuctionRoundProtocol(market_announcement(), registry)
    protocol.close()

    with pytest.raises(SettlementError, match="no admitted bids"):
        DeterministicSettlement(registry).settle(protocol=protocol, awarded_at=NOW)


def test_full_market_round_selects_atlas_deterministically() -> None:
    registry = build_default_registry()
    protocol = admitted_market_round()
    summary = protocol.close()

    result = DeterministicSettlement(registry).settle(
        protocol=protocol,
        awarded_at=NOW,
    )

    assert summary.response_count == 6
    assert len(summary.accepted_bid_ids) == 4
    assert len(summary.accepted_abstention_ids) == 2
    assert result.award.awarded_agent_id == "peer.atlas"
    assert result.award.winning_bid_id == "bid:auction.market.1:1:peer.atlas"
    assert result.winning_scorecard.capability_score == pytest.approx(0.94)
    assert result.winning_scorecard.total_score == pytest.approx(0.8385)
    assert result.second_place_score is not None
    assert result.second_place_score < result.award.winning_score


def test_settlement_preserves_full_ranked_evidence() -> None:
    registry = build_default_registry()
    protocol = admitted_market_round()
    protocol.close()

    result = DeterministicSettlement(registry).settle(
        protocol=protocol,
        awarded_at=NOW,
    )

    assert [card.bidder_id for card in result.ranked_scorecards] == [
        "peer.atlas",
        "peer.veritas",
        "peer.quill",
        "peer.mosaic",
    ]
    assert result.award.winning_score == result.ranked_scorecards[0].total_score
