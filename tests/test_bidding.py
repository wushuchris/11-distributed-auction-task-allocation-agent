"""Tests for Agent 11 deterministic local BID / ABSTAIN policy."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from auction_coordination.bidding import (
    BidAction,
    BiddingPolicyError,
    LocalBidPolicy,
    PeerBidSettings,
)
from auction_coordination.models import AbstentionReason, TaskAnnouncement
from auction_coordination.registry import RegistryError
from auction_coordination.scenario import build_default_registry, default_bid_settings

NOW = datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc)


def task(**overrides) -> TaskAnnouncement:
    values = {
        "task_id": "task.market",
        "auction_id": "auction.market",
        "auction_round": 1,
        "requested_capability": "market_research",
        "summary": "Assess the fictional target market and competitive environment.",
        "minimum_capability_score": 0.70,
        "maximum_cost": 100.0,
        "required_input_ids": (),
        "max_auction_rounds": 2,
        "opened_at": NOW,
    }
    values.update(overrides)
    return TaskAnnouncement(**values)


def policy() -> LocalBidPolicy:
    return LocalBidPolicy(build_default_registry())


def settings(agent_id: str) -> PeerBidSettings:
    return default_bid_settings()[agent_id]


def test_qualified_peer_submits_bid() -> None:
    decision = policy().evaluate(
        announcement=task(),
        settings=settings("peer.atlas"),
        submitted_at=NOW,
    )

    assert decision.action is BidAction.BID
    assert decision.bid is not None
    assert decision.abstention is None
    assert decision.bid.bidder_id == "peer.atlas"
    assert decision.bid.capability == "market_research"
    assert decision.bid.confidence == pytest.approx(0.91)
    assert decision.bid.availability == pytest.approx(0.75)
    assert decision.bid.estimated_cost == pytest.approx(62.0)


def test_bid_identifier_is_deterministic() -> None:
    first = policy().evaluate(
        announcement=task(),
        settings=settings("peer.atlas"),
        submitted_at=NOW,
    )
    second = policy().evaluate(
        announcement=task(),
        settings=settings("peer.atlas"),
        submitted_at=NOW,
    )

    assert first.bid is not None
    assert second.bid is not None
    assert first.bid.bid_id == second.bid.bid_id


def test_unregistered_capability_abstains() -> None:
    decision = policy().evaluate(
        announcement=task(requested_capability="financial_analysis"),
        settings=settings("peer.atlas"),
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.CAPABILITY_MISMATCH


def test_registered_but_below_task_floor_abstains() -> None:
    decision = policy().evaluate(
        announcement=task(minimum_capability_score=0.80),
        settings=settings("peer.veritas"),
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.CAPABILITY_MISMATCH


def test_unavailable_peer_abstains() -> None:
    atlas = replace(settings("peer.atlas"), availability=0.05)

    decision = policy().evaluate(
        announcement=task(),
        settings=atlas,
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.UNAVAILABLE


def test_over_budget_peer_abstains() -> None:
    decision = policy().evaluate(
        announcement=task(maximum_cost=60.0),
        settings=settings("peer.atlas"),
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.OVER_BUDGET


def test_low_confidence_peer_abstains() -> None:
    atlas = replace(
        settings("peer.atlas"),
        confidence_by_capability={
            **settings("peer.atlas").confidence_by_capability,
            "market_research": 0.55,
        },
    )

    decision = policy().evaluate(
        announcement=task(),
        settings=atlas,
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.LOW_CONFIDENCE


def test_missing_local_economics_abstains_fail_closed() -> None:
    atlas = replace(settings("peer.atlas"), cost_by_capability={})

    decision = policy().evaluate(
        announcement=task(),
        settings=atlas,
        submitted_at=NOW,
    )

    assert decision.action is BidAction.ABSTAIN
    assert decision.abstention is not None
    assert decision.abstention.reason is AbstentionReason.LOCAL_POLICY


def test_unknown_peer_cannot_participate() -> None:
    unknown = PeerBidSettings(
        agent_id="peer.unknown",
        availability=1.0,
        confidence_by_capability={"market_research": 0.99},
        cost_by_capability={"market_research": 1.0},
    )

    with pytest.raises(RegistryError):
        policy().evaluate(
            announcement=task(),
            settings=unknown,
            submitted_at=NOW,
        )


def test_invalid_local_settings_are_rejected_before_bidding() -> None:
    with pytest.raises(BiddingPolicyError):
        PeerBidSettings(
            agent_id="peer.atlas",
            availability=1.1,
            confidence_by_capability={"market_research": 0.90},
            cost_by_capability={"market_research": 60.0},
        )


def test_policy_does_not_select_a_winner() -> None:
    decisions = [
        policy().evaluate(
            announcement=task(),
            settings=settings(agent_id),
            submitted_at=NOW,
        )
        for agent_id in ("peer.atlas", "peer.veritas", "peer.mosaic", "peer.quill")
    ]

    assert sum(decision.action is BidAction.BID for decision in decisions) == 4
    assert all(not hasattr(decision, "winning_bid_id") for decision in decisions)
