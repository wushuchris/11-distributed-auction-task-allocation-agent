"""Validation tests for Agent 11 auction contracts."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from auction_coordination.models import (
    AbstentionReason,
    AuctionState,
    AuctionStatus,
    Bid,
    BidAbstention,
    CapabilityProfile,
    FailureClass,
    RegisteredCapability,
    TaskAnnouncement,
    TaskAward,
    TaskFailure,
)

NOW = datetime(2026, 9, 11, 22, 0, tzinfo=timezone.utc)


def capability_profile() -> CapabilityProfile:
    return CapabilityProfile(
        agent_id="peer.atlas",
        display_name="Atlas Research",
        capabilities=(
            RegisteredCapability(capability="market_research", score=0.92),
            RegisteredCapability(capability="source_research", score=0.86),
        ),
    )


def task_announcement(**overrides) -> TaskAnnouncement:
    values = {
        "task_id": "task.market",
        "auction_id": "auction.market.1",
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


def valid_bid(**overrides) -> Bid:
    values = {
        "bid_id": "bid.atlas.market.1",
        "auction_id": "auction.market.1",
        "task_id": "task.market",
        "auction_round": 1,
        "bidder_id": "peer.atlas",
        "capability": "market_research",
        "confidence": 0.88,
        "availability": 0.75,
        "estimated_cost": 60.0,
        "submitted_at": NOW,
    }
    values.update(overrides)
    return Bid(**values)


def test_capability_profile_preserves_registered_scores() -> None:
    profile = capability_profile()

    assert profile.score_for("market_research") == pytest.approx(0.92)
    assert profile.score_for("financial_analysis") is None


def test_capability_profile_rejects_duplicate_capability_names() -> None:
    with pytest.raises(ValidationError):
        CapabilityProfile(
            agent_id="peer.atlas",
            display_name="Atlas Research",
            capabilities=(
                RegisteredCapability(capability="market_research", score=0.92),
                RegisteredCapability(capability="market_research", score=0.80),
            ),
        )


def test_registered_capability_score_must_be_normalized() -> None:
    with pytest.raises(ValidationError):
        RegisteredCapability(capability="market_research", score=1.01)


def test_task_announcement_rejects_duplicate_inputs() -> None:
    with pytest.raises(ValidationError):
        task_announcement(required_input_ids=("work.1", "work.1"))


def test_task_announcement_rejects_round_beyond_configured_limit() -> None:
    with pytest.raises(ValidationError):
        task_announcement(auction_round=3, max_auction_rounds=2)


def test_task_announcement_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        task_announcement(opened_at=datetime(2026, 9, 11, 22, 0))


def test_bid_rejects_self_declared_capability_score() -> None:
    with pytest.raises(ValidationError):
        Bid(
            **valid_bid().model_dump(),
            capability_score=0.99,
        )


def test_bid_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        valid_bid(confidence=1.10)


def test_bid_rejects_non_positive_cost() -> None:
    with pytest.raises(ValidationError):
        valid_bid(estimated_cost=0.0)


def test_abstention_is_explicit_and_auditable() -> None:
    abstention = BidAbstention(
        abstention_id="abstain.mosaic.market.1",
        auction_id="auction.market.1",
        task_id="task.market",
        auction_round=1,
        bidder_id="peer.mosaic",
        reason=AbstentionReason.LOW_CONFIDENCE,
        detail="Current evidence coverage is below the local bidding threshold.",
        submitted_at=NOW,
    )

    assert abstention.reason is AbstentionReason.LOW_CONFIDENCE


def test_awarded_state_requires_winner_fields() -> None:
    with pytest.raises(ValidationError):
        AuctionState(
            auction_id="auction.market.1",
            task_id="task.market",
            auction_round=1,
            status=AuctionStatus.AWARDED,
            valid_bid_ids=("bid.atlas.market.1",),
        )


def test_winning_bid_must_be_one_of_valid_bids() -> None:
    with pytest.raises(ValidationError):
        AuctionState(
            auction_id="auction.market.1",
            task_id="task.market",
            auction_round=1,
            status=AuctionStatus.AWARDED,
            valid_bid_ids=("bid.atlas.market.1",),
            winning_bid_id="bid.ledger.market.1",
            awarded_agent_id="peer.ledger",
        )


def test_failed_state_requires_failure_record() -> None:
    with pytest.raises(ValidationError):
        AuctionState(
            auction_id="auction.market.1",
            task_id="task.market",
            auction_round=1,
            status=AuctionStatus.FAILED,
            valid_bid_ids=("bid.atlas.market.1",),
            winning_bid_id="bid.atlas.market.1",
            awarded_agent_id="peer.atlas",
        )


def test_no_valid_bids_state_cannot_contain_valid_bids() -> None:
    with pytest.raises(ValidationError):
        AuctionState(
            auction_id="auction.market.1",
            task_id="task.market",
            auction_round=1,
            status=AuctionStatus.NO_VALID_BIDS,
            valid_bid_ids=("bid.atlas.market.1",),
        )


def test_task_award_records_normalized_winning_score() -> None:
    award = TaskAward(
        award_id="award.market.1",
        auction_id="auction.market.1",
        task_id="task.market",
        auction_round=1,
        winning_bid_id="bid.atlas.market.1",
        awarded_agent_id="peer.atlas",
        winning_score=0.84,
        awarded_at=NOW,
    )

    assert award.winning_score == pytest.approx(0.84)


def test_task_failure_records_failure_class_without_deciding_reauction_policy() -> None:
    failure = TaskFailure(
        failure_id="failure.market.1",
        award_id="award.market.1",
        auction_id="auction.market.1",
        task_id="task.market",
        auction_round=1,
        agent_id="peer.atlas",
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Synthetic handler failed before producing a valid work product.",
        failed_at=NOW,
    )

    assert failure.failure_class is FailureClass.EXECUTION_ERROR
