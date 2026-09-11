"""Tests for Agent 11 auction-round admission and collection controls."""

from dataclasses import fields
from datetime import datetime, timezone

from auction_coordination.auction import (
    AuctionRoundProtocol,
    SubmissionCode,
)
from auction_coordination.bidding import BidAction, LocalBidPolicy
from auction_coordination.models import (
    AbstentionReason,
    Bid,
    BidAbstention,
    TaskAnnouncement,
)
from auction_coordination.scenario import build_default_registry, default_bid_settings

NOW = datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc)


def announcement(**overrides) -> TaskAnnouncement:
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


def bid(**overrides) -> Bid:
    values = {
        "bid_id": "bid:auction.market.1:1:peer.atlas",
        "auction_id": "auction.market.1",
        "task_id": "task.market",
        "auction_round": 1,
        "bidder_id": "peer.atlas",
        "capability": "market_research",
        "confidence": 0.91,
        "availability": 0.75,
        "estimated_cost": 62.0,
        "submitted_at": NOW,
    }
    values.update(overrides)
    return Bid(**values)


def abstention(**overrides) -> BidAbstention:
    values = {
        "abstention_id": "abstain:auction.market.1:1:peer.ledger",
        "auction_id": "auction.market.1",
        "task_id": "task.market",
        "auction_round": 1,
        "bidder_id": "peer.ledger",
        "reason": AbstentionReason.CAPABILITY_MISMATCH,
        "detail": "Ledger is not registered for market research.",
        "submitted_at": NOW,
    }
    values.update(overrides)
    return BidAbstention(**values)


def protocol(**announcement_overrides) -> AuctionRoundProtocol:
    return AuctionRoundProtocol(
        announcement(**announcement_overrides),
        build_default_registry(),
    )


def test_valid_bid_is_admitted_without_settlement() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid())

    assert receipt.accepted is True
    assert receipt.code is SubmissionCode.ACCEPTED
    assert [item.bidder_id for item in auction.accepted_bids()] == ["peer.atlas"]
    summary_field_names = {field.name for field in fields(auction.summary())}
    assert "winner" not in summary_field_names
    assert "winning_score" not in summary_field_names


def test_wrong_auction_id_is_rejected() -> None:
    auction = protocol()

    receipt = auction.submit_bid(
        bid(bid_id="bid:wrong:1:peer.atlas", auction_id="auction.wrong.1")
    )

    assert receipt.code is SubmissionCode.AUCTION_ID_MISMATCH
    assert auction.accepted_bids() == ()


def test_wrong_task_id_is_rejected() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid(task_id="task.financial"))

    assert receipt.code is SubmissionCode.TASK_ID_MISMATCH


def test_stale_or_future_round_is_rejected() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid(auction_round=2))

    assert receipt.code is SubmissionCode.ROUND_MISMATCH


def test_unknown_bidder_is_rejected() -> None:
    auction = protocol()

    receipt = auction.submit_bid(
        bid(
            bid_id="bid:auction.market.1:1:peer.unknown",
            bidder_id="peer.unknown",
        )
    )

    assert receipt.code is SubmissionCode.UNKNOWN_BIDDER


def test_bid_capability_must_match_requested_capability() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid(capability="financial_analysis"))

    assert receipt.code is SubmissionCode.CAPABILITY_MISMATCH


def test_bidder_must_be_registered_for_requested_capability() -> None:
    auction = protocol(
        task_id="task.risk",
        auction_id="auction.risk.1",
        requested_capability="operating_risk",
    )
    atlas_bid = bid(
        bid_id="bid:auction.risk.1:1:peer.atlas",
        auction_id="auction.risk.1",
        task_id="task.risk",
        capability="operating_risk",
    )

    receipt = auction.submit_bid(atlas_bid)

    assert receipt.code is SubmissionCode.UNREGISTERED_CAPABILITY


def test_registered_capability_must_meet_task_floor() -> None:
    auction = protocol(minimum_capability_score=0.90)
    veritas_bid = bid(
        bid_id="bid:auction.market.1:1:peer.veritas",
        bidder_id="peer.veritas",
        confidence=0.81,
        availability=0.85,
        estimated_cost=66.0,
    )

    receipt = auction.submit_bid(veritas_bid)

    assert receipt.code is SubmissionCode.BELOW_MINIMUM_CAPABILITY


def test_zero_availability_is_rejected_even_when_bid_schema_is_valid() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid(availability=0.0))

    assert receipt.code is SubmissionCode.UNAVAILABLE


def test_bid_above_task_budget_is_rejected() -> None:
    auction = protocol()

    receipt = auction.submit_bid(bid(estimated_cost=101.0))

    assert receipt.code is SubmissionCode.OVER_BUDGET


def test_duplicate_submission_id_is_rejected_across_different_bidders() -> None:
    auction = protocol()
    first = bid()
    second = bid(
        bidder_id="peer.veritas",
        confidence=0.81,
        availability=0.85,
        estimated_cost=66.0,
    )

    assert auction.submit_bid(first).accepted is True
    receipt = auction.submit_bid(second)

    assert receipt.code is SubmissionCode.DUPLICATE_RESPONSE_ID
    assert len(auction.accepted_bids()) == 1


def test_one_accepted_response_per_bidder_per_round() -> None:
    auction = protocol()

    assert auction.submit_bid(bid()).accepted is True
    receipt = auction.submit_bid(
        bid(bid_id="bid:auction.market.1:1:peer.atlas:retry")
    )

    assert receipt.code is SubmissionCode.DUPLICATE_BIDDER_RESPONSE


def test_accepted_abstention_counts_as_the_bidder_response() -> None:
    auction = protocol()

    assert auction.submit_abstention(abstention()).accepted is True
    ledger_bid = bid(
        bid_id="bid:auction.market.1:1:peer.ledger",
        bidder_id="peer.ledger",
    )
    receipt = auction.submit_bid(ledger_bid)

    assert receipt.code is SubmissionCode.DUPLICATE_BIDDER_RESPONSE
    assert len(auction.accepted_abstentions()) == 1


def test_rejected_submission_does_not_consume_response_slot() -> None:
    auction = protocol()

    rejected = auction.submit_bid(bid(estimated_cost=101.0))
    corrected = auction.submit_bid(bid())

    assert rejected.code is SubmissionCode.OVER_BUDGET
    assert corrected.code is SubmissionCode.ACCEPTED
    assert auction.summary().response_count == 1


def test_closed_round_rejects_new_responses() -> None:
    auction = protocol()
    auction.close()

    receipt = auction.submit_bid(bid())

    assert receipt.code is SubmissionCode.AUCTION_CLOSED
    assert auction.summary().is_open is False


def test_abstention_must_match_active_auction_context() -> None:
    auction = protocol()

    receipt = auction.submit_abstention(abstention(auction_round=2))

    assert receipt.code is SubmissionCode.ROUND_MISMATCH
    assert auction.accepted_abstentions() == ()


def test_receipts_preserve_accepted_and_rejected_audit_events() -> None:
    auction = protocol()

    auction.submit_bid(bid(estimated_cost=101.0))
    auction.submit_bid(bid())

    receipts = auction.receipts()
    assert [receipt.code for receipt in receipts] == [
        SubmissionCode.OVER_BUDGET,
        SubmissionCode.ACCEPTED,
    ]


def test_local_peer_decisions_feed_one_round_without_selecting_winner() -> None:
    registry = build_default_registry()
    policy = LocalBidPolicy(registry)
    auction = AuctionRoundProtocol(announcement(), registry)

    for settings in default_bid_settings().values():
        decision = policy.evaluate(
            announcement=auction.announcement,
            settings=settings,
            submitted_at=NOW,
        )
        if decision.action is BidAction.BID:
            assert decision.bid is not None
            receipt = auction.submit_bid(decision.bid)
        else:
            assert decision.abstention is not None
            receipt = auction.submit_abstention(decision.abstention)
        assert receipt.code is SubmissionCode.ACCEPTED

    summary = auction.close()
    assert summary.response_count == 6
    assert len(summary.accepted_bid_ids) == 4
    assert len(summary.accepted_abstention_ids) == 2
    assert summary.is_open is False
