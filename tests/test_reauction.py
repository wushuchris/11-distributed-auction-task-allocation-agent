"""Tests for bounded Agent 11 failure recovery and reauctioning."""

from datetime import datetime, timedelta, timezone

import pytest

from auction_coordination.auction import AuctionRoundProtocol, SubmissionCode
from auction_coordination.bidding import BidAction, LocalBidPolicy
from auction_coordination.models import FailureClass, TaskAnnouncement, TaskFailure
from auction_coordination.reauction import (
    BoundedReauctionPolicy,
    ReauctionError,
    RecoveryAction,
)
from auction_coordination.scenario import build_default_registry, default_bid_settings
from auction_coordination.settlement import DeterministicSettlement

NOW = datetime(2026, 9, 11, 22, 20, tzinfo=timezone.utc)
ROUND_TWO_TIME = NOW + timedelta(minutes=5)
SECOND_FAILURE_TIME = NOW + timedelta(minutes=10)


def market_announcement(*, auction_round: int = 1, opened_at: datetime = NOW) -> TaskAnnouncement:
    return TaskAnnouncement(
        task_id="task.market",
        auction_id="auction.market.1",
        auction_round=auction_round,
        requested_capability="market_research",
        summary="Assess the fictional target market and competitive environment.",
        minimum_capability_score=0.70,
        maximum_cost=100.0,
        required_input_ids=(),
        max_auction_rounds=2,
        opened_at=opened_at,
    )


def run_round(
    announcement: TaskAnnouncement,
    *,
    excluded_bidder_ids: tuple[str, ...] = (),
) -> tuple[AuctionRoundProtocol, list[tuple[str, SubmissionCode]]]:
    registry = build_default_registry()
    protocol = AuctionRoundProtocol(
        announcement,
        registry,
        excluded_bidder_ids=excluded_bidder_ids,
    )
    bid_policy = LocalBidPolicy(registry)
    receipts: list[tuple[str, SubmissionCode]] = []

    for settings in default_bid_settings().values():
        decision = bid_policy.evaluate(
            announcement=announcement,
            settings=settings,
            submitted_at=announcement.opened_at,
        )
        if decision.action is BidAction.BID:
            assert decision.bid is not None
            receipt = protocol.submit_bid(decision.bid)
        else:
            assert decision.abstention is not None
            receipt = protocol.submit_abstention(decision.abstention)
        receipts.append((settings.agent_id, receipt.code))

    return protocol, receipts


def settle_round(protocol: AuctionRoundProtocol):
    registry = build_default_registry()
    protocol.close()
    return DeterministicSettlement(registry).settle(
        protocol=protocol,
        awarded_at=protocol.announcement.opened_at,
    )


def first_round_award():
    protocol, _ = run_round(market_announcement())
    result = settle_round(protocol)
    assert result.award.awarded_agent_id == "peer.atlas"
    return result.award


def test_record_failure_binds_failure_to_awarded_worker() -> None:
    award = first_round_award()
    policy = BoundedReauctionPolicy()

    failure = policy.record_failure(
        award=award,
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Synthetic worker failed before producing a valid work product.",
        failed_at=NOW,
    )

    assert failure.award_id == award.award_id
    assert failure.agent_id == "peer.atlas"
    assert failure.auction_round == 1
    assert failure.failure_class is FailureClass.EXECUTION_ERROR


def test_reauction_rejects_failure_from_wrong_worker() -> None:
    award = first_round_award()
    policy = BoundedReauctionPolicy()
    valid_failure = policy.record_failure(
        award=award,
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Synthetic failure.",
        failed_at=NOW,
    )
    wrong_failure = TaskFailure(
        failure_id="failure:wrong-worker",
        award_id=valid_failure.award_id,
        auction_id=valid_failure.auction_id,
        task_id=valid_failure.task_id,
        auction_round=valid_failure.auction_round,
        agent_id="peer.veritas",
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Incorrectly attributed failure.",
        failed_at=NOW,
    )

    with pytest.raises(ReauctionError, match="awarded worker"):
        policy.plan(
            announcement=market_announcement(),
            award=award,
            failure=wrong_failure,
            reopened_at=ROUND_TWO_TIME,
        )


def test_first_failure_opens_round_two_and_excludes_failed_winner() -> None:
    award = first_round_award()
    policy = BoundedReauctionPolicy()
    failure = policy.record_failure(
        award=award,
        failure_class=FailureClass.INVALID_OUTPUT,
        detail="Synthetic output failed validation.",
        failed_at=NOW,
    )

    plan = policy.plan(
        announcement=market_announcement(),
        award=award,
        failure=failure,
        reopened_at=ROUND_TWO_TIME,
    )

    assert plan.action is RecoveryAction.REAUCTION
    assert plan.next_announcement is not None
    assert plan.next_announcement.auction_round == 2
    assert plan.next_announcement.auction_id == "auction.market.1"
    assert plan.excluded_bidder_ids == ("peer.atlas",)


def test_failed_winner_cannot_reenter_immediate_reauction() -> None:
    registry = build_default_registry()
    award = first_round_award()
    recovery = BoundedReauctionPolicy()
    failure = recovery.record_failure(
        award=award,
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Synthetic execution failure.",
        failed_at=NOW,
    )
    plan = recovery.plan(
        announcement=market_announcement(),
        award=award,
        failure=failure,
        reopened_at=ROUND_TWO_TIME,
    )
    protocol = recovery.open_reauction_round(plan=plan, registry=registry)
    atlas_settings = default_bid_settings()["peer.atlas"]
    decision = LocalBidPolicy(registry).evaluate(
        announcement=protocol.announcement,
        settings=atlas_settings,
        submitted_at=ROUND_TWO_TIME,
    )

    assert decision.action is BidAction.BID
    assert decision.bid is not None
    receipt = protocol.submit_bid(decision.bid)

    assert not receipt.accepted
    assert receipt.code is SubmissionCode.EXCLUDED_BIDDER
    assert protocol.summary().response_count == 0


def test_reauction_reallocates_failed_atlas_task_to_veritas() -> None:
    registry = build_default_registry()
    recovery = BoundedReauctionPolicy()

    round_one_protocol, _ = run_round(market_announcement())
    round_one = settle_round(round_one_protocol)
    assert round_one.award.awarded_agent_id == "peer.atlas"

    first_failure = recovery.record_failure(
        award=round_one.award,
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="Atlas synthetic handler failed explicitly.",
        failed_at=NOW,
    )
    plan = recovery.plan(
        announcement=market_announcement(),
        award=round_one.award,
        failure=first_failure,
        reopened_at=ROUND_TWO_TIME,
    )
    assert plan.next_announcement is not None

    round_two_protocol, receipts = run_round(
        plan.next_announcement,
        excluded_bidder_ids=plan.excluded_bidder_ids,
    )
    round_two_summary = round_two_protocol.summary()

    atlas_receipt = dict(receipts)["peer.atlas"]
    assert atlas_receipt is SubmissionCode.EXCLUDED_BIDDER
    assert round_two_summary.response_count == 5
    assert len(round_two_summary.accepted_bid_ids) == 3
    assert len(round_two_summary.accepted_abstention_ids) == 2
    assert round_two_summary.excluded_bidder_ids == ("peer.atlas",)

    round_two = settle_round(round_two_protocol)
    assert round_two.award.auction_round == 2
    assert round_two.award.awarded_agent_id == "peer.veritas"
    assert round_two.winning_scorecard.total_score == pytest.approx(0.7575)


def test_second_round_failure_escalates_instead_of_looping() -> None:
    registry = build_default_registry()
    recovery = BoundedReauctionPolicy()

    round_one_protocol, _ = run_round(market_announcement())
    round_one = settle_round(round_one_protocol)
    first_failure = recovery.record_failure(
        award=round_one.award,
        failure_class=FailureClass.EXECUTION_ERROR,
        detail="First synthetic failure.",
        failed_at=NOW,
    )
    first_plan = recovery.plan(
        announcement=market_announcement(),
        award=round_one.award,
        failure=first_failure,
        reopened_at=ROUND_TWO_TIME,
    )
    assert first_plan.next_announcement is not None

    round_two_protocol, _ = run_round(
        first_plan.next_announcement,
        excluded_bidder_ids=first_plan.excluded_bidder_ids,
    )
    round_two = settle_round(round_two_protocol)
    assert round_two.award.awarded_agent_id == "peer.veritas"

    second_failure = recovery.record_failure(
        award=round_two.award,
        failure_class=FailureClass.INVALID_OUTPUT,
        detail="Second-round synthetic output also failed validation.",
        failed_at=SECOND_FAILURE_TIME,
    )
    final_plan = recovery.plan(
        announcement=first_plan.next_announcement,
        award=round_two.award,
        failure=second_failure,
        reopened_at=SECOND_FAILURE_TIME,
    )

    assert final_plan.action is RecoveryAction.ESCALATE
    assert final_plan.next_announcement is None
    assert final_plan.excluded_bidder_ids == ("peer.veritas",)
    assert "human escalation required" in final_plan.reason

    with pytest.raises(ReauctionError, match="escalation"):
        recovery.open_reauction_round(plan=final_plan, registry=registry)
