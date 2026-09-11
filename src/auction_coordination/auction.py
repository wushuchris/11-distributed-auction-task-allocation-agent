"""Auction-round collection and admission controls for Agent 11.

This module answers one question only: may a submitted BID or ABSTAIN response
enter the current auction round? It deliberately does not score bids, rank
peers, or choose a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import Bid, BidAbstention, TaskAnnouncement
from .registry import PeerRegistry


class SubmissionCode(str, Enum):
    """Deterministic admission outcomes for auction-round responses."""

    ACCEPTED = "accepted"
    AUCTION_CLOSED = "auction_closed"
    AUCTION_ID_MISMATCH = "auction_id_mismatch"
    TASK_ID_MISMATCH = "task_id_mismatch"
    ROUND_MISMATCH = "round_mismatch"
    UNKNOWN_BIDDER = "unknown_bidder"
    CAPABILITY_MISMATCH = "capability_mismatch"
    UNREGISTERED_CAPABILITY = "unregistered_capability"
    BELOW_MINIMUM_CAPABILITY = "below_minimum_capability"
    UNAVAILABLE = "unavailable"
    OVER_BUDGET = "over_budget"
    DUPLICATE_RESPONSE_ID = "duplicate_response_id"
    DUPLICATE_BIDDER_RESPONSE = "duplicate_bidder_response"


@dataclass(frozen=True)
class SubmissionReceipt:
    """Auditable result of attempting to admit one auction response."""

    accepted: bool
    code: SubmissionCode
    detail: str
    submission_id: str
    bidder_id: str


@dataclass(frozen=True)
class AuctionRoundSummary:
    """Read-only summary of the responses admitted to one auction round.

    No score, ranking, or winner appears here by design. Settlement belongs to
    a later layer.
    """

    auction_id: str
    task_id: str
    auction_round: int
    is_open: bool
    accepted_bid_ids: tuple[str, ...]
    accepted_abstention_ids: tuple[str, ...]
    response_count: int


class AuctionRoundProtocol:
    """Collect and validate responses for exactly one task auction round."""

    def __init__(
        self,
        announcement: TaskAnnouncement,
        registry: PeerRegistry,
    ) -> None:
        self._announcement = announcement
        self._registry = registry
        self._is_open = True
        self._bids: dict[str, Bid] = {}
        self._abstentions: dict[str, BidAbstention] = {}
        self._response_ids: set[str] = set()
        self._responded_bidder_ids: set[str] = set()
        self._receipts: list[SubmissionReceipt] = []

    @property
    def announcement(self) -> TaskAnnouncement:
        return self._announcement

    @property
    def is_open(self) -> bool:
        return self._is_open

    def submit_bid(self, bid: Bid) -> SubmissionReceipt:
        """Validate and, when admissible, record one bid."""

        common_rejection = self._validate_common(
            auction_id=bid.auction_id,
            task_id=bid.task_id,
            auction_round=bid.auction_round,
            bidder_id=bid.bidder_id,
            submission_id=bid.bid_id,
        )
        if common_rejection is not None:
            return common_rejection

        if bid.capability != self._announcement.requested_capability:
            return self._reject(
                submission_id=bid.bid_id,
                bidder_id=bid.bidder_id,
                code=SubmissionCode.CAPABILITY_MISMATCH,
                detail=(
                    f"bid capability {bid.capability} does not match requested capability "
                    f"{self._announcement.requested_capability}"
                ),
            )

        profile = self._registry.get(bid.bidder_id)
        # Unknown bidders were already rejected by _validate_common.
        assert profile is not None
        registered_score = profile.score_for(bid.capability)
        if registered_score is None:
            return self._reject(
                submission_id=bid.bid_id,
                bidder_id=bid.bidder_id,
                code=SubmissionCode.UNREGISTERED_CAPABILITY,
                detail=(
                    f"{bid.bidder_id} is not registered for capability {bid.capability}"
                ),
            )

        if registered_score < self._announcement.minimum_capability_score:
            return self._reject(
                submission_id=bid.bid_id,
                bidder_id=bid.bidder_id,
                code=SubmissionCode.BELOW_MINIMUM_CAPABILITY,
                detail=(
                    f"registered capability {registered_score:.2f} is below task floor "
                    f"{self._announcement.minimum_capability_score:.2f}"
                ),
            )

        if bid.availability <= 0.0:
            return self._reject(
                submission_id=bid.bid_id,
                bidder_id=bid.bidder_id,
                code=SubmissionCode.UNAVAILABLE,
                detail="bidder reported no current availability",
            )

        if bid.estimated_cost > self._announcement.maximum_cost:
            return self._reject(
                submission_id=bid.bid_id,
                bidder_id=bid.bidder_id,
                code=SubmissionCode.OVER_BUDGET,
                detail=(
                    f"estimated cost {bid.estimated_cost:.2f} exceeds task maximum "
                    f"{self._announcement.maximum_cost:.2f}"
                ),
            )

        self._bids[bid.bid_id] = bid
        return self._accept(
            submission_id=bid.bid_id,
            bidder_id=bid.bidder_id,
            detail="bid admitted to the current auction round",
        )

    def submit_abstention(self, abstention: BidAbstention) -> SubmissionReceipt:
        """Validate and record one explicit abstention for the current round."""

        common_rejection = self._validate_common(
            auction_id=abstention.auction_id,
            task_id=abstention.task_id,
            auction_round=abstention.auction_round,
            bidder_id=abstention.bidder_id,
            submission_id=abstention.abstention_id,
        )
        if common_rejection is not None:
            return common_rejection

        self._abstentions[abstention.abstention_id] = abstention
        return self._accept(
            submission_id=abstention.abstention_id,
            bidder_id=abstention.bidder_id,
            detail="abstention admitted to the current auction round",
        )

    def close(self) -> AuctionRoundSummary:
        """Close bidding for this round and return its non-settlement summary."""

        self._is_open = False
        return self.summary()

    def accepted_bids(self) -> tuple[Bid, ...]:
        """Return admitted bids in deterministic identifier order."""

        return tuple(self._bids[bid_id] for bid_id in sorted(self._bids))

    def accepted_abstentions(self) -> tuple[BidAbstention, ...]:
        """Return admitted abstentions in deterministic identifier order."""

        return tuple(
            self._abstentions[abstention_id]
            for abstention_id in sorted(self._abstentions)
        )

    def receipts(self) -> tuple[SubmissionReceipt, ...]:
        """Return the complete admission audit trail in submission order."""

        return tuple(self._receipts)

    def summary(self) -> AuctionRoundSummary:
        """Return current round counts and accepted response identifiers."""

        return AuctionRoundSummary(
            auction_id=self._announcement.auction_id,
            task_id=self._announcement.task_id,
            auction_round=self._announcement.auction_round,
            is_open=self._is_open,
            accepted_bid_ids=tuple(sorted(self._bids)),
            accepted_abstention_ids=tuple(sorted(self._abstentions)),
            response_count=len(self._responded_bidder_ids),
        )

    def _validate_common(
        self,
        *,
        auction_id: str,
        task_id: str,
        auction_round: int,
        bidder_id: str,
        submission_id: str,
    ) -> SubmissionReceipt | None:
        if not self._is_open:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.AUCTION_CLOSED,
                detail="auction round is closed to new responses",
            )

        if auction_id != self._announcement.auction_id:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.AUCTION_ID_MISMATCH,
                detail="response auction_id does not match the active auction",
            )

        if task_id != self._announcement.task_id:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.TASK_ID_MISMATCH,
                detail="response task_id does not match the active task",
            )

        if auction_round != self._announcement.auction_round:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.ROUND_MISMATCH,
                detail="response auction_round does not match the active round",
            )

        if self._registry.get(bidder_id) is None:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.UNKNOWN_BIDDER,
                detail="response came from an unregistered bidder",
            )

        if submission_id in self._response_ids:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.DUPLICATE_RESPONSE_ID,
                detail="submission identifier has already been used in this round",
            )

        if bidder_id in self._responded_bidder_ids:
            return self._reject(
                submission_id=submission_id,
                bidder_id=bidder_id,
                code=SubmissionCode.DUPLICATE_BIDDER_RESPONSE,
                detail="bidder has already submitted a response in this round",
            )

        return None

    def _accept(
        self,
        *,
        submission_id: str,
        bidder_id: str,
        detail: str,
    ) -> SubmissionReceipt:
        self._response_ids.add(submission_id)
        self._responded_bidder_ids.add(bidder_id)
        receipt = SubmissionReceipt(
            accepted=True,
            code=SubmissionCode.ACCEPTED,
            detail=detail,
            submission_id=submission_id,
            bidder_id=bidder_id,
        )
        self._receipts.append(receipt)
        return receipt

    def _reject(
        self,
        *,
        submission_id: str,
        bidder_id: str,
        code: SubmissionCode,
        detail: str,
    ) -> SubmissionReceipt:
        receipt = SubmissionReceipt(
            accepted=False,
            code=code,
            detail=detail,
            submission_id=submission_id,
            bidder_id=bidder_id,
        )
        self._receipts.append(receipt)
        return receipt
