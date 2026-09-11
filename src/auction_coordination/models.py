"""Typed contracts for Agent 11 distributed task auctions.

These models define the data boundary only. They deliberately do not decide
who wins an auction or whether a task should be reauctioned. Those policies
belong in later deterministic application layers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
Capability = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
DisplayName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
LongText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveCost = Annotated[float, Field(gt=0.0, le=1_000_000.0)]
AuctionRound = Annotated[int, Field(ge=1, le=100)]
MaxAuctionRounds = Annotated[int, Field(ge=1, le=10)]


def _timezone_aware(value: datetime) -> datetime:
    """Reject ambiguous timestamps so audit events are globally comparable."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    return value


class StrictModel(BaseModel):
    """Base contract that rejects unknown fields and trims string input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AuctionStatus(str, Enum):
    """Allowed lifecycle states for one task auction."""

    ANNOUNCED = "announced"
    BIDDING = "bidding"
    AWARDED = "awarded"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NO_VALID_BIDS = "no_valid_bids"
    FAILED = "failed"
    REAUCTIONING = "reauctioning"
    ESCALATED = "escalated"


class AbstentionReason(str, Enum):
    """Auditable reasons a peer may decline to compete for a task."""

    CAPABILITY_MISMATCH = "capability_mismatch"
    UNAVAILABLE = "unavailable"
    OVER_BUDGET = "over_budget"
    LOW_CONFIDENCE = "low_confidence"
    LOCAL_POLICY = "local_policy"


class FailureClass(str, Enum):
    """Execution failure categories recorded after a task has been awarded."""

    EXECUTION_ERROR = "execution_error"
    INVALID_OUTPUT = "invalid_output"
    UNAVAILABLE = "unavailable"
    DEPENDENCY_FAILURE = "dependency_failure"
    POLICY_REJECTION = "policy_rejection"


class RegisteredCapability(StrictModel):
    """One application-owned capability rating for a registered peer."""

    capability: Capability
    score: UnitInterval


class CapabilityProfile(StrictModel):
    """Authoritative registered capabilities for one peer.

    Capability scores live here rather than inside bids so a bidder cannot
    promote a self-asserted qualification into system truth.
    """

    agent_id: Identifier
    display_name: DisplayName
    capabilities: tuple[RegisteredCapability, ...] = Field(min_length=1)

    @field_validator("capabilities")
    @classmethod
    def capability_names_must_be_unique(
        cls, value: tuple[RegisteredCapability, ...]
    ) -> tuple[RegisteredCapability, ...]:
        names = [item.capability for item in value]
        if len(names) != len(set(names)):
            raise ValueError("capabilities must not contain duplicate capability names")
        return value

    def score_for(self, capability: str) -> float | None:
        """Return the registered score for a capability, if present."""

        for item in self.capabilities:
            if item.capability == capability:
                return item.score
        return None


class TaskAnnouncement(StrictModel):
    """A typed invitation for eligible peers to compete for one task."""

    task_id: Identifier
    auction_id: Identifier
    auction_round: AuctionRound = 1
    requested_capability: Capability
    summary: ShortText
    minimum_capability_score: UnitInterval
    maximum_cost: PositiveCost
    required_input_ids: tuple[Identifier, ...] = ()
    max_auction_rounds: MaxAuctionRounds = 2
    opened_at: datetime

    @field_validator("required_input_ids")
    @classmethod
    def required_inputs_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("required_input_ids must not contain duplicates")
        return value

    @field_validator("opened_at")
    @classmethod
    def opened_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)

    @model_validator(mode="after")
    def round_must_not_exceed_configured_limit(self) -> "TaskAnnouncement":
        if self.auction_round > self.max_auction_rounds:
            raise ValueError("auction_round cannot exceed max_auction_rounds")
        return self


class Bid(StrictModel):
    """A bounded task-specific offer submitted by one peer.

    The bid deliberately does not contain a capability score. The settlement
    layer must obtain that value from the registered CapabilityProfile.
    """

    bid_id: Identifier
    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    bidder_id: Identifier
    capability: Capability
    confidence: UnitInterval
    availability: UnitInterval
    estimated_cost: PositiveCost
    submitted_at: datetime

    @field_validator("submitted_at")
    @classmethod
    def submitted_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class BidAbstention(StrictModel):
    """An explicit, auditable decision by a peer not to compete."""

    abstention_id: Identifier
    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    bidder_id: Identifier
    reason: AbstentionReason
    detail: ShortText
    submitted_at: datetime

    @field_validator("submitted_at")
    @classmethod
    def submitted_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class AuctionState(StrictModel):
    """Current auditable state for one task auction round."""

    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    status: AuctionStatus
    valid_bid_ids: tuple[Identifier, ...] = ()
    winning_bid_id: Identifier | None = None
    awarded_agent_id: Identifier | None = None
    failure_id: Identifier | None = None

    @field_validator("valid_bid_ids")
    @classmethod
    def valid_bid_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("valid_bid_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def state_fields_must_match_lifecycle(self) -> "AuctionState":
        has_bid = self.winning_bid_id is not None
        has_agent = self.awarded_agent_id is not None
        if has_bid != has_agent:
            raise ValueError(
                "winning_bid_id and awarded_agent_id must either both be set or both be absent"
            )

        if self.winning_bid_id is not None and self.winning_bid_id not in self.valid_bid_ids:
            raise ValueError("winning_bid_id must reference one of valid_bid_ids")

        award_required = {
            AuctionStatus.AWARDED,
            AuctionStatus.IN_PROGRESS,
            AuctionStatus.COMPLETED,
            AuctionStatus.FAILED,
            AuctionStatus.REAUCTIONING,
        }
        if self.status in award_required and not has_bid:
            raise ValueError(f"status {self.status.value} requires an awarded bid and agent")

        if self.status in {AuctionStatus.FAILED, AuctionStatus.REAUCTIONING}:
            if self.failure_id is None:
                raise ValueError(f"status {self.status.value} requires failure_id")

        if self.status is AuctionStatus.NO_VALID_BIDS and self.valid_bid_ids:
            raise ValueError("no_valid_bids state cannot contain valid_bid_ids")

        return self


class TaskAward(StrictModel):
    """Immutable-looking record of deterministic auction settlement."""

    award_id: Identifier
    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    winning_bid_id: Identifier
    awarded_agent_id: Identifier
    winning_score: UnitInterval
    awarded_at: datetime

    @field_validator("awarded_at")
    @classmethod
    def awarded_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class TaskFailure(StrictModel):
    """Explicit execution failure associated with an existing task award."""

    failure_id: Identifier
    award_id: Identifier
    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    agent_id: Identifier
    failure_class: FailureClass
    detail: LongText
    failed_at: datetime

    @field_validator("failed_at")
    @classmethod
    def failed_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class WorkProduct(StrictModel):
    """Validated task output produced by the awarded peer.

    Execution lineage is application-owned. Handlers provide substantive
    content only; they do not choose their own task, award, worker identity, or
    capability.
    """

    work_product_id: Identifier
    award_id: Identifier
    auction_id: Identifier
    task_id: Identifier
    auction_round: AuctionRound
    agent_id: Identifier
    capability: Capability
    title: ShortText
    summary: LongText
    findings: tuple[ShortText, ...] = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)
    completed_at: datetime

    @field_validator("findings")
    @classmethod
    def findings_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("findings must not contain duplicates")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must not contain duplicates")
        return value

    @field_validator("completed_at")
    @classmethod
    def completed_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _timezone_aware(value)
