"""Distributed auction task-allocation primitives for Agent 11."""

from .models import (
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
    WorkProduct,
)

__version__ = "0.1.0"

__all__ = [
    "AbstentionReason",
    "AuctionState",
    "AuctionStatus",
    "Bid",
    "BidAbstention",
    "CapabilityProfile",
    "FailureClass",
    "RegisteredCapability",
    "TaskAnnouncement",
    "TaskAward",
    "TaskFailure",
    "WorkProduct",
    "__version__",
]
