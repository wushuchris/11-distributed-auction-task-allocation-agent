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
from .runtime import (
    DeterministicMissionRuntime,
    MissionEvent,
    MissionEventType,
    MissionMetrics,
    MissionResult,
    TaskMetrics,
    TaskRunResult,
    TaskRunStatus,
)
from .scenario import TaskBlueprint

__version__ = "0.1.0"

__all__ = [
    "AbstentionReason",
    "AuctionState",
    "AuctionStatus",
    "Bid",
    "BidAbstention",
    "CapabilityProfile",
    "DeterministicMissionRuntime",
    "FailureClass",
    "MissionEvent",
    "MissionEventType",
    "MissionMetrics",
    "MissionResult",
    "RegisteredCapability",
    "TaskAnnouncement",
    "TaskAward",
    "TaskBlueprint",
    "TaskFailure",
    "TaskMetrics",
    "TaskRunResult",
    "TaskRunStatus",
    "WorkProduct",
    "__version__",
]
