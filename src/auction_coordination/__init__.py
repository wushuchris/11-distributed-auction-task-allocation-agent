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
from .baseline import (
    AllocationComparison,
    CentralEvent,
    CentralEventType,
    CentralMissionMetrics,
    CentralMissionResult,
    CentralTaskMetrics,
    CentralTaskResult,
    CentralizedAllocationBaseline,
    compare_allocation_models,
)

__version__ = "0.1.0"

__all__ = [
    "AbstentionReason",
    "AllocationComparison",
    "AuctionState",
    "AuctionStatus",
    "Bid",
    "BidAbstention",
    "CapabilityProfile",
    "CentralEvent",
    "CentralEventType",
    "CentralMissionMetrics",
    "CentralMissionResult",
    "CentralTaskMetrics",
    "CentralTaskResult",
    "CentralizedAllocationBaseline",
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
    "compare_allocation_models",
    "__version__",
]
