"""Synthetic due-diligence peer and task configuration for the Agent 11 demo."""

from __future__ import annotations

from dataclasses import dataclass

from .bidding import PeerBidSettings
from .models import CapabilityProfile, RegisteredCapability
from .registry import PeerRegistry


@dataclass(frozen=True)
class TaskBlueprint:
    """Deterministic task definition used by the synthetic mission runtime."""

    task_id: str
    auction_id: str
    capability: str
    summary: str
    dependency_task_ids: tuple[str, ...] = ()
    minimum_capability_score: float = 0.70
    maximum_cost: float = 100.0
    max_auction_rounds: int = 2


def default_task_blueprints() -> tuple[TaskBlueprint, ...]:
    """Return the five-task synthetic due-diligence dependency graph.

    The runtime advances this fixed task graph but never assigns a worker.
    Allocation remains owned by peer bidding plus deterministic settlement.
    """

    return (
        TaskBlueprint(
            task_id="task.market",
            auction_id="auction.market",
            capability="market_research",
            summary="Assess the fictional target market and competitive environment.",
        ),
        TaskBlueprint(
            task_id="task.finance",
            auction_id="auction.finance",
            capability="financial_analysis",
            summary="Assess synthetic financial quality, cash conversion, and margin durability.",
        ),
        TaskBlueprint(
            task_id="task.risk",
            auction_id="auction.risk",
            capability="operating_risk",
            summary="Assess synthetic operating, supplier, integration, and key-person risks.",
        ),
        TaskBlueprint(
            task_id="task.verify",
            auction_id="auction.verify",
            capability="evidence_verification",
            summary="Verify the synthetic market, financial, and operating-risk evidence.",
            dependency_task_ids=("task.market", "task.finance", "task.risk"),
        ),
        TaskBlueprint(
            task_id="task.synthesis",
            auction_id="auction.synthesis",
            capability="executive_synthesis",
            summary="Synthesize validated diligence into a bounded next-stage recommendation.",
            dependency_task_ids=(
                "task.market",
                "task.finance",
                "task.risk",
                "task.verify",
            ),
        ),
    )


def default_peer_profiles() -> tuple[CapabilityProfile, ...]:
    """Return six fictional peers with intentionally overlapping capabilities."""

    return (
        CapabilityProfile(
            agent_id="peer.atlas",
            display_name="Atlas Research",
            capabilities=(
                RegisteredCapability(capability="market_research", score=0.94),
                RegisteredCapability(capability="evidence_verification", score=0.74),
            ),
        ),
        CapabilityProfile(
            agent_id="peer.ledger",
            display_name="Ledger Analyst",
            capabilities=(
                RegisteredCapability(capability="financial_analysis", score=0.96),
                RegisteredCapability(capability="operating_risk", score=0.72),
                RegisteredCapability(capability="executive_synthesis", score=0.68),
            ),
        ),
        CapabilityProfile(
            agent_id="peer.sentinel",
            display_name="Sentinel Risk",
            capabilities=(
                RegisteredCapability(capability="operating_risk", score=0.95),
                RegisteredCapability(capability="evidence_verification", score=0.80),
                RegisteredCapability(capability="financial_analysis", score=0.75),
            ),
        ),
        CapabilityProfile(
            agent_id="peer.veritas",
            display_name="Veritas Evidence",
            capabilities=(
                RegisteredCapability(capability="evidence_verification", score=0.97),
                RegisteredCapability(capability="market_research", score=0.78),
                RegisteredCapability(capability="operating_risk", score=0.76),
            ),
        ),
        CapabilityProfile(
            agent_id="peer.mosaic",
            display_name="Mosaic Generalist",
            capabilities=(
                RegisteredCapability(capability="market_research", score=0.82),
                RegisteredCapability(capability="financial_analysis", score=0.82),
                RegisteredCapability(capability="operating_risk", score=0.84),
                RegisteredCapability(capability="evidence_verification", score=0.80),
                RegisteredCapability(capability="executive_synthesis", score=0.84),
            ),
        ),
        CapabilityProfile(
            agent_id="peer.quill",
            display_name="Quill Synthesis",
            capabilities=(
                RegisteredCapability(capability="executive_synthesis", score=0.96),
                RegisteredCapability(capability="market_research", score=0.72),
                RegisteredCapability(capability="financial_analysis", score=0.74),
            ),
        ),
    )


def build_default_registry() -> PeerRegistry:
    """Build the application-owned registry for the synthetic demo."""

    return PeerRegistry(default_peer_profiles())


def default_bid_settings() -> dict[str, PeerBidSettings]:
    """Return deterministic local economics for the six fictional peers.

    These values are synthetic demo inputs. They are not capability authority;
    qualification remains in the registry.
    """

    return {
        "peer.atlas": PeerBidSettings(
            agent_id="peer.atlas",
            availability=0.75,
            confidence_by_capability={
                "market_research": 0.91,
                "evidence_verification": 0.77,
            },
            cost_by_capability={
                "market_research": 62.0,
                "evidence_verification": 55.0,
            },
        ),
        "peer.ledger": PeerBidSettings(
            agent_id="peer.ledger",
            availability=0.70,
            confidence_by_capability={
                "financial_analysis": 0.93,
                "operating_risk": 0.75,
                "executive_synthesis": 0.66,
            },
            cost_by_capability={
                "financial_analysis": 70.0,
                "operating_risk": 65.0,
                "executive_synthesis": 74.0,
            },
        ),
        "peer.sentinel": PeerBidSettings(
            agent_id="peer.sentinel",
            availability=0.65,
            confidence_by_capability={
                "operating_risk": 0.92,
                "evidence_verification": 0.83,
                "financial_analysis": 0.78,
            },
            cost_by_capability={
                "operating_risk": 68.0,
                "evidence_verification": 63.0,
                "financial_analysis": 75.0,
            },
        ),
        "peer.veritas": PeerBidSettings(
            agent_id="peer.veritas",
            availability=0.85,
            confidence_by_capability={
                "evidence_verification": 0.95,
                "market_research": 0.81,
                "operating_risk": 0.79,
            },
            cost_by_capability={
                "evidence_verification": 72.0,
                "market_research": 66.0,
                "operating_risk": 70.0,
            },
        ),
        "peer.mosaic": PeerBidSettings(
            agent_id="peer.mosaic",
            availability=0.55,
            confidence_by_capability={
                "market_research": 0.82,
                "financial_analysis": 0.84,
                "operating_risk": 0.85,
                "evidence_verification": 0.82,
                "executive_synthesis": 0.86,
            },
            cost_by_capability={
                "market_research": 88.0,
                "financial_analysis": 92.0,
                "operating_risk": 90.0,
                "evidence_verification": 85.0,
                "executive_synthesis": 95.0,
            },
        ),
        "peer.quill": PeerBidSettings(
            agent_id="peer.quill",
            availability=0.80,
            confidence_by_capability={
                "executive_synthesis": 0.94,
                "market_research": 0.76,
                "financial_analysis": 0.76,
            },
            cost_by_capability={
                "executive_synthesis": 75.0,
                "market_research": 65.0,
                "financial_analysis": 70.0,
            },
        ),
    }
