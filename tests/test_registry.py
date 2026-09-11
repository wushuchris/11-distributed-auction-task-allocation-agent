"""Tests for Agent 11 application-owned peer registry."""

import pytest

from auction_coordination.models import CapabilityProfile, RegisteredCapability
from auction_coordination.registry import PeerRegistry, RegistryError
from auction_coordination.scenario import build_default_registry


def test_default_registry_contains_six_fictional_peers() -> None:
    registry = build_default_registry()

    assert len(registry) == 6
    assert {profile.agent_id for profile in registry.all_profiles()} == {
        "peer.atlas",
        "peer.ledger",
        "peer.sentinel",
        "peer.veritas",
        "peer.mosaic",
        "peer.quill",
    }


def test_registry_rejects_duplicate_agent_identity() -> None:
    profile = CapabilityProfile(
        agent_id="peer.atlas",
        display_name="Atlas Research",
        capabilities=(RegisteredCapability(capability="market_research", score=0.94),),
    )
    registry = PeerRegistry((profile,))

    with pytest.raises(RegistryError):
        registry.register(profile)


def test_registry_fails_closed_for_unknown_peer() -> None:
    registry = build_default_registry()

    with pytest.raises(RegistryError):
        registry.require("peer.unknown")


def test_registry_is_authoritative_for_capability_score() -> None:
    registry = build_default_registry()

    assert registry.capability_score("peer.atlas", "market_research") == pytest.approx(0.94)
    assert registry.capability_score("peer.atlas", "financial_analysis") is None


def test_market_research_has_multiple_eligible_competitors() -> None:
    registry = build_default_registry()

    eligible = registry.eligible_profiles("market_research", 0.70)

    assert [profile.agent_id for profile in eligible] == [
        "peer.atlas",
        "peer.mosaic",
        "peer.quill",
        "peer.veritas",
    ]


def test_every_mvp_task_capability_has_competition() -> None:
    registry = build_default_registry()

    for capability in (
        "market_research",
        "financial_analysis",
        "operating_risk",
        "evidence_verification",
        "executive_synthesis",
    ):
        assert len(registry.eligible_profiles(capability, 0.70)) >= 2
