"""Application-owned registry for Agent 11 peer capabilities."""

from __future__ import annotations

from collections.abc import Iterable

from .models import CapabilityProfile


class RegistryError(ValueError):
    """Raised when registry state or lookup requests are invalid."""


class PeerRegistry:
    """Authoritative registry of peer capability profiles.

    The registry owns qualification data used by later bid validation and
    settlement. Peers may reference their registered capabilities, but they do
    not get to redefine those scores inside an auction bid.
    """

    def __init__(self, profiles: Iterable[CapabilityProfile] = ()) -> None:
        self._profiles: dict[str, CapabilityProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: CapabilityProfile) -> None:
        """Register one peer profile, rejecting duplicate identities."""

        if profile.agent_id in self._profiles:
            raise RegistryError(f"duplicate agent_id: {profile.agent_id}")
        self._profiles[profile.agent_id] = profile

    def get(self, agent_id: str) -> CapabilityProfile | None:
        """Return a profile when present without changing registry state."""

        return self._profiles.get(agent_id)

    def require(self, agent_id: str) -> CapabilityProfile:
        """Return a registered profile or fail closed for an unknown peer."""

        profile = self.get(agent_id)
        if profile is None:
            raise RegistryError(f"unknown agent_id: {agent_id}")
        return profile

    def capability_score(self, agent_id: str, capability: str) -> float | None:
        """Return the application-owned capability score for one peer."""

        return self.require(agent_id).score_for(capability)

    def eligible_profiles(
        self,
        capability: str,
        minimum_score: float,
    ) -> tuple[CapabilityProfile, ...]:
        """Return registered peers meeting a capability floor in stable order."""

        eligible = []
        for profile in self._profiles.values():
            score = profile.score_for(capability)
            if score is not None and score >= minimum_score:
                eligible.append(profile)
        return tuple(sorted(eligible, key=lambda profile: profile.agent_id))

    def all_profiles(self) -> tuple[CapabilityProfile, ...]:
        """Return all registered profiles in deterministic identity order."""

        return tuple(sorted(self._profiles.values(), key=lambda profile: profile.agent_id))

    def __len__(self) -> int:
        return len(self._profiles)
