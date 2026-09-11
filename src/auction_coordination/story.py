"""Business-readable storytelling helpers for the Agent 11 demo.

This module does not introduce auction policy. It replays the same registered
profiles, local BID/ABSTAIN policy, and deterministic settlement score function
used by the runtime so the UI can explain the marketplace to a non-technical
visitor without inventing a second decision system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .bidding import BidAction, LocalBidPolicy
from .runtime import MissionResult
from .scenario import build_default_registry, default_bid_settings, default_peer_profiles
from .settlement import BidScorecard, DeterministicSettlement


_TASK_NAMES = {
    "task.market": "Market Attractiveness Research",
    "task.finance": "Financial Quality Analysis",
    "task.risk": "Operating & Execution Risk Review",
    "task.verify": "Evidence Verification",
    "task.synthesis": "Executive Due-Diligence Synthesis",
}

_TASK_QUESTIONS = {
    "task.market": "Is the target operating in an attractive market with durable demand?",
    "task.finance": "Does the target show enough financial quality to justify deeper diligence?",
    "task.risk": "What operating or execution risks could impair the acquisition thesis?",
    "task.verify": "Can the evidence behind the research, finance, and risk work be trusted?",
    "task.synthesis": "Given the validated work, should the buyer advance to the next diligence stage?",
}

_ROLE_COPY = {
    "peer.atlas": (
        "Market Research Specialist",
        "Best suited to market mapping and source-heavy commercial research.",
    ),
    "peer.ledger": (
        "Financial Analysis Specialist",
        "Best suited to financial quality, economics, and quantitative diligence.",
    ),
    "peer.sentinel": (
        "Operating Risk Specialist",
        "Best suited to identifying execution, integration, and operating risks.",
    ),
    "peer.veritas": (
        "Evidence Verification Specialist",
        "Best suited to independent source checking and evidence quality review.",
    ),
    "peer.mosaic": (
        "Cross-Functional Generalist",
        "Can compete across most tasks, but broader coverage comes at a higher synthetic cost.",
    ),
    "peer.quill": (
        "Executive Synthesis Specialist",
        "Best suited to turning validated specialist work into an executive recommendation.",
    ),
}


@dataclass(frozen=True)
class PeerStoryCard:
    agent_id: str
    name: str
    role: str
    business_value: str
    strengths: str
    availability: str


@dataclass(frozen=True)
class BidStoryRow:
    task: str
    peer: str
    decision: str
    capability: str
    confidence: str
    availability: str
    estimated_cost: str
    auction_score: str
    outcome: str
    reason: str


@dataclass(frozen=True)
class AuctionStoryCard:
    task: str
    business_question: str
    winner: str
    runner_up: str
    winning_score: str
    margin: str
    bids: int
    abstentions: int
    why_winner: str


def _pretty_capability(value: str) -> str:
    return value.replace("_", " ").title()


def build_peer_story_cards() -> tuple[PeerStoryCard, ...]:
    """Return six business-readable cards derived from authoritative profiles."""

    settings = default_bid_settings()
    cards: list[PeerStoryCard] = []
    for profile in default_peer_profiles():
        role, business_value = _ROLE_COPY[profile.agent_id]
        ranked = sorted(profile.capabilities, key=lambda item: item.score, reverse=True)
        strengths = " · ".join(
            f"{_pretty_capability(item.capability)} {item.score:.0%}"
            for item in ranked[:3]
        )
        cards.append(
            PeerStoryCard(
                agent_id=profile.agent_id,
                name=profile.display_name,
                role=role,
                business_value=business_value,
                strengths=strengths,
                availability=f"{settings[profile.agent_id].availability:.0%}",
            )
        )
    return tuple(cards)


def _why_winner(
    *,
    winner_name: str,
    winner: BidScorecard,
    runner_name: str | None,
    runner: BidScorecard | None,
) -> str:
    base = (
        f"{winner_name} won with a weighted score of {winner.total_score:.4f}: "
        f"{winner.capability_score:.0%} registered capability, "
        f"{winner.confidence:.0%} confidence, {winner.availability:.0%} availability, "
        f"and ${winner.estimated_cost:,.0f} synthetic task cost."
    )
    if runner is None or runner_name is None:
        return base + " It was the only admitted bid."
    return (
        base
        + f" The next-best bid was {runner_name} at {runner.total_score:.4f}, "
        f"so the deterministic margin was {winner.total_score - runner.total_score:.4f}."
    )


def build_auction_story(
    mission: MissionResult,
) -> tuple[tuple[AuctionStoryCard, ...], tuple[BidStoryRow, ...]]:
    """Explain the final auction round for every attempted task.

    Presentation decisions are derived by replaying the same application-owned
    local bidding and deterministic settlement primitives used by the runtime.
    A peer that explicitly failed in an earlier round is shown as excluded from
    the immediate reauction, matching the bounded recovery protocol.
    """

    registry = build_default_registry()
    settings = default_bid_settings()
    policy = LocalBidPolicy(registry)
    settlement = DeterministicSettlement(registry)
    names = {profile.agent_id: profile.display_name for profile in default_peer_profiles()}

    story_cards: list[AuctionStoryCard] = []
    bid_rows: list[BidStoryRow] = []

    for result in mission.task_results:
        announcement = result.final_announcement
        excluded = {
            failure.agent_id
            for failure in result.failures
            if failure.auction_round < announcement.auction_round
        }
        scorecards: list[BidScorecard] = []
        decisions: dict[str, tuple[str, str, object | None]] = {}

        for index, agent_id in enumerate(sorted(settings)):
            if agent_id in excluded:
                decisions[agent_id] = (
                    "ABSTAIN",
                    "Excluded from the immediate reauction after explicit prior-round failure.",
                    None,
                )
                continue

            decision = policy.evaluate(
                announcement=announcement,
                settings=settings[agent_id],
                submitted_at=announcement.opened_at + timedelta(milliseconds=index + 1),
            )
            if decision.action is BidAction.BID:
                assert decision.bid is not None
                scorecard = settlement.score_bid(
                    bid=decision.bid,
                    announcement=announcement,
                )
                scorecards.append(scorecard)
                decisions[agent_id] = ("BID", decision.reason, scorecard)
            else:
                decisions[agent_id] = ("ABSTAIN", decision.reason, None)

        ranked = settlement.rank_scorecards(tuple(scorecards)) if scorecards else ()
        rank_by_agent = {card.bidder_id: rank for rank, card in enumerate(ranked, start=1)}
        winning_peer = result.metrics.winning_peer

        for agent_id in sorted(settings):
            action, reason, scorecard_obj = decisions[agent_id]
            scorecard = scorecard_obj if isinstance(scorecard_obj, BidScorecard) else None
            capability = registry.capability_score(agent_id, announcement.requested_capability)
            confidence = settings[agent_id].confidence_for(announcement.requested_capability)
            cost = settings[agent_id].cost_for(announcement.requested_capability)
            if action == "BID" and scorecard is not None:
                outcome = "Winner" if agent_id == winning_peer else f"Bid · rank {rank_by_agent[agent_id]}"
            else:
                outcome = "Did not compete"

            bid_rows.append(
                BidStoryRow(
                    task=_TASK_NAMES.get(announcement.task_id, announcement.task_id),
                    peer=names[agent_id],
                    decision=action,
                    capability=f"{capability:.0%}" if capability is not None else "—",
                    confidence=f"{confidence:.0%}" if confidence is not None else "—",
                    availability=f"{settings[agent_id].availability:.0%}",
                    estimated_cost=f"${cost:,.0f}" if cost is not None else "—",
                    auction_score=f"{scorecard.total_score:.4f}" if scorecard is not None else "—",
                    outcome=outcome,
                    reason=reason,
                )
            )

        if ranked and winning_peer is not None:
            winner = next(card for card in ranked if card.bidder_id == winning_peer)
            runner = next((card for card in ranked if card.bidder_id != winning_peer), None)
            winner_name = names[winning_peer]
            runner_name = names[runner.bidder_id] if runner is not None else None
            story_cards.append(
                AuctionStoryCard(
                    task=_TASK_NAMES.get(announcement.task_id, announcement.task_id),
                    business_question=_TASK_QUESTIONS.get(announcement.task_id, announcement.summary),
                    winner=winner_name,
                    runner_up=runner_name or "—",
                    winning_score=f"{winner.total_score:.4f}",
                    margin=(
                        f"{winner.total_score - runner.total_score:.4f}"
                        if runner is not None
                        else "—"
                    ),
                    bids=len(ranked),
                    abstentions=sum(action == "ABSTAIN" for action, _, _ in decisions.values()),
                    why_winner=_why_winner(
                        winner_name=winner_name,
                        winner=winner,
                        runner_name=runner_name,
                        runner=runner,
                    ),
                )
            )
        else:
            story_cards.append(
                AuctionStoryCard(
                    task=_TASK_NAMES.get(announcement.task_id, announcement.task_id),
                    business_question=_TASK_QUESTIONS.get(announcement.task_id, announcement.summary),
                    winner="No valid winner",
                    runner_up="—",
                    winning_score="—",
                    margin="—",
                    bids=0,
                    abstentions=sum(action == "ABSTAIN" for action, _, _ in decisions.values()),
                    why_winner="No valid bids remained, so the task escalated instead of being assigned unsafely.",
                )
            )

    return tuple(story_cards), tuple(bid_rows)
