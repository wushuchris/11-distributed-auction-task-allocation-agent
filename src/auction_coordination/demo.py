"""Business-first presentation layer for the Agent 11 demo.

This module translates the tested allocation runtimes into UI-friendly rows. It
contains no auction policy of its own: distributed allocation, centralized
comparison, stress evaluation, and optional LLM execution all reuse the existing
application primitives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from .baseline import CentralizedAllocationBaseline, compare_allocation_models
from .evaluation import run_stress_evaluation
from .execution import DeterministicTaskExecutor
from .hf_runtime import HuggingFaceStructuredChatClient
from .llm import ModelConfigurationError, build_llm_handlers
from .runtime import DeterministicMissionRuntime, MissionResult, TaskRunStatus
from .scenario import (
    build_default_registry,
    default_bid_settings,
    default_peer_profiles,
    default_task_blueprints,
)

DEMO_STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class DemoMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_ASSISTED = "llm_assisted"


@dataclass(frozen=True)
class AuctionRow:
    task: str
    capability: str
    winner: str
    score: str
    bids: int
    abstentions: int
    rounds: int
    messages: int
    execution_cost: str
    status: str


@dataclass(frozen=True)
class WorkProductRow:
    task: str
    peer: str
    title: str
    summary: str
    findings: str
    evidence: str


@dataclass(frozen=True)
class EventRow:
    sequence: int
    event: str
    task: str
    round: int
    actor: str
    accepted: str
    detail: str


@dataclass(frozen=True)
class ArchitectureRow:
    architecture: str
    mission_success: str
    tasks_completed: int
    messages: int
    messages_per_task: str
    execution_cost: str
    allocation_efficiency: str
    allocation_authority: str


@dataclass(frozen=True)
class StressRow:
    scenario: str
    outcome: str
    distributed_messages: int
    centralized_messages: int
    message_multiplier: str
    execution_cost: str
    same_workers: str
    observation: str


@dataclass(frozen=True)
class DemoSnapshot:
    mode: DemoMode
    mission: MissionResult
    executive_summary: str
    auction_rows: tuple[AuctionRow, ...]
    work_product_rows: tuple[WorkProductRow, ...]
    event_rows: tuple[EventRow, ...]
    architecture_rows: tuple[ArchitectureRow, ...]
    stress_rows: tuple[StressRow, ...]
    stress_summary: str


def _peer_names() -> dict[str, str]:
    return {profile.agent_id: profile.display_name for profile in default_peer_profiles()}


def _task_name(task_id: str) -> str:
    names = {
        "task.market": "Market Attractiveness",
        "task.finance": "Financial Quality",
        "task.risk": "Operating Risk",
        "task.verify": "Evidence Verification",
        "task.synthesis": "Executive Synthesis",
    }
    return names.get(task_id, task_id)


def _capability_name(capability: str) -> str:
    return capability.replace("_", " ").title()


def llm_runtime_status(env: Mapping[str, str] | None = None) -> str:
    """Describe optional live configuration without exposing credential values."""

    values = env if env is not None else os.environ
    token_ready = bool(values.get("HF_TOKEN", "").strip())
    model_id = values.get("MODEL_ID", "").strip()
    if token_ready and model_id:
        return f"configured for `{model_id}`"
    missing = []
    if not token_ready:
        missing.append("HF_TOKEN")
    if not model_id:
        missing.append("MODEL_ID")
    return "not configured (missing " + ", ".join(missing) + ")"


def _executor_for_mode(mode: DemoMode):
    registry = build_default_registry()
    if mode is DemoMode.DETERMINISTIC:
        return registry, DeterministicTaskExecutor(registry)

    model = HuggingFaceStructuredChatClient.from_env()
    return registry, DeterministicTaskExecutor(
        registry,
        handlers=build_llm_handlers(model),
    )


def _run_distributed(mode: DemoMode) -> MissionResult:
    registry, executor = _executor_for_mode(mode)
    return DeterministicMissionRuntime(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
        executor=executor,
    ).run(started_at=DEMO_STARTED_AT)


def _architecture_rows(distributed: MissionResult) -> tuple[ArchitectureRow, ...]:
    registry = build_default_registry()
    centralized = CentralizedAllocationBaseline(
        registry=registry,
        bid_settings=default_bid_settings(),
        task_blueprints=default_task_blueprints(),
    ).run(started_at=DEMO_STARTED_AT)
    comparison = compare_allocation_models(
        distributed=distributed,
        centralized=centralized,
    )

    return (
        ArchitectureRow(
            architecture="Distributed auction",
            mission_success="Yes" if distributed.metrics.mission_success else "No",
            tasks_completed=distributed.metrics.tasks_completed,
            messages=distributed.metrics.total_messages,
            messages_per_task=f"{distributed.metrics.average_messages_per_task:.1f}",
            execution_cost=f"${distributed.metrics.total_synthetic_execution_cost:,.0f}",
            allocation_efficiency=(
                f"{distributed.metrics.allocation_efficiency:.2f}"
                if distributed.metrics.allocation_efficiency is not None
                else "—"
            ),
            allocation_authority="Peers bid; shared protocol settles",
        ),
        ArchitectureRow(
            architecture="Centralized baseline",
            mission_success="Yes" if centralized.metrics.mission_success else "No",
            tasks_completed=centralized.metrics.tasks_completed,
            messages=centralized.metrics.total_messages,
            messages_per_task=f"{centralized.metrics.average_messages_per_task:.1f}",
            execution_cost=f"${centralized.metrics.total_synthetic_execution_cost:,.0f}",
            allocation_efficiency=(
                f"{centralized.metrics.allocation_efficiency:.2f}"
                if centralized.metrics.allocation_efficiency is not None
                else "—"
            ),
            allocation_authority=(
                "Central manager directly assigns"
                + (
                    " · same workers"
                    if comparison.same_completed_workers
                    else " · different workers"
                )
            ),
        ),
    )


def _stress_rows() -> tuple[tuple[StressRow, ...], str]:
    report = run_stress_evaluation(started_at=DEMO_STARTED_AT)
    rows: list[StressRow] = []
    for outcome in report.outcomes:
        comparison = outcome.comparison
        distributed_success = outcome.distributed.metrics.mission_success
        centralized_success = outcome.centralized.metrics.mission_success
        if distributed_success and centralized_success:
            result = "Both succeed"
        elif not distributed_success and not centralized_success:
            result = "Both escalate"
        else:
            result = "Architecture disagreement"

        observation = outcome.scenario.description
        rows.append(
            StressRow(
                scenario=outcome.scenario.name.replace("_", " ").title(),
                outcome=result,
                distributed_messages=comparison.distributed_messages,
                centralized_messages=comparison.centralized_messages,
                message_multiplier=(
                    f"{comparison.message_multiplier:.1f}×"
                    if comparison.message_multiplier is not None
                    else "—"
                ),
                execution_cost=f"${comparison.distributed_execution_cost:,.0f}",
                same_workers="Yes" if comparison.same_completed_workers else "No",
                observation=observation,
            )
        )

    metrics = report.metrics
    summary = (
        "### Stress evaluation summary\n"
        f"Across **{metrics.scenario_count} deterministic scenarios**, both architectures "
        f"succeeded together in **{metrics.both_succeeded}** cases and escalated together in "
        f"**{metrics.both_failed}**. There were **{metrics.success_disagreements} success "
        f"disagreements** and **{metrics.worker_disagreements} worker disagreements**.\n\n"
        f"The distributed system used **{metrics.distributed_total_messages} protocol messages** "
        f"versus **{metrics.centralized_total_messages}** for the centralized baseline "
        f"(**{metrics.message_multiplier:.1f}× coordination traffic**) while aggregate synthetic "
        f"execution cost remained **${metrics.distributed_total_execution_cost:,.0f}** for both."
    )
    return tuple(rows), summary


def _executive_summary(mission: MissionResult, mode: DemoMode) -> str:
    mode_name = "LLM-assisted work products" if mode is DemoMode.LLM_ASSISTED else "deterministic work products"
    if not mission.metrics.mission_success:
        return (
            "### Mission escalated safely\n"
            "No final diligence recommendation was published because a bounded task could not "
            "complete safely. The auction system stops dependent work rather than inventing a "
            "worker or bypassing a failed execution boundary."
        )

    synthesis = next(
        product for product in mission.work_products if product.capability == "executive_synthesis"
    )
    winner_names = _peer_names()
    allocations = ", ".join(
        f"{_task_name(result.blueprint.task_id)} → {winner_names.get(result.metrics.winning_peer or '', result.metrics.winning_peer or '—')}"
        for result in mission.task_results
    )
    return (
        "### Executive due-diligence result\n"
        f"**Decision framing:** {synthesis.summary}\n\n"
        f"**Allocation path:** {allocations}.\n\n"
        f"**Control result:** {mission.metrics.tasks_completed}/5 tasks completed using "
        f"{mode_name}; allocation efficiency was **{mission.metrics.allocation_efficiency:.2f}**, "
        f"protocol traffic was **{mission.metrics.total_messages} messages**, and synthetic "
        f"execution cost was **${mission.metrics.total_synthetic_execution_cost:,.0f}**.\n\n"
        "**Why it matters:** no semantic manager chose the workers. Peers decided whether to "
        "compete, deterministic protocol rules selected the winners, and only validated work "
        "products were allowed downstream."
    )


def run_demo(mode: DemoMode = DemoMode.DETERMINISTIC) -> DemoSnapshot:
    """Run the business demo and return UI-friendly immutable presentation data."""

    mission = _run_distributed(mode)
    names = _peer_names()
    auction_rows = tuple(
        AuctionRow(
            task=_task_name(result.blueprint.task_id),
            capability=_capability_name(result.blueprint.capability),
            winner=names.get(result.metrics.winning_peer or "", result.metrics.winning_peer or "—"),
            score=(
                f"{result.metrics.winning_score:.4f}"
                if result.metrics.winning_score is not None
                else "—"
            ),
            bids=result.metrics.bids,
            abstentions=result.metrics.abstentions,
            rounds=result.metrics.auction_rounds,
            messages=result.metrics.messages,
            execution_cost=f"${result.metrics.execution_cost:,.0f}",
            status=result.status.value.replace("_", " ").title(),
        )
        for result in mission.task_results
    )
    work_rows = tuple(
        WorkProductRow(
            task=_task_name(product.task_id),
            peer=names.get(product.agent_id, product.agent_id),
            title=product.title,
            summary=product.summary,
            findings=" • ".join(product.findings),
            evidence=", ".join(product.evidence_ids),
        )
        for product in mission.work_products
    )
    event_rows = tuple(
        EventRow(
            sequence=event.sequence,
            event=event.event_type.value.replace("_", " ").title(),
            task=_task_name(event.task_id),
            round=event.auction_round,
            actor=names.get(event.actor_id or "", event.actor_id or "—"),
            accepted=(
                "Yes" if event.accepted is True else "No" if event.accepted is False else "—"
            ),
            detail=event.detail,
        )
        for event in mission.events
    )
    stress_rows, stress_summary = _stress_rows()
    return DemoSnapshot(
        mode=mode,
        mission=mission,
        executive_summary=_executive_summary(mission, mode),
        auction_rows=auction_rows,
        work_product_rows=work_rows,
        event_rows=event_rows,
        architecture_rows=_architecture_rows(mission),
        stress_rows=stress_rows,
        stress_summary=stress_summary,
    )


def safe_run_demo(mode: DemoMode) -> tuple[DemoSnapshot | None, str | None]:
    """UI boundary that converts missing live configuration into a friendly message."""

    try:
        return run_demo(mode), None
    except ModelConfigurationError as exc:
        return None, str(exc)
