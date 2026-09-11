"""Deterministic task execution for awarded Agent 11 work.

The auction decides who receives work. This module only validates award/task
lineage, resolves required upstream inputs, invokes an approved deterministic
handler, and validates the resulting typed WorkProduct.

Handlers own substantive synthetic content only. They cannot choose their own
award, task, worker identity, capability, or work-product identifier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from pydantic import ValidationError

from .models import FailureClass, TaskAnnouncement, TaskAward, WorkProduct
from .registry import PeerRegistry


class ExecutionError(ValueError):
    """Bounded execution failure suitable for conversion into TaskFailure."""

    def __init__(self, message: str, *, failure_class: FailureClass) -> None:
        super().__init__(message)
        self.failure_class = failure_class


@dataclass(frozen=True)
class HandlerOutput:
    """Untrusted substantive payload returned by an execution handler.

    The executor converts this payload into the strict WorkProduct contract.
    Keeping handler output separate lets validation catch malformed handler
    behavior before a work product is admitted downstream.
    """

    title: str
    summary: str
    findings: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionContext:
    """Read-only context supplied to one approved deterministic handler."""

    announcement: TaskAnnouncement
    award: TaskAward
    required_inputs: tuple[WorkProduct, ...]


TaskHandler = Callable[[ExecutionContext], HandlerOutput]


class DeterministicTaskExecutor:
    """Execute an awarded task through an allowlisted deterministic handler."""

    def __init__(
        self,
        registry: PeerRegistry,
        *,
        handlers: Mapping[str, TaskHandler] | None = None,
    ) -> None:
        self._registry = registry
        self._handlers = dict(default_handlers() if handlers is None else handlers)

    def execute(
        self,
        *,
        announcement: TaskAnnouncement,
        award: TaskAward,
        completed_at: datetime,
        available_inputs: tuple[WorkProduct, ...] = (),
    ) -> WorkProduct:
        """Validate lineage, run one handler, and return a strict work product."""

        self._validate_award(announcement=announcement, award=award)
        required_inputs = self._resolve_required_inputs(
            announcement=announcement,
            available_inputs=available_inputs,
        )

        handler = self._handlers.get(announcement.requested_capability)
        if handler is None:
            raise ExecutionError(
                f"no approved handler for capability {announcement.requested_capability}",
                failure_class=FailureClass.POLICY_REJECTION,
            )

        context = ExecutionContext(
            announcement=announcement,
            award=award,
            required_inputs=required_inputs,
        )
        try:
            output = handler(context)
        except ExecutionError:
            raise
        except Exception as exc:  # deterministic containment boundary
            raise ExecutionError(
                f"handler raised {type(exc).__name__}: {exc}",
                failure_class=FailureClass.EXECUTION_ERROR,
            ) from exc

        if not isinstance(output, HandlerOutput):
            raise ExecutionError(
                "handler returned an unsupported output type",
                failure_class=FailureClass.INVALID_OUTPUT,
            )

        try:
            return WorkProduct(
                work_product_id=self._work_product_id(announcement, award),
                award_id=award.award_id,
                auction_id=announcement.auction_id,
                task_id=announcement.task_id,
                auction_round=announcement.auction_round,
                agent_id=award.awarded_agent_id,
                capability=announcement.requested_capability,
                title=output.title,
                summary=output.summary,
                findings=output.findings,
                evidence_ids=output.evidence_ids,
                completed_at=completed_at,
            )
        except ValidationError as exc:
            raise ExecutionError(
                f"handler output failed WorkProduct validation: {exc.errors()[0]['msg']}",
                failure_class=FailureClass.INVALID_OUTPUT,
            ) from exc

    def _validate_award(
        self,
        *,
        announcement: TaskAnnouncement,
        award: TaskAward,
    ) -> None:
        if award.auction_id != announcement.auction_id:
            raise ExecutionError(
                "award auction_id does not match announcement",
                failure_class=FailureClass.POLICY_REJECTION,
            )
        if award.task_id != announcement.task_id:
            raise ExecutionError(
                "award task_id does not match announcement",
                failure_class=FailureClass.POLICY_REJECTION,
            )
        if award.auction_round != announcement.auction_round:
            raise ExecutionError(
                "award auction_round does not match announcement",
                failure_class=FailureClass.POLICY_REJECTION,
            )

        profile = self._registry.get(award.awarded_agent_id)
        if profile is None:
            raise ExecutionError(
                "awarded agent is not registered",
                failure_class=FailureClass.POLICY_REJECTION,
            )
        capability_score = profile.score_for(announcement.requested_capability)
        if capability_score is None:
            raise ExecutionError(
                "awarded agent is not registered for the requested capability",
                failure_class=FailureClass.POLICY_REJECTION,
            )
        if capability_score < announcement.minimum_capability_score:
            raise ExecutionError(
                "awarded agent capability is below the task minimum",
                failure_class=FailureClass.POLICY_REJECTION,
            )

    @staticmethod
    def _resolve_required_inputs(
        *,
        announcement: TaskAnnouncement,
        available_inputs: tuple[WorkProduct, ...],
    ) -> tuple[WorkProduct, ...]:
        by_id: dict[str, WorkProduct] = {}
        for product in available_inputs:
            if product.work_product_id in by_id:
                raise ExecutionError(
                    f"duplicate available input {product.work_product_id}",
                    failure_class=FailureClass.DEPENDENCY_FAILURE,
                )
            by_id[product.work_product_id] = product

        missing = [
            input_id
            for input_id in announcement.required_input_ids
            if input_id not in by_id
        ]
        if missing:
            raise ExecutionError(
                f"missing required input work products: {', '.join(missing)}",
                failure_class=FailureClass.DEPENDENCY_FAILURE,
            )

        return tuple(by_id[input_id] for input_id in announcement.required_input_ids)

    @staticmethod
    def _work_product_id(
        announcement: TaskAnnouncement,
        award: TaskAward,
    ) -> str:
        material = (
            f"{announcement.auction_id}|{announcement.task_id}|"
            f"{announcement.auction_round}|{award.awarded_agent_id}"
        )
        digest = sha256(material.encode("utf-8")).hexdigest()[:20]
        return f"work:{digest}"


def default_handlers() -> dict[str, TaskHandler]:
    """Return the five deterministic synthetic due-diligence handlers."""

    return {
        "market_research": _market_research_handler,
        "financial_analysis": _financial_analysis_handler,
        "operating_risk": _operating_risk_handler,
        "evidence_verification": _evidence_verification_handler,
        "executive_synthesis": _executive_synthesis_handler,
    }


def _market_research_handler(context: ExecutionContext) -> HandlerOutput:
    return HandlerOutput(
        title="Synthetic Market Attractiveness Review",
        summary=(
            "Meridian Industrial Systems operates in a fictional niche with stable end-market "
            "demand, fragmented competition, and moderate customer switching friction."
        ),
        findings=(
            "Fictional end-market demand is stable rather than hyper-growth dependent.",
            "Competition is fragmented enough to support differentiated positioning.",
            "Customer switching friction appears moderate and supports recurring relationships.",
        ),
        evidence_ids=("evidence.market.1", "evidence.market.2", "evidence.market.3"),
    )


def _financial_analysis_handler(context: ExecutionContext) -> HandlerOutput:
    return HandlerOutput(
        title="Synthetic Financial Quality Review",
        summary=(
            "The fictional target shows recurring revenue characteristics, positive cash "
            "conversion, and margins that are healthy but sensitive to integration execution."
        ),
        findings=(
            "Synthetic recurring revenue quality is stronger than one-time project revenue.",
            "Synthetic free-cash conversion remains positive across the reviewed period.",
            "Margin durability depends on maintaining procurement and labor discipline.",
        ),
        evidence_ids=("evidence.finance.1", "evidence.finance.2", "evidence.finance.3"),
    )


def _operating_risk_handler(context: ExecutionContext) -> HandlerOutput:
    return HandlerOutput(
        title="Synthetic Operating Risk Review",
        summary=(
            "The principal fictional execution risks are supplier concentration, integration "
            "complexity, and dependence on a small group of experienced operators."
        ),
        findings=(
            "Supplier concentration creates a bounded but material continuity risk.",
            "Integration complexity could temporarily pressure service quality and margins.",
            "Key-person dependence should be reduced before aggressive post-close scaling.",
        ),
        evidence_ids=("evidence.risk.1", "evidence.risk.2", "evidence.risk.3"),
    )


def _evidence_verification_handler(context: ExecutionContext) -> HandlerOutput:
    return HandlerOutput(
        title="Synthetic Evidence Verification Review",
        summary=(
            "The synthetic diligence evidence is internally consistent enough for continued "
            "review, while several claims still warrant explicit confirmation before a final bid."
        ),
        findings=(
            "Key synthetic claims have at least one corroborating evidence record.",
            "No material contradiction is present in the synthetic evidence set.",
            "Open verification items should remain visible rather than being silently resolved.",
        ),
        evidence_ids=("evidence.verify.1", "evidence.verify.2", "evidence.verify.3"),
    )


def _executive_synthesis_handler(context: ExecutionContext) -> HandlerOutput:
    upstream_count = len(context.required_inputs)
    upstream_titles = tuple(product.title for product in context.required_inputs)
    integrated_note = (
        f"Integrated {upstream_count} validated upstream work products."
        if upstream_count
        else "No upstream work products were required for this synthetic execution."
    )
    title_note = (
        f" Reviewed inputs: {', '.join(upstream_titles)}."
        if upstream_titles
        else ""
    )
    return HandlerOutput(
        title="Synthetic Executive Due-Diligence Synthesis",
        summary=(
            "The fictional acquisition case supports continued diligence rather than an "
            f"unconditional proceed decision. {integrated_note}{title_note}"
        ),
        findings=(
            "Market and financial characteristics support continued diligence.",
            "Operating and evidence risks remain decision-relevant and should stay explicit.",
            "The synthetic recommendation is to proceed only to the next diligence stage.",
        ),
        evidence_ids=("evidence.synthesis.1", "evidence.synthesis.2"),
    )
