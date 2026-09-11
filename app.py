"""Gradio demo for Agent 11 — Distributed Auction Task Allocation Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auction_coordination.demo import (  # noqa: E402
    DemoMode,
    DemoSnapshot,
    llm_runtime_status,
    run_demo,
    safe_run_demo,
)


AUCTION_HEADERS = [
    "Task",
    "Capability",
    "Winning peer",
    "Winning score",
    "Bids",
    "Abstentions",
    "Rounds",
    "Messages",
    "Execution cost",
    "Status",
]
ARCHITECTURE_HEADERS = [
    "Architecture",
    "Mission success",
    "Tasks completed",
    "Messages",
    "Messages / task",
    "Execution cost",
    "Allocation efficiency",
    "Allocation authority",
]
STRESS_HEADERS = [
    "Scenario",
    "Outcome",
    "Distributed messages",
    "Centralized messages",
    "Traffic multiple",
    "Execution cost",
    "Same workers",
    "Scenario design",
]
WORK_HEADERS = [
    "Task",
    "Peer",
    "Title",
    "Summary",
    "Findings",
    "Evidence IDs",
]
EVENT_HEADERS = [
    "Seq",
    "Event",
    "Task",
    "Round",
    "Actor",
    "Accepted",
    "Detail",
]

APP_CSS = """
.gradio-container { max-width: 1560px !important; }
.hero-card {
    border: 1px solid rgba(148, 163, 184, .28);
    border-radius: 22px;
    padding: 28px 30px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(37,99,235,.16), rgba(16,185,129,.08));
}
.hero-card h1 { margin: 6px 0 10px; font-size: 2.1rem; line-height: 1.15; }
.hero-card p { font-size: 1.02rem; max-width: 1120px; margin: 0 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 700; opacity: .8; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { border: 1px solid rgba(148,163,184,.35); border-radius: 999px; padding: 6px 10px; font-size: .82rem; }
.market-grid { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 10px; margin: 10px 0 18px; }
.market-card { border: 1px solid rgba(148,163,184,.28); border-radius: 15px; padding: 14px; background: rgba(148,163,184,.05); }
.market-card strong { display:block; margin-bottom:5px; }
.result-card { border: 1px solid rgba(148,163,184,.25); border-radius: 16px; padding: 10px 14px; }
.control-note { border-left: 4px solid #2563eb; padding: 10px 14px; background: rgba(37,99,235,.07); border-radius: 8px; }
@media (max-width: 1050px) { .market-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 650px) { .market-grid { grid-template-columns: 1fr; } }
"""


def _auction_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [
        [
            row.task,
            row.capability,
            row.winner,
            row.score,
            row.bids,
            row.abstentions,
            row.rounds,
            row.messages,
            row.execution_cost,
            row.status,
        ]
        for row in snapshot.auction_rows
    ]


def _architecture_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [
        [
            row.architecture,
            row.mission_success,
            row.tasks_completed,
            row.messages,
            row.messages_per_task,
            row.execution_cost,
            row.allocation_efficiency,
            row.allocation_authority,
        ]
        for row in snapshot.architecture_rows
    ]


def _stress_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [
        [
            row.scenario,
            row.outcome,
            row.distributed_messages,
            row.centralized_messages,
            row.message_multiplier,
            row.execution_cost,
            row.same_workers,
            row.observation,
        ]
        for row in snapshot.stress_rows
    ]


def _work_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [
        [row.task, row.peer, row.title, row.summary, row.findings, row.evidence]
        for row in snapshot.work_product_rows
    ]


def _event_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [
        [
            row.sequence,
            row.event,
            row.task,
            row.round,
            row.actor,
            row.accepted,
            row.detail,
        ]
        for row in snapshot.event_rows
    ]


def _mission_status(snapshot: DemoSnapshot) -> str:
    metrics = snapshot.mission.metrics
    if metrics.mission_success:
        return (
            "### Mission completed ✅\n"
            f"**Mode:** {snapshot.mode.value.replace('_', ' ').title()}  \n"
            f"**Tasks:** {metrics.tasks_completed}/5  \n"
            f"**Protocol messages:** {metrics.total_messages}  \n"
            f"**Synthetic execution cost:** ${metrics.total_synthetic_execution_cost:,.0f}  \n"
            f"**Allocation efficiency:** {metrics.allocation_efficiency:.2f}"
        )
    return (
        "### Mission escalated safely ⚠️\n"
        f"**Mode:** {snapshot.mode.value.replace('_', ' ').title()}  \n"
        f"**Tasks completed:** {metrics.tasks_completed}/5  \n"
        "Dependent work was stopped rather than bypassing the failed boundary."
    )


def _raw_snapshot(snapshot: DemoSnapshot) -> dict[str, object]:
    metrics = snapshot.mission.metrics
    return {
        "mode": snapshot.mode.value,
        "mission_success": metrics.mission_success,
        "tasks_completed": metrics.tasks_completed,
        "total_bids": metrics.total_bids,
        "total_abstentions": metrics.total_abstentions,
        "total_messages": metrics.total_messages,
        "total_synthetic_execution_cost": metrics.total_synthetic_execution_cost,
        "allocation_efficiency": metrics.allocation_efficiency,
        "winning_peers": [row.winner for row in snapshot.auction_rows],
    }


def _outputs_for_snapshot(snapshot: DemoSnapshot):
    return (
        _mission_status(snapshot),
        snapshot.executive_summary,
        _auction_rows(snapshot),
        _architecture_rows(snapshot),
        snapshot.stress_summary,
        _stress_rows(snapshot),
        _work_rows(snapshot),
        _event_rows(snapshot),
        _raw_snapshot(snapshot),
    )


def _run_from_ui(mode_label: str):
    mode = (
        DemoMode.LLM_ASSISTED
        if mode_label == "LLM-assisted"
        else DemoMode.DETERMINISTIC
    )
    snapshot, error = safe_run_demo(mode)
    if snapshot is None:
        status = "### Mission not started\n" + (error or "Live model configuration is unavailable.")
        summary = (
            "### No business result published\n"
            "LLM-assisted mode requires deployment runtime configuration. Choose **Deterministic** "
            "to run the complete auction system without a model provider."
        )
        return status, summary, [], [], "", [], [], [], {"error": error, "mode": mode.value}
    return _outputs_for_snapshot(snapshot)


DEFAULT_SNAPSHOT = run_demo(DemoMode.DETERMINISTIC)
DEFAULT_OUTPUTS = _outputs_for_snapshot(DEFAULT_SNAPSHOT)


with gr.Blocks(
    title="Agent 11 — Distributed Auction Task Allocation",
    analytics_enabled=False,
) as demo:
    gr.HTML(
        """
<div class="hero-card">
  <div class="eyebrow">Agent 11 · Business + Engineering Demo</div>
  <h1>Who should do the work when several AI peers are qualified?</h1>
  <p>A fictional acquisition team needs market research, financial analysis, risk review, evidence verification, and executive synthesis. Six AI peers have overlapping skills, different confidence, availability, and synthetic costs. No permanent AI manager assigns them. Peers decide whether to bid; an engineered protocol determines who wins.</p>
  <div class="pill-row">
    <span class="pill">📣 Typed task auctions</span>
    <span class="pill">🤝 Local bid / abstain decisions</span>
    <span class="pill">⚖️ Deterministic settlement</span>
    <span class="pill">🔁 Bounded reauction</span>
    <span class="pill">🧾 Auditable allocation</span>
    <span class="pill">🔒 LLM outside control plane</span>
  </div>
</div>
"""
    )

    with gr.Row():
        mode = gr.Radio(
            choices=["Deterministic", "LLM-assisted"],
            value="Deterministic",
            label="Work-product execution mode",
        )
        gr.Markdown(
            "**Live LLM runtime:** " + llm_runtime_status()
            + "\n\n**Control principle:** *Peers decide whether to compete. The protocol decides who wins.*"
        )

    run_button = gr.Button("Run due-diligence marketplace", variant="primary")

    with gr.Row():
        mission_status = gr.Markdown(DEFAULT_OUTPUTS[0], elem_classes=["result-card"])
        executive_summary = gr.Markdown(DEFAULT_OUTPUTS[1], elem_classes=["result-card"])

    with gr.Tabs():
        with gr.Tab("Task Marketplace"):
            gr.Markdown(
                "## Watch five business tasks get allocated\n"
                "Each eligible peer independently decides whether to **BID** or **ABSTAIN**. "
                "Valid bids are scored using application-owned capability plus bounded confidence, "
                "availability, and cost inputs. The winner is derived deterministically."
            )
            gr.HTML(
                """
<div class="market-grid">
  <div class="market-card"><strong>1 · Market</strong>Who best fits the market-research task?</div>
  <div class="market-card"><strong>2 · Finance</strong>Who should analyze financial quality?</div>
  <div class="market-card"><strong>3 · Risk</strong>Who should own operating-risk review?</div>
  <div class="market-card"><strong>4 · Verify</strong>Who should independently verify the evidence?</div>
  <div class="market-card"><strong>5 · Synthesize</strong>Who should prepare the executive diligence synthesis?</div>
</div>
<div class="control-note"><strong>What the runtime does not do:</strong> it never manually selects Atlas, Ledger, Sentinel, Veritas, Mosaic, or Quill. It opens tasks and advances dependencies; peer policy and deterministic settlement own allocation.</div>
"""
            )
            auction_table = gr.Dataframe(
                headers=AUCTION_HEADERS,
                value=DEFAULT_OUTPUTS[2],
                interactive=False,
                label="Distributed auction results",
            )

        with gr.Tab("Centralized vs Distributed"):
            gr.Markdown(
                "## What does decentralization cost?\n"
                "The comparison uses the same peers, task economics, capability authority, utility "
                "function, executor, and bounded failure limit. Only the coordination mechanism changes."
            )
            architecture_table = gr.Dataframe(
                headers=ARCHITECTURE_HEADERS,
                value=DEFAULT_OUTPUTS[3],
                interactive=False,
                label="Controlled allocation comparison",
            )
            gr.Markdown(
                "**Interpretation:** In the clean full-information case, both approaches select the same "
                "workers at the same synthetic execution cost and utility. The distributed market pays "
                "additional message overhead in exchange for decentralized allocation authority."
            )

        with gr.Tab("Stress Evaluation"):
            stress_summary = gr.Markdown(DEFAULT_OUTPUTS[4], elem_classes=["result-card"])
            stress_table = gr.Dataframe(
                headers=STRESS_HEADERS,
                value=DEFAULT_OUTPUTS[5],
                interactive=False,
                label="Nine deterministic stress scenarios",
            )

        with gr.Tab("Work Products"):
            gr.Markdown(
                "## What the winning peers produced\n"
                "Deterministic mode uses synthetic handlers. LLM-assisted mode lets the model draft only "
                "the substantive title, summary, findings, and approved evidence references after a task "
                "has already been awarded."
            )
            work_table = gr.Dataframe(
                headers=WORK_HEADERS,
                value=DEFAULT_OUTPUTS[6],
                interactive=False,
                label="Validated work products",
            )

        with gr.Tab("Protocol Audit"):
            gr.Markdown(
                "## Allocation audit trail\n"
                "Announcements, bids, abstentions, auction closes, awards, results, failures, and "
                "escalations are recorded as protocol-level events. This is the engineering evidence "
                "underneath the business story."
            )
            event_table = gr.Dataframe(
                headers=EVENT_HEADERS,
                value=DEFAULT_OUTPUTS[7],
                interactive=False,
                label="Append-only mission events",
            )

        with gr.Tab("Engineering Boundary"):
            gr.Markdown(
                "## What the model can and cannot control\n\n"
                "**Model may draft:** title, summary, findings, and evidence references for an already-awarded task.\n\n"
                "**Application owns:** registry capability, BID/ABSTAIN admission, scoring weights, tie-breaks, "
                "winner selection, award identity, evidence allowlists, work-product lineage, reauction, escalation, "
                "and publication boundaries.\n\n"
                "This keeps probabilistic reasoning inside a deterministic execution envelope rather than "
                "letting an LLM become the hidden auctioneer."
            )
            raw_snapshot = gr.JSON(value=DEFAULT_OUTPUTS[8], label="Compact mission snapshot")

    run_button.click(
        fn=_run_from_ui,
        inputs=[mode],
        outputs=[
            mission_status,
            executive_summary,
            auction_table,
            architecture_table,
            stress_summary,
            stress_table,
            work_table,
            event_table,
            raw_snapshot,
        ],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
