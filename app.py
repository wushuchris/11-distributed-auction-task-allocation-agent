"""Gradio demo for Agent 11 — Distributed Auction Task Allocation Agent."""

from __future__ import annotations

import html
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
from auction_coordination.story import (  # noqa: E402
    build_auction_story,
    build_peer_story_cards,
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
BID_HEADERS = [
    "Task",
    "Peer",
    "Decision",
    "Capability",
    "Confidence",
    "Availability",
    "Est. cost",
    "Auction score",
    "Outcome",
    "Why",
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
    padding: 30px 32px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(37,99,235,.17), rgba(16,185,129,.08));
}
.hero-card h1 { margin: 6px 0 10px; font-size: 2.15rem; line-height: 1.15; }
.hero-card p { font-size: 1.04rem; max-width: 1160px; margin: 0 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 700; opacity: .8; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.pill { border: 1px solid rgba(148,163,184,.35); border-radius: 999px; padding: 6px 10px; font-size: .82rem; }
.section-title { margin: 26px 0 6px; }
.section-kicker { text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; font-weight: 700; opacity: .72; margin-bottom: 4px; }
.business-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 12px 0 20px; }
.business-card, .peer-card, .step-card, .flow-card {
    border: 1px solid rgba(148,163,184,.28);
    border-radius: 16px;
    padding: 16px;
    background: rgba(148,163,184,.045);
}
.business-card strong, .peer-card strong, .step-card strong, .flow-card strong { display:block; margin-bottom:6px; }
.peer-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 12px 0 20px; }
.peer-role { font-weight: 650; margin-bottom: 5px; }
.peer-meta { margin-top: 8px; font-size: .85rem; opacity: .84; }
.step-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 12px 0 20px; }
.step-num { display:inline-flex; width:28px; height:28px; align-items:center; justify-content:center; border-radius:999px; background:rgba(37,99,235,.16); font-weight:700; margin-bottom:8px; }
.flow-grid { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 9px; margin: 12px 0 20px; }
.flow-arrow { opacity:.7; font-size:.82rem; margin-top:4px; }
.result-card { border: 1px solid rgba(148,163,184,.25); border-radius: 16px; padding: 12px 16px; }
.control-note { border-left: 4px solid #2563eb; padding: 12px 15px; background: rgba(37,99,235,.07); border-radius: 8px; margin: 12px 0; }
.business-note { border-left: 4px solid #10b981; padding: 12px 15px; background: rgba(16,185,129,.07); border-radius: 8px; margin: 12px 0; }
.tech-divider { margin-top: 26px; padding-top: 10px; border-top: 1px solid rgba(148,163,184,.25); }
@media (max-width: 1100px) {
    .peer-grid { grid-template-columns: 1fr 1fr; }
    .step-grid { grid-template-columns: 1fr 1fr; }
    .flow-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 720px) {
    .business-grid, .peer-grid, .step-grid, .flow-grid { grid-template-columns: 1fr; }
}
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


def _bid_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    _, rows = build_auction_story(snapshot.mission)
    return [
        [
            row.task,
            row.peer,
            row.decision,
            row.capability,
            row.confidence,
            row.availability,
            row.estimated_cost,
            row.auction_score,
            row.outcome,
            row.reason,
        ]
        for row in rows
    ]


def _auction_story(snapshot: DemoSnapshot) -> str:
    cards, _ = build_auction_story(snapshot.mission)
    lines = [
        "## What happened in the marketplace",
        "For each work item, every peer evaluates the task independently. Qualified peers may bid; others explicitly abstain. The application validates those responses, calculates the same weighted score for every admitted bid, and awards the task to the deterministic winner.",
    ]
    for index, card in enumerate(cards, start=1):
        lines.extend(
            [
                f"### {index}. {card.task}",
                f"**Business question:** {card.business_question}",
                f"**Auction result:** **{card.winner}** won from **{card.bids} bids** and **{card.abstentions} abstentions**. Runner-up: **{card.runner_up}**.",
                card.why_winner,
            ]
        )
    return "\n\n".join(lines)


def _mission_status(snapshot: DemoSnapshot) -> str:
    metrics = snapshot.mission.metrics
    if metrics.mission_success:
        return (
            "### Due-diligence mission completed ✅\n"
            f"**Work mode:** {snapshot.mode.value.replace('_', ' ').title()}  \n"
            f"**Business tasks completed:** {metrics.tasks_completed}/5  \n"
            f"**Peers competed through:** {metrics.total_bids} bids + {metrics.total_abstentions} abstentions  \n"
            f"**Synthetic execution cost:** ${metrics.total_synthetic_execution_cost:,.0f}  \n"
            f"**Allocation efficiency:** {metrics.allocation_efficiency:.2f}"
        )
    return (
        "### Mission escalated safely ⚠️\n"
        f"**Work mode:** {snapshot.mode.value.replace('_', ' ').title()}  \n"
        f"**Business tasks completed:** {metrics.tasks_completed}/5  \n"
        "Dependent work was stopped rather than bypassing a failed allocation or execution boundary."
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


def _team_html() -> str:
    cards = build_peer_story_cards()
    peers = []
    for card in cards:
        peers.append(
            "<div class='peer-card'>"
            f"<strong>{html.escape(card.name)}</strong>"
            f"<div class='peer-role'>{html.escape(card.role)}</div>"
            f"<div>{html.escape(card.business_value)}</div>"
            f"<div class='peer-meta'><b>Strongest registered skills:</b> {html.escape(card.strengths)}<br>"
            f"<b>Current availability:</b> {html.escape(card.availability)}</div>"
            "</div>"
        )
    return "<div class='peer-grid'>" + "".join(peers) + "</div>"


def _outputs_for_snapshot(snapshot: DemoSnapshot):
    return (
        _mission_status(snapshot),
        snapshot.executive_summary,
        _auction_story(snapshot),
        _auction_rows(snapshot),
        _bid_rows(snapshot),
        _work_rows(snapshot),
        _architecture_rows(snapshot),
        snapshot.stress_summary,
        _stress_rows(snapshot),
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
        return (
            status,
            summary,
            "## Marketplace did not start\nNo auction results are available.",
            [],
            [],
            [],
            [],
            "",
            [],
            [],
            {"error": error, "mode": mode.value},
        )
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
  <div class="eyebrow">Agent 11 · Distributed Task Marketplace</div>
  <h1>A company has five diligence jobs. Six AI specialists can do overlapping work. Who should get each assignment?</h1>
  <p>This demo treats AI work allocation like a controlled internal marketplace. Instead of a permanent AI manager manually assigning every task, peers decide whether they are qualified and willing to compete. Application-owned rules validate the bids and select the winner.</p>
  <div class="pill-row">
    <span class="pill">Business case: fictional acquisition diligence</span>
    <span class="pill">6 competing AI peers</span>
    <span class="pill">5 auctioned work packages</span>
    <span class="pill">Deterministic winner selection</span>
    <span class="pill">Auditable recovery after failure</span>
  </div>
</div>

<div class="section-title">
  <div class="section-kicker">1 · The business situation</div>
  <h2>Horizon Dynamics is deciding whether to advance an acquisition</h2>
</div>
<div class="business-grid">
  <div class="business-card"><strong>🏢 Buyer</strong>Horizon Dynamics is evaluating a potential acquisition and needs a fast but controlled first-pass diligence package.</div>
  <div class="business-card"><strong>🎯 Fictional target</strong>Meridian Industrial Systems must be assessed across market attractiveness, financial quality, operating risk, and evidence quality.</div>
  <div class="business-card"><strong>📋 Decision</strong>The output is not a final acquisition approval. It is an executive diligence synthesis that helps decide whether the target should advance to deeper review.</div>
</div>
<div class="business-note"><strong>Why an auction at all?</strong> In a larger AI workforce, several agents may be capable of the same job. A marketplace makes the allocation rule explicit: who is qualified, who is available, who is confident enough to compete, what the work is expected to cost, and why one valid bid beat another.</div>

<div class="section-title">
  <div class="section-kicker">2 · Meet the AI team</div>
  <h2>Six peers have overlapping specialties — intentionally</h2>
  <p>The overlap is the point. If only one agent can do each task, there is nothing to allocate. Here, specialists and a generalist compete under the same published rules.</p>
</div>
"""
        + _team_html()
        + """
<div class="section-title">
  <div class="section-kicker">3 · How the marketplace works</div>
  <h2>Each work package follows the same four-step auction</h2>
</div>
<div class="step-grid">
  <div class="step-card"><span class="step-num">1</span><strong>Task announced</strong>The system publishes the required capability, minimum qualification, budget ceiling, dependencies, and auction round.</div>
  <div class="step-card"><span class="step-num">2</span><strong>Peers choose BID or ABSTAIN</strong>Each peer evaluates the task locally. It can compete or explicitly stay out. No central manager tells it what to do.</div>
  <div class="step-card"><span class="step-num">3</span><strong>Valid bids are scored</strong>Capability carries the most weight, followed by confidence, availability, and cost efficiency. Invalid responses never enter settlement.</div>
  <div class="step-card"><span class="step-num">4</span><strong>Winner executes</strong>The protocol awards the task deterministically. If execution explicitly fails, one bounded reauction may occur before escalation.</div>
</div>
<div class="control-note"><strong>Important:</strong> the LLM does not choose the winner. Even in LLM-assisted mode, the model only drafts the substantive work product <em>after</em> the application has already awarded the task.</div>

<div class="section-title">
  <div class="section-kicker">4 · The five work packages</div>
  <h2>Specialist work feeds verification, then executive synthesis</h2>
</div>
<div class="flow-grid">
  <div class="flow-card"><strong>① Market</strong>Assess market attractiveness.<div class="flow-arrow">Independent starting task</div></div>
  <div class="flow-card"><strong>② Finance</strong>Assess financial quality.<div class="flow-arrow">Independent starting task</div></div>
  <div class="flow-card"><strong>③ Risk</strong>Assess operating and execution risk.<div class="flow-arrow">Independent starting task</div></div>
  <div class="flow-card"><strong>④ Verify</strong>Check the evidence behind the first three work products.<div class="flow-arrow">Depends on ① ② ③</div></div>
  <div class="flow-card"><strong>⑤ Synthesize</strong>Turn validated work into an executive diligence result.<div class="flow-arrow">Depends on ① ② ③ ④</div></div>
</div>
"""
    )

    with gr.Row():
        mode = gr.Radio(
            choices=["Deterministic", "LLM-assisted"],
            value="Deterministic",
            label="Work-product execution mode",
            info="Allocation is deterministic in both modes. This switch changes only how the winning peer drafts its work product.",
        )
        gr.Markdown(
            "**Live LLM runtime:** " + llm_runtime_status()
            + "\n\n**Core rule:** *Peers decide whether to compete. The protocol decides who wins.*"
        )

    run_button = gr.Button("Run the diligence marketplace", variant="primary")

    with gr.Row():
        mission_status = gr.Markdown(DEFAULT_OUTPUTS[0], elem_classes=["result-card"])
        executive_summary = gr.Markdown(DEFAULT_OUTPUTS[1], elem_classes=["result-card"])

    gr.HTML("<div class='tech-divider'><div class='section-kicker'>5 · See what happened</div><h2>Business story first. Engineering evidence underneath.</h2></div>")

    with gr.Tabs():
        with gr.Tab("Business Walkthrough"):
            gr.Markdown(
                "This is the recommended starting point for a non-technical visitor. It explains each allocation in business language before exposing the underlying bid mechanics."
            )
            auction_story = gr.Markdown(DEFAULT_OUTPUTS[2], elem_classes=["result-card"])
            auction_table = gr.Dataframe(
                headers=AUCTION_HEADERS,
                value=DEFAULT_OUTPUTS[3],
                interactive=False,
                label="Five awarded work packages",
            )

        with gr.Tab("Auction Room"):
            gr.Markdown(
                "## See every peer decision\n"
                "This table shows the marketplace underneath the summary: which peers bid, which abstained, the inputs used for scoring, and the resulting rank. **Synthetic cost** is a demo workload/economic input, not a real invoice."
            )
            with gr.Accordion("How the auction score is calculated", open=False):
                gr.Markdown(
                    "**Auction Score = 45% Capability + 25% Confidence + 20% Availability + 10% Cost Efficiency**\n\n"
                    "- **Capability** comes from the application-owned registry; the bidder cannot invent it.\n"
                    "- **Confidence** is the peer's bounded task-specific estimate.\n"
                    "- **Availability** represents current capacity.\n"
                    "- **Cost efficiency** rewards lower cost within the task budget, but has the smallest weight so cheap work cannot overwhelm competence.\n"
                    "- Deterministic ties break on capability, then lower cost, then stable agent ID."
                )
            bid_table = gr.Dataframe(
                headers=BID_HEADERS,
                value=DEFAULT_OUTPUTS[4],
                interactive=False,
                label="All BID / ABSTAIN decisions",
            )

        with gr.Tab("Work Products"):
            gr.Markdown(
                "## What the winning peers actually produced\n"
                "Deterministic mode uses synthetic handlers. LLM-assisted mode lets the model draft only the title, summary, findings, and approved evidence references after the task has already been awarded."
            )
            work_table = gr.Dataframe(
                headers=WORK_HEADERS,
                value=DEFAULT_OUTPUTS[5],
                interactive=False,
                label="Validated work products",
            )

        with gr.Tab("Architecture Tradeoff"):
            gr.Markdown(
                "## Central manager vs distributed marketplace\n"
                "This is a controlled comparison. Both systems receive the same peers, capabilities, economics, task graph, scoring utility, executor, and failure limit. Only allocation authority changes."
            )
            architecture_table = gr.Dataframe(
                headers=ARCHITECTURE_HEADERS,
                value=DEFAULT_OUTPUTS[6],
                interactive=False,
                label="Controlled allocation comparison",
            )
            gr.Markdown(
                "**Business interpretation:** in the clean full-information case, decentralizing allocation did not improve the selected workers or synthetic execution cost. It increased coordination traffic. The benefit is organizational: allocation authority no longer lives in one semantic manager."
            )

        with gr.Tab("Stress & Recovery"):
            gr.Markdown(
                "## What happens when the marketplace is stressed?\n"
                "The deterministic evaluation changes budgets, capability floors, confidence, availability, exact ties, bid availability, and execution failures. The purpose is to show when the market reallocates work and when it correctly refuses to proceed."
            )
            stress_summary = gr.Markdown(DEFAULT_OUTPUTS[7], elem_classes=["result-card"])
            stress_table = gr.Dataframe(
                headers=STRESS_HEADERS,
                value=DEFAULT_OUTPUTS[8],
                interactive=False,
                label="Nine deterministic stress scenarios",
            )

        with gr.Tab("Engineering Audit"):
            gr.Markdown(
                "## Technical evidence\n\n"
                "**Model may draft:** title, summary, findings, and approved evidence references for an already-awarded task.\n\n"
                "**Application owns:** peer identity, registered capability, BID/ABSTAIN admission, scoring weights, tie-breaks, winner selection, award identity, evidence allowlists, work-product lineage, reauction, escalation, and publication boundaries.\n\n"
                "The append-only event stream below is the protocol evidence behind the business story."
            )
            event_table = gr.Dataframe(
                headers=EVENT_HEADERS,
                value=DEFAULT_OUTPUTS[9],
                interactive=False,
                label="Append-only mission events",
            )
            raw_snapshot = gr.JSON(value=DEFAULT_OUTPUTS[10], label="Compact mission snapshot")

    run_button.click(
        fn=_run_from_ui,
        inputs=[mode],
        outputs=[
            mission_status,
            executive_summary,
            auction_story,
            auction_table,
            bid_table,
            work_table,
            architecture_table,
            stress_summary,
            stress_table,
            event_table,
            raw_snapshot,
        ],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
