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

AUCTION_HEADERS = ["Task", "Capability", "Winning peer", "Winning score", "Bids", "Abstentions", "Rounds", "Messages", "Execution cost", "Status"]
BID_HEADERS = ["Task", "Peer", "Decision", "Capability", "Confidence", "Availability", "Est. cost", "Auction score", "Outcome", "Why"]
ARCHITECTURE_HEADERS = ["Architecture", "Mission success", "Tasks completed", "Messages", "Messages / task", "Execution cost", "Allocation efficiency", "Allocation authority"]
STRESS_HEADERS = ["Scenario", "Outcome", "Distributed messages", "Centralized messages", "Traffic multiple", "Execution cost", "Same workers", "Scenario design"]
WORK_HEADERS = ["Task", "Peer", "Title", "Summary", "Findings", "Evidence IDs"]
EVENT_HEADERS = ["Seq", "Event", "Task", "Round", "Actor", "Accepted", "Detail"]

PEER_VISUALS = {
    "Atlas Research": ("🔭", "Research"),
    "Ledger Analyst": ("🧮", "Finance"),
    "Sentinel Risk": ("🛡️", "Risk"),
    "Veritas Evidence": ("🔎", "Verification"),
    "Mosaic Generalist": ("🧩", "Generalist"),
    "Quill Synthesis": ("✍️", "Synthesis"),
}
TASK_VISUALS = {
    "Market Attractiveness Research": "🌐",
    "Financial Quality Analysis": "💹",
    "Operating & Execution Risk Review": "⚠️",
    "Evidence Verification": "✅",
    "Executive Due-Diligence Synthesis": "📝",
}

APP_CSS = """
.gradio-container { max-width: 1480px !important; }
.hero-card { border:1px solid rgba(148,163,184,.28); border-radius:24px; padding:28px 30px; margin-bottom:16px; background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.08)); }
.hero-card h1 { margin:6px 0 8px; font-size:2.1rem; line-height:1.12; max-width:1050px; }
.hero-card p { margin:0; font-size:1rem; max-width:980px; opacity:.9; }
.eyebrow,.section-kicker { text-transform:uppercase; letter-spacing:.11em; font-size:.72rem; font-weight:750; opacity:.72; }
.kpi-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin:14px 0 22px; }
.kpi-card { border:1px solid rgba(148,163,184,.26); border-radius:15px; padding:14px 12px; background:rgba(148,163,184,.04); text-align:center; }
.kpi-value { font-size:1.55rem; font-weight:800; line-height:1.05; }
.kpi-label { margin-top:5px; font-size:.78rem; opacity:.72; }
.story-strip { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; align-items:center; gap:10px; margin:12px 0 18px; }
.story-node { border:1px solid rgba(148,163,184,.27); border-radius:16px; padding:15px; min-height:105px; background:rgba(148,163,184,.045); }
.story-icon { font-size:1.55rem; margin-bottom:6px; }
.story-node strong { display:block; margin-bottom:4px; }
.story-arrow { font-size:1.25rem; opacity:.5; }
.section-title { margin:26px 0 8px; }
.section-title h2 { margin:3px 0; }
.peer-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:11px; margin:12px 0 18px; }
.peer-card { border:1px solid rgba(148,163,184,.27); border-radius:18px; padding:15px; background:rgba(148,163,184,.04); }
.peer-head { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.peer-avatar { width:42px; height:42px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:1.35rem; background:rgba(37,99,235,.10); border:1px solid rgba(37,99,235,.20); flex:0 0 auto; }
.peer-name { font-weight:800; line-height:1.1; }
.peer-role { font-size:.82rem; opacity:.76; margin-top:2px; }
.peer-value { font-size:.88rem; line-height:1.35; margin:7px 0 9px; }
.chip-row { display:flex; gap:5px; flex-wrap:wrap; }
.chip { border:1px solid rgba(148,163,184,.30); border-radius:999px; padding:4px 7px; font-size:.72rem; opacity:.88; }
.flow-wrap { display:grid; grid-template-columns:minmax(0,1.7fr) auto minmax(0,.8fr) auto minmax(0,.8fr); gap:12px; align-items:center; margin:12px 0 18px; }
.start-stack { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.flow-card { border:1px solid rgba(148,163,184,.27); border-radius:15px; padding:13px; background:rgba(148,163,184,.04); min-height:88px; }
.flow-card strong { display:block; margin-bottom:4px; }
.flow-card small { opacity:.7; }
.flow-arrow-big { font-size:1.4rem; opacity:.5; text-align:center; }
.auction-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:11px; margin-top:12px; }
.auction-card { border:1px solid rgba(148,163,184,.27); border-radius:17px; padding:15px; background:rgba(148,163,184,.04); }
.auction-card h3 { margin:0 0 7px; font-size:1rem; }
.auction-question { font-size:.86rem; opacity:.78; margin-bottom:10px; }
.auction-winner { font-weight:800; margin-bottom:7px; }
.auction-stats { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:7px; }
.auction-stat { border-radius:999px; border:1px solid rgba(148,163,184,.28); padding:4px 7px; font-size:.72rem; }
.auction-card details { margin-top:8px; font-size:.82rem; }
.auction-card summary { cursor:pointer; font-weight:700; }
.result-card { border:1px solid rgba(148,163,184,.25); border-radius:16px; padding:12px 16px; }
.control-note { border-left:4px solid #2563eb; padding:11px 14px; background:rgba(37,99,235,.07); border-radius:8px; margin:10px 0; }
.tech-divider { margin-top:24px; padding-top:10px; border-top:1px solid rgba(148,163,184,.25); }
@media(max-width:1100px){ .kpi-grid{grid-template-columns:repeat(3,1fr)} .peer-grid{grid-template-columns:1fr 1fr} .story-strip{grid-template-columns:1fr} .story-arrow{transform:rotate(90deg);text-align:center} .flow-wrap{grid-template-columns:1fr} .flow-arrow-big{transform:rotate(90deg)} }
@media(max-width:720px){ .kpi-grid{grid-template-columns:1fr 1fr} .peer-grid,.auction-grid,.start-stack{grid-template-columns:1fr} }
"""


def _auction_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [[r.task, r.capability, r.winner, r.score, r.bids, r.abstentions, r.rounds, r.messages, r.execution_cost, r.status] for r in snapshot.auction_rows]


def _architecture_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [[r.architecture, r.mission_success, r.tasks_completed, r.messages, r.messages_per_task, r.execution_cost, r.allocation_efficiency, r.allocation_authority] for r in snapshot.architecture_rows]


def _stress_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [[r.scenario, r.outcome, r.distributed_messages, r.centralized_messages, r.message_multiplier, r.execution_cost, r.same_workers, r.observation] for r in snapshot.stress_rows]


def _work_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [[r.task, r.peer, r.title, r.summary, r.findings, r.evidence] for r in snapshot.work_product_rows]


def _event_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    return [[r.sequence, r.event, r.task, r.round, r.actor, r.accepted, r.detail] for r in snapshot.event_rows]


def _bid_rows(snapshot: DemoSnapshot) -> list[list[object]]:
    _, rows = build_auction_story(snapshot.mission)
    return [[r.task, r.peer, r.decision, r.capability, r.confidence, r.availability, r.estimated_cost, r.auction_score, r.outcome, r.reason] for r in rows]


def _kpi_html(snapshot: DemoSnapshot) -> str:
    m = snapshot.mission.metrics
    efficiency = f"{m.allocation_efficiency:.0%}" if m.allocation_efficiency is not None else "—"
    items = [
        ("6", "AI peers"),
        (f"{m.tasks_completed}/5", "work packages"),
        (str(m.total_bids), "bids submitted"),
        (str(m.total_messages), "protocol messages"),
        (f"${m.total_synthetic_execution_cost:,.0f}", "synthetic cost"),
        (efficiency, "allocation efficiency"),
    ]
    return "<div class='kpi-grid'>" + "".join(
        f"<div class='kpi-card'><div class='kpi-value'>{html.escape(value)}</div><div class='kpi-label'>{html.escape(label)}</div></div>"
        for value, label in items
    ) + "</div>"


def _team_html() -> str:
    cards = build_peer_story_cards()
    rendered = []
    for card in cards:
        icon, label = PEER_VISUALS[card.name]
        strengths = [part.strip() for part in card.strengths.split("·")]
        chips = "".join(f"<span class='chip'>{html.escape(item)}</span>" for item in strengths[:2])
        rendered.append(
            "<div class='peer-card'>"
            "<div class='peer-head'>"
            f"<div class='peer-avatar'>{icon}</div>"
            f"<div><div class='peer-name'>{html.escape(card.name)}</div><div class='peer-role'>{html.escape(label)} · {html.escape(card.availability)} available</div></div>"
            "</div>"
            f"<div class='peer-value'>{html.escape(card.business_value)}</div>"
            f"<div class='chip-row'>{chips}</div>"
            "</div>"
        )
    return "<div class='peer-grid'>" + "".join(rendered) + "</div>"


def _auction_story_html(snapshot: DemoSnapshot) -> str:
    cards, _ = build_auction_story(snapshot.mission)
    rendered = []
    for card in cards:
        icon = TASK_VISUALS.get(card.task, "📌")
        rendered.append(
            "<div class='auction-card'>"
            f"<h3>{icon} {html.escape(card.task)}</h3>"
            f"<div class='auction-question'>{html.escape(card.business_question)}</div>"
            f"<div class='auction-winner'>🏆 {html.escape(card.winner)}</div>"
            "<div class='auction-stats'>"
            f"<span class='auction-stat'>{card.bids} bids</span>"
            f"<span class='auction-stat'>{card.abstentions} abstain</span>"
            f"<span class='auction-stat'>score {html.escape(card.winning_score)}</span>"
            f"<span class='auction-stat'>runner-up {html.escape(card.runner_up)}</span>"
            "</div>"
            f"<details><summary>Why this peer won</summary><p>{html.escape(card.why_winner)}</p></details>"
            "</div>"
        )
    return "<div class='auction-grid'>" + "".join(rendered) + "</div>"


def _mission_status(snapshot: DemoSnapshot) -> str:
    m = snapshot.mission.metrics
    if m.mission_success:
        return f"### Mission completed ✅\n**{m.tasks_completed}/5** diligence tasks completed with application-controlled allocation."
    return f"### Mission escalated safely ⚠️\n**{m.tasks_completed}/5** tasks completed. Dependent work stopped instead of bypassing a failed boundary."


def _raw_snapshot(snapshot: DemoSnapshot) -> dict[str, object]:
    m = snapshot.mission.metrics
    return {
        "mode": snapshot.mode.value,
        "mission_success": m.mission_success,
        "tasks_completed": m.tasks_completed,
        "total_bids": m.total_bids,
        "total_abstentions": m.total_abstentions,
        "total_messages": m.total_messages,
        "total_synthetic_execution_cost": m.total_synthetic_execution_cost,
        "allocation_efficiency": m.allocation_efficiency,
        "winning_peers": [row.winner for row in snapshot.auction_rows],
    }


def _outputs_for_snapshot(snapshot: DemoSnapshot):
    return (
        _kpi_html(snapshot),
        _mission_status(snapshot),
        snapshot.executive_summary,
        _auction_story_html(snapshot),
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
    mode = DemoMode.LLM_ASSISTED if mode_label == "LLM-assisted" else DemoMode.DETERMINISTIC
    snapshot, error = safe_run_demo(mode)
    if snapshot is None:
        return (
            "",
            "### Mission not started\n" + (error or "Live model configuration is unavailable."),
            "### No business result published\nChoose **Deterministic** to run the complete marketplace without a model provider.",
            "<div class='control-note'>Marketplace did not start, so there are no auction results to display.</div>",
            [], [], [], [], "", [], [],
            {"error": error, "mode": mode.value},
        )
    return _outputs_for_snapshot(snapshot)


DEFAULT_SNAPSHOT = run_demo(DemoMode.DETERMINISTIC)
DEFAULT_OUTPUTS = _outputs_for_snapshot(DEFAULT_SNAPSHOT)

with gr.Blocks(title="Agent 11 — Distributed Auction Task Allocation", analytics_enabled=False) as demo:
    gr.HTML("""
<div class="hero-card">
  <div class="eyebrow">Agent 11 · Distributed Task Marketplace</div>
  <h1>When several AI specialists can do the same work, who gets the assignment?</h1>
  <p>A fictional acquisition team turns five diligence jobs into a controlled internal marketplace. Peers choose whether to compete; deterministic application rules choose the winner.</p>
</div>
""")

    kpi_html = gr.HTML(DEFAULT_OUTPUTS[0])

    gr.HTML("""
<div class="section-title"><div class="section-kicker">The business case</div><h2>One acquisition question, five work packages</h2></div>
<div class="story-strip">
  <div class="story-node"><div class="story-icon">🏢</div><strong>Horizon Dynamics</strong><span>Buyer considering whether to advance a fictional acquisition.</span></div>
  <div class="story-arrow">→</div>
  <div class="story-node"><div class="story-icon">📦</div><strong>5 diligence jobs</strong><span>Market, finance, risk, verification, and executive synthesis.</span></div>
  <div class="story-arrow">→</div>
  <div class="story-node"><div class="story-icon">⚖️</div><strong>AI task marketplace</strong><span>Six peers independently BID or ABSTAIN under the same rules.</span></div>
  <div class="story-arrow">→</div>
  <div class="story-node"><div class="story-icon">📋</div><strong>Next-stage decision</strong><span>Validated specialist work becomes an executive diligence package.</span></div>
</div>
""")

    with gr.Accordion("Why use an auction instead of a manager?", open=False):
        gr.Markdown(
            "Several peers are qualified for overlapping work. The auction makes allocation explicit and auditable: "
            "who is qualified, available, confident enough to compete, within budget, and why one valid bid beat another. "
            "The demo is about **allocation authority**, not claiming auctions are always cheaper than centralized assignment."
        )

    gr.HTML("<div class='section-title'><div class='section-kicker'>Meet the team</div><h2>Six peers, overlapping specialties</h2></div>" + _team_html())

    gr.HTML("""
<div class="section-title"><div class="section-kicker">Work flow</div><h2>Three specialist tasks feed verification, then synthesis</h2></div>
<div class="flow-wrap">
  <div class="start-stack">
    <div class="flow-card"><strong>🌐 Market</strong><small>Independent starting task</small></div>
    <div class="flow-card"><strong>💹 Finance</strong><small>Independent starting task</small></div>
    <div class="flow-card"><strong>⚠️ Risk</strong><small>Independent starting task</small></div>
  </div>
  <div class="flow-arrow-big">→</div>
  <div class="flow-card"><strong>✅ Verify</strong><small>Checks all three specialist outputs</small></div>
  <div class="flow-arrow-big">→</div>
  <div class="flow-card"><strong>📝 Synthesize</strong><small>Creates the executive diligence result</small></div>
</div>
""")

    with gr.Accordion("How each auction works", open=False):
        gr.Markdown(
            "**1. Announce** the task and constraints → **2. Peers choose BID or ABSTAIN** → "
            "**3. Valid bids are scored** → **4. Winner executes**. If an awarded peer explicitly fails, "
            "the protocol allows one bounded reauction before escalation.\n\n"
            "**Score:** 45% capability + 25% confidence + 20% availability + 10% cost efficiency. "
            "The LLM never selects the winner."
        )

    with gr.Row():
        mode = gr.Radio(
            choices=["Deterministic", "LLM-assisted"], value="Deterministic",
            label="Work-product execution mode",
            info="Allocation is deterministic in both modes; this changes only how the winner drafts its work product.",
        )
        gr.Markdown("**Live LLM runtime:** " + llm_runtime_status() + "\n\n**Core rule:** *Peers decide whether to compete. The protocol decides who wins.*")

    run_button = gr.Button("Run the diligence marketplace", variant="primary")

    with gr.Row():
        mission_status = gr.Markdown(DEFAULT_OUTPUTS[1], elem_classes=["result-card"])
        with gr.Column():
            gr.Markdown("**Decision output**")
            with gr.Accordion("Open executive diligence result", open=False):
                executive_summary = gr.Markdown(DEFAULT_OUTPUTS[2])

    gr.HTML("<div class='tech-divider'><div class='section-kicker'>See the result</div><h2>Business story first. Engineering evidence underneath.</h2></div>")

    with gr.Tabs():
        with gr.Tab("Business Walkthrough"):
            gr.Markdown("### Five auctions at a glance\nOpen **Why this peer won** only when you want the scoring detail.")
            auction_story = gr.HTML(DEFAULT_OUTPUTS[3])
            with gr.Accordion("Show the compact allocation table", open=False):
                auction_table = gr.Dataframe(headers=AUCTION_HEADERS, value=DEFAULT_OUTPUTS[4], interactive=False, label="Five awarded work packages")

        with gr.Tab("Auction Room"):
            gr.Markdown("### Every BID / ABSTAIN decision\nFor the technical viewer: inspect the inputs, score, rank, and reason behind each peer decision.")
            bid_table = gr.Dataframe(headers=BID_HEADERS, value=DEFAULT_OUTPUTS[5], interactive=False, label="All peer decisions")

        with gr.Tab("Work Products"):
            gr.Markdown("### What the winning peers produced\nLLM-assisted mode changes the drafted content **after award**; it does not change allocation authority.")
            work_table = gr.Dataframe(headers=WORK_HEADERS, value=DEFAULT_OUTPUTS[6], interactive=False, label="Validated work products")

        with gr.Tab("Architecture Tradeoff"):
            gr.Markdown("### Central manager vs distributed marketplace\nSame peers, tasks, economics, utility, executor, and failure limit. Only allocation authority changes.")
            architecture_table = gr.Dataframe(headers=ARCHITECTURE_HEADERS, value=DEFAULT_OUTPUTS[7], interactive=False, label="Controlled comparison")

        with gr.Tab("Stress & Recovery"):
            gr.Markdown("### What happens under pressure?\nBudgets, capability floors, confidence, availability, ties, missing bids, and execution failures are varied deterministically.")
            stress_summary = gr.Markdown(DEFAULT_OUTPUTS[8], elem_classes=["result-card"])
            stress_table = gr.Dataframe(headers=STRESS_HEADERS, value=DEFAULT_OUTPUTS[9], interactive=False, label="Nine stress scenarios")

        with gr.Tab("Engineering Audit"):
            gr.Markdown(
                "### Technical evidence\n"
                "**Model may draft:** title, summary, findings, approved evidence references.  \n"
                "**Application owns:** identity, capability authority, admission, weights, tie-breaks, winner selection, award lineage, reauction, escalation, evidence allowlists, and publication."
            )
            event_table = gr.Dataframe(headers=EVENT_HEADERS, value=DEFAULT_OUTPUTS[10], interactive=False, label="Append-only mission events")
            raw_snapshot = gr.JSON(value=DEFAULT_OUTPUTS[11], label="Compact mission snapshot")

    run_button.click(
        fn=_run_from_ui,
        inputs=[mode],
        outputs=[kpi_html, mission_status, executive_summary, auction_story, auction_table, bid_table, work_table, architecture_table, stress_summary, stress_table, event_table, raw_snapshot],
    )

if __name__ == "__main__":
    demo.launch(css=APP_CSS)
