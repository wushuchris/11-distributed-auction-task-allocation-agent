"""Gradio demo for Agent 11 — Distributed Auction Task Allocation Agent."""

from __future__ import annotations

import html
import sys
import time
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
EVENT_VISUALS = {
    "Task Announcement": "📣",
    "Bid": "💬",
    "Bid Abstention": "↩️",
    "Auction Closed": "🔒",
    "Task Award": "🏆",
    "Task Accepted": "🤝",
    "Task Result": "✅",
    "Task Failure": "❌",
    "Reauction Announcement": "🔁",
    "Escalation": "⚠️",
}
PLAYBACK_DELAY_SECONDS = 0.10

APP_CSS = """
.gradio-container { max-width: 980px !important; }
.hero-card {
    border:1px solid rgba(148,163,184,.28);
    border-radius:24px;
    padding:28px 30px;
    margin-bottom:16px;
    background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.08));
}
.hero-card h1 { margin:6px 0 8px; font-size:2.08rem; line-height:1.14; max-width:820px; }
.hero-card p { margin:0; font-size:1.06rem; line-height:1.5; max-width:790px; opacity:.92; }
.eyebrow,.section-kicker { text-transform:uppercase; letter-spacing:.11em; font-size:.78rem; font-weight:760; opacity:.74; }
.section-title { margin:30px 0 10px; }
.section-title h2 { margin:3px 0; font-size:1.58rem; line-height:1.25; }

/* Vertical mission snapshot: three rows instead of six side-by-side tiles. */
.kpi-grid { display:grid; grid-template-columns:1fr; gap:9px; margin:14px 0 24px; }
.kpi-card {
    display:grid;
    grid-template-columns:minmax(145px,.75fr) 1fr 1fr;
    gap:14px;
    align-items:center;
    border:1px solid rgba(148,163,184,.25);
    border-radius:15px;
    padding:13px 16px;
    background:rgba(148,163,184,.035);
}
.kpi-topic { font-size:.84rem; font-weight:770; letter-spacing:.05em; text-transform:uppercase; opacity:.70; }
.kpi-pair { display:flex; align-items:baseline; gap:8px; }
.kpi-value { font-size:1.32rem; font-weight:800; line-height:1; }
.kpi-label { font-size:.86rem; opacity:.76; }

/* One obvious top-to-bottom business path. */
.story-strip { position:relative; display:grid; grid-template-columns:1fr; gap:10px; margin:12px 0 18px; padding-left:42px; }
.story-strip::before { content:""; position:absolute; left:17px; top:22px; bottom:22px; width:2px; background:rgba(37,99,235,.20); }
.story-node { position:relative; border:1px solid rgba(148,163,184,.25); border-radius:15px; padding:15px 17px; background:rgba(148,163,184,.035); }
.story-node::before { content:""; position:absolute; left:-32px; top:23px; width:12px; height:12px; border-radius:50%; background:#2563eb; box-shadow:0 0 0 5px rgba(37,99,235,.10); }
.story-head { display:flex; align-items:center; gap:9px; margin-bottom:5px; }
.story-icon { font-size:1.32rem; }
.story-node strong { font-size:1.06rem; }
.story-node span { display:block; font-size:.97rem; opacity:.82; line-height:1.45; }

/* Vertical roster removes 3x2 eye jumps. */
.peer-grid { display:grid; grid-template-columns:1fr; gap:9px; margin:12px 0 18px; }
.peer-card {
    display:grid;
    grid-template-columns:50px minmax(180px,.75fr) 1.35fr;
    gap:13px;
    align-items:center;
    border:1px solid rgba(148,163,184,.25);
    border-radius:16px;
    padding:14px 16px;
    background:rgba(148,163,184,.035);
}
.peer-head { display:contents; }
.peer-avatar { width:44px; height:44px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:1.34rem; background:rgba(37,99,235,.10); border:1px solid rgba(37,99,235,.20); }
.peer-name { font-size:1.06rem; font-weight:800; line-height:1.15; }
.peer-role { font-size:.89rem; opacity:.76; margin-top:4px; }
.peer-details { min-width:0; }
.peer-value { font-size:.96rem; line-height:1.45; margin-bottom:8px; }
.chip-row { display:flex; gap:6px; flex-wrap:wrap; }
.chip { border:1px solid rgba(148,163,184,.30); border-radius:999px; padding:4px 8px; font-size:.78rem; opacity:.90; }

/* Vertical dependency timeline instead of left-to-right arrows. */
.flow-wrap { position:relative; display:grid; grid-template-columns:1fr; gap:9px; margin:12px 0 18px; padding-left:42px; }
.flow-wrap::before { content:""; position:absolute; left:17px; top:25px; bottom:25px; width:2px; background:rgba(16,185,129,.23); }
.flow-stage { position:relative; border:1px solid rgba(148,163,184,.25); border-radius:15px; padding:15px 17px; background:rgba(148,163,184,.035); }
.flow-stage::before { content:""; position:absolute; left:-32px; top:23px; width:12px; height:12px; border-radius:50%; background:#10b981; box-shadow:0 0 0 5px rgba(16,185,129,.10); }
.flow-stage strong { display:block; margin-bottom:6px; font-size:1.05rem; }
.flow-stage small { font-size:.94rem; line-height:1.4; opacity:.78; }
.start-stack { display:flex; gap:7px; flex-wrap:wrap; margin-top:9px; }
.flow-card { border:1px solid rgba(148,163,184,.24); border-radius:999px; padding:6px 10px; font-size:.86rem; background:rgba(148,163,184,.03); }

/* Live protocol playback makes execution visible instead of showing only a spinner. */
.activity-shell { border:1px solid rgba(37,99,235,.26); border-radius:18px; padding:16px 18px; margin:14px 0 12px; background:linear-gradient(135deg,rgba(37,99,235,.09),rgba(16,185,129,.035)); }
.activity-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:10px; }
.activity-title { font-size:1.04rem; font-weight:800; }
.activity-subtitle { font-size:.91rem; line-height:1.4; opacity:.78; margin-top:3px; }
.activity-count { flex:0 0 auto; font-size:.82rem; font-weight:750; border:1px solid rgba(148,163,184,.28); border-radius:999px; padding:5px 9px; }
.activity-progress { height:7px; border-radius:999px; overflow:hidden; background:rgba(148,163,184,.16); margin-bottom:12px; }
.activity-progress > span { display:block; height:100%; background:linear-gradient(90deg,#2563eb,#10b981); transition:width .18s ease; }
.activity-feed { display:grid; gap:7px; }
.activity-event { display:grid; grid-template-columns:32px minmax(0,1fr); gap:9px; align-items:start; border-top:1px solid rgba(148,163,184,.16); padding-top:8px; }
.activity-event:first-child { border-top:0; padding-top:0; }
.activity-event.latest { border-radius:10px; padding:8px 9px; margin:0 -9px; background:rgba(37,99,235,.075); border-top-color:transparent; }
.activity-icon { font-size:1.05rem; line-height:1.45; }
.activity-main { font-size:.94rem; line-height:1.35; }
.activity-main strong { font-weight:800; }
.activity-meta { font-size:.78rem; opacity:.67; margin-top:2px; }
.activity-detail { font-size:.82rem; opacity:.77; margin-top:3px; line-height:1.35; }
.activity-complete { border-color:rgba(16,185,129,.32); background:linear-gradient(135deg,rgba(16,185,129,.10),rgba(37,99,235,.04)); }

/* Auction outcomes also become a single reading column. */
.auction-grid { display:grid; grid-template-columns:1fr; gap:10px; margin-top:12px; }
.auction-card { border:1px solid rgba(148,163,184,.25); border-radius:16px; padding:16px 18px; background:rgba(148,163,184,.035); }
.auction-top { display:grid; grid-template-columns:minmax(0,1.28fr) minmax(210px,.72fr); gap:16px; align-items:start; }
.auction-card h3 { margin:0 0 6px; font-size:1.10rem; line-height:1.3; }
.auction-question { font-size:.95rem; line-height:1.45; opacity:.80; }
.auction-winner { font-size:1rem; font-weight:800; margin-bottom:7px; }
.auction-stats { display:flex; gap:6px; flex-wrap:wrap; }
.auction-stat { border-radius:999px; border:1px solid rgba(148,163,184,.28); padding:4px 8px; font-size:.78rem; }
.auction-card details { margin-top:10px; font-size:.92rem; line-height:1.45; }
.auction-card summary { cursor:pointer; font-size:.94rem; font-weight:700; }

.result-card { border:1px solid rgba(148,163,184,.25); border-radius:16px; padding:14px 17px; font-size:1rem; line-height:1.45; }
.control-note { border-left:4px solid #2563eb; padding:12px 15px; background:rgba(37,99,235,.07); border-radius:8px; margin:10px 0; font-size:.96rem; line-height:1.45; }
.tech-divider { margin-top:26px; padding-top:10px; border-top:1px solid rgba(148,163,184,.25); }

@media(max-width:780px){
    .gradio-container { max-width:100% !important; }
    .kpi-card { grid-template-columns:1fr; gap:7px; }
    .peer-card { grid-template-columns:50px 1fr; }
    .peer-details { grid-column:1 / -1; padding-left:63px; }
    .auction-top { grid-template-columns:1fr; }
    .activity-head { display:block; }
    .activity-count { display:inline-block; margin-top:8px; }
}
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
    groups = [
        ("Workforce", ("6", "AI peers"), (f"{m.tasks_completed}/5", "work packages")),
        ("Marketplace", (str(m.total_bids), "bids submitted"), (str(m.total_messages), "protocol messages")),
        ("Outcome", (f"${m.total_synthetic_execution_cost:,.0f}", "synthetic cost"), (efficiency, "allocation efficiency")),
    ]
    cards = []
    for topic, left, right in groups:
        cards.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-topic'>{html.escape(topic)}</div>"
            f"<div class='kpi-pair'><span class='kpi-value'>{html.escape(left[0])}</span><span class='kpi-label'>{html.escape(left[1])}</span></div>"
            f"<div class='kpi-pair'><span class='kpi-value'>{html.escape(right[0])}</span><span class='kpi-label'>{html.escape(right[1])}</span></div>"
            "</div>"
        )
    return "<div class='kpi-grid'>" + "".join(cards) + "</div>"


def _team_html() -> str:
    cards = build_peer_story_cards()
    rendered = []
    for card in cards:
        icon, label = PEER_VISUALS[card.name]
        strengths = [part.strip() for part in card.strengths.split("·")]
        chips = "".join(f"<span class='chip'>{html.escape(item)}</span>" for item in strengths[:2])
        rendered.append(
            "<div class='peer-card'>"
            f"<div class='peer-avatar'>{icon}</div>"
            f"<div><div class='peer-name'>{html.escape(card.name)}</div><div class='peer-role'>{html.escape(label)} · {html.escape(card.availability)} available</div></div>"
            "<div class='peer-details'>"
            f"<div class='peer-value'>{html.escape(card.business_value)}</div>"
            f"<div class='chip-row'>{chips}</div>"
            "</div>"
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
            "<div class='auction-top'>"
            "<div>"
            f"<h3>{icon} {html.escape(card.task)}</h3>"
            f"<div class='auction-question'>{html.escape(card.business_question)}</div>"
            "</div>"
            "<div>"
            f"<div class='auction-winner'>🏆 {html.escape(card.winner)}</div>"
            "<div class='auction-stats'>"
            f"<span class='auction-stat'>{card.bids} bids</span>"
            f"<span class='auction-stat'>{card.abstentions} abstain</span>"
            f"<span class='auction-stat'>score {html.escape(card.winning_score)}</span>"
            f"<span class='auction-stat'>runner-up {html.escape(card.runner_up)}</span>"
            "</div>"
            "</div>"
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


def _idle_activity_html() -> str:
    return (
        "<div class='activity-shell'>"
        "<div class='activity-head'><div><div class='activity-title'>▶ Live Marketplace Activity</div>"
        "<div class='activity-subtitle'>Click <b>Run the diligence marketplace</b> to watch task announcements, peer bids, abstentions, awards, and results appear here.</div></div>"
        "<div class='activity-count'>Ready</div></div>"
        "<div class='activity-progress'><span style='width:0%'></span></div>"
        "</div>"
    )


def _starting_activity_html(mode: DemoMode) -> str:
    mode_name = "LLM-assisted" if mode is DemoMode.LLM_ASSISTED else "Deterministic"
    return (
        "<div class='activity-shell'>"
        "<div class='activity-head'><div><div class='activity-title'>⚙️ Starting marketplace</div>"
        f"<div class='activity-subtitle'>{html.escape(mode_name)} work-product execution selected. The allocation protocol is preparing the first task auction.</div></div>"
        "<div class='activity-count'>Starting</div></div>"
        "<div class='activity-progress'><span style='width:2%'></span></div>"
        "</div>"
    )


def _event_action(event_name: str) -> str:
    return {
        "Task Announcement": "opened an auction",
        "Bid": "submitted a BID",
        "Bid Abstention": "ABSTAINED",
        "Auction Closed": "closed the auction",
        "Task Award": "issued the award",
        "Task Accepted": "accepted the assignment",
        "Task Result": "completed the work",
        "Task Failure": "reported a failure",
        "Reauction Announcement": "reopened the auction",
        "Escalation": "escalated the task",
    }.get(event_name, event_name.lower())


def _activity_html(snapshot: DemoSnapshot, event_count: int, *, complete: bool = False) -> str:
    total = max(len(snapshot.event_rows), 1)
    event_count = max(0, min(event_count, len(snapshot.event_rows)))
    progress = 100 if complete else max(2, round((event_count / total) * 100))
    visible_events = snapshot.event_rows[max(0, event_count - 8):event_count]
    completed_tasks = sum(1 for row in snapshot.event_rows[:event_count] if row.event == "Task Result")

    if event_count:
        current = snapshot.event_rows[event_count - 1]
        current_task = current.task
        current_phase = current.event
    else:
        current_task = "Preparing first auction"
        current_phase = "Starting"

    if complete:
        title = "✅ Marketplace run complete"
        subtitle = f"{snapshot.mission.metrics.tasks_completed}/5 work packages completed. Final results are now published below."
        counter = f"{len(snapshot.event_rows)} messages"
        shell_class = "activity-shell activity-complete"
    else:
        title = f"⚡ Live Marketplace Activity · {current_task}"
        subtitle = f"Phase: {current_phase} · Completed work packages: {completed_tasks}/5"
        counter = f"{event_count}/{len(snapshot.event_rows)} messages"
        shell_class = "activity-shell"

    rendered = []
    for index, row in enumerate(visible_events):
        icon = EVENT_VISUALS.get(row.event, "•")
        actor = row.actor if row.actor != "—" else "Protocol"
        latest = " latest" if index == len(visible_events) - 1 and not complete else ""
        rendered.append(
            f"<div class='activity-event{latest}'>"
            f"<div class='activity-icon'>{icon}</div>"
            "<div>"
            f"<div class='activity-main'><strong>{html.escape(actor)}</strong> {_event_action(row.event)} · {html.escape(row.task)}</div>"
            f"<div class='activity-meta'>Round {row.round} · Message {row.sequence} · {html.escape(row.event)}</div>"
            f"<div class='activity-detail'>{html.escape(row.detail)}</div>"
            "</div></div>"
        )

    feed = "".join(rendered) or "<div class='activity-subtitle'>Waiting for the first protocol message…</div>"
    return (
        f"<div class='{shell_class}'>"
        "<div class='activity-head'><div>"
        f"<div class='activity-title'>{title}</div>"
        f"<div class='activity-subtitle'>{html.escape(subtitle)}</div>"
        "</div>"
        f"<div class='activity-count'>{html.escape(counter)}</div></div>"
        f"<div class='activity-progress'><span style='width:{progress}%'></span></div>"
        f"<div class='activity-feed'>{feed}</div>"
        "</div>"
    )


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


def _running_outputs(mode: DemoMode):
    return (
        "<div class='control-note'><strong>Marketplace running.</strong> Final metrics will publish after the protocol playback completes.</div>",
        "### Mission running…\nWatch **Live Marketplace Activity** above for the current task and protocol phase.",
        "### Executive result pending\nThe final diligence synthesis will publish only after the work chain completes.",
        "<div class='control-note'>Auction outcomes will publish after the live protocol playback.</div>",
        [], [], [], [], "", [], [],
        {"status": "running", "mode": mode.value},
    )


def _stream_run_from_ui(mode_label: str, playback_delay: float = PLAYBACK_DELAY_SECONDS):
    """Yield visible protocol playback frames before publishing final mission outputs."""

    mode = DemoMode.LLM_ASSISTED if mode_label == "LLM-assisted" else DemoMode.DETERMINISTIC
    running = _running_outputs(mode)
    yield (_starting_activity_html(mode), *running)

    snapshot, error = safe_run_demo(mode)
    if snapshot is None:
        failed = _run_from_ui(mode_label)
        error_activity = (
            "<div class='activity-shell'><div class='activity-head'><div>"
            "<div class='activity-title'>⚠️ Marketplace could not start</div>"
            f"<div class='activity-subtitle'>{html.escape(error or 'Runtime configuration is unavailable.')}</div>"
            "</div><div class='activity-count'>Stopped safely</div></div></div>"
        )
        yield (error_activity, *failed)
        return

    for event_count in range(1, len(snapshot.event_rows) + 1):
        yield (_activity_html(snapshot, event_count), *running)
        if playback_delay > 0:
            time.sleep(playback_delay)

    yield (_activity_html(snapshot, len(snapshot.event_rows), complete=True), *_outputs_for_snapshot(snapshot))


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
<div class="section-title"><div class="section-kicker">The business case</div><h2>Follow one decision from buyer to diligence result</h2></div>
<div class="story-strip">
  <div class="story-node"><div class="story-head"><div class="story-icon">🏢</div><strong>1 · Horizon Dynamics</strong></div><span>Buyer considering whether to advance a fictional acquisition.</span></div>
  <div class="story-node"><div class="story-head"><div class="story-icon">📦</div><strong>2 · Five diligence jobs</strong></div><span>Market, finance, risk, verification, and executive synthesis must be completed.</span></div>
  <div class="story-node"><div class="story-head"><div class="story-icon">⚖️</div><strong>3 · AI task marketplace</strong></div><span>Six peers independently BID or ABSTAIN under the same published rules.</span></div>
  <div class="story-node"><div class="story-head"><div class="story-icon">📋</div><strong>4 · Next-stage decision</strong></div><span>Validated specialist work becomes an executive diligence package.</span></div>
</div>
""")

    with gr.Accordion("Why use an auction instead of a manager?", open=False):
        gr.Markdown(
            "Several peers are qualified for overlapping work. The auction makes allocation explicit and auditable: "
            "who is qualified, available, confident enough to compete, within budget, and why one valid bid beat another. "
            "The demo is about **allocation authority**, not claiming auctions are always cheaper than centralized assignment."
        )

    gr.HTML("<div class='section-title'><div class='section-kicker'>Meet the team</div><h2>Read the roster from top to bottom</h2></div>" + _team_html())

    gr.HTML("""
<div class="section-title"><div class="section-kicker">Work flow</div><h2>The diligence package advances through three stages</h2></div>
<div class="flow-wrap">
  <div class="flow-stage"><strong>Stage 1 · Specialist analysis</strong><small>Three independent tasks can begin without upstream work.</small><div class="start-stack"><span class="flow-card">🌐 Market</span><span class="flow-card">💹 Finance</span><span class="flow-card">⚠️ Risk</span></div></div>
  <div class="flow-stage"><strong>Stage 2 · Evidence verification</strong><small>✅ Veritas checks the evidence and lineage behind all three specialist outputs.</small></div>
  <div class="flow-stage"><strong>Stage 3 · Executive synthesis</strong><small>📝 Validated work is converted into the final next-stage diligence recommendation.</small></div>
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

    gr.Markdown("### Run the marketplace")
    mode = gr.Radio(
        choices=["Deterministic", "LLM-assisted"], value="Deterministic",
        label="Work-product execution mode",
        info="Allocation is deterministic in both modes; this changes only how the winner drafts its work product.",
    )
    gr.Markdown("**Live LLM runtime:** " + llm_runtime_status() + "  \n**Core rule:** *Peers decide whether to compete. The protocol decides who wins.*")
    run_button = gr.Button("Run the diligence marketplace", variant="primary")

    live_activity = gr.HTML(_idle_activity_html())
    mission_status = gr.Markdown(DEFAULT_OUTPUTS[1], elem_classes=["result-card"])
    with gr.Accordion("Open executive diligence result", open=False):
        executive_summary = gr.Markdown(DEFAULT_OUTPUTS[2])

    gr.HTML("<div class='tech-divider'><div class='section-kicker'>See the result</div><h2>Business story first. Engineering evidence underneath.</h2></div>")

    with gr.Tabs():
        with gr.Tab("Business Walkthrough"):
            gr.Markdown("### Five auctions, one vertical reading path\nOpen **Why this peer won** only when you want the scoring detail.")
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
        fn=_stream_run_from_ui,
        inputs=[mode],
        outputs=[live_activity, kpi_html, mission_status, executive_summary, auction_story, auction_table, bid_table, work_table, architecture_table, stress_summary, stress_table, event_table, raw_snapshot],
    )

if __name__ == "__main__":
    demo.launch(css=APP_CSS)
