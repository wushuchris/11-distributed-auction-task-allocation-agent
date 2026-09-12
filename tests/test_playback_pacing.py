"""Regression tests for the human-paced live marketplace playback."""

from __future__ import annotations

import importlib


def test_starting_activity_visibly_indicates_work_is_active() -> None:
    app = importlib.import_module("app")
    html = app._starting_activity_html(app.DemoMode.DETERMINISTIC)

    assert "activity-spinner" in html
    assert "activity-running-label" in html
    assert "RUNNING" in html
    assert "indeterminate" in html
    assert "working" in html.lower()


def test_live_activity_keeps_spinner_until_completion() -> None:
    app = importlib.import_module("app")
    snapshot = app.DEFAULT_SNAPSHOT

    running = app._activity_html(snapshot, 1)
    complete = app._activity_html(snapshot, len(snapshot.event_rows), complete=True)

    assert "activity-spinner" in running
    assert "RUNNING" in running
    assert "Marketplace run complete" in complete
    assert "activity-spinner" not in complete
    assert "RUNNING" not in complete


def test_completed_activity_retains_all_protocol_messages_in_scrollable_transcript() -> None:
    app = importlib.import_module("app")
    snapshot = app.DEFAULT_SNAPSHOT

    complete = app._activity_html(snapshot, len(snapshot.event_rows), complete=True)

    assert "activity-feed activity-scroll" in complete
    assert "Scroll inside this transcript" in complete
    assert complete.count("activity-event") == len(snapshot.event_rows)
    assert "Message 1" in complete
    assert f"Message {len(snapshot.event_rows)}" in complete


def test_live_activity_remains_compact_before_completion() -> None:
    app = importlib.import_module("app")
    snapshot = app.DEFAULT_SNAPSHOT

    running = app._activity_html(snapshot, 20)

    assert "activity-scroll" not in running
    assert running.count("activity-event") == 8


def test_important_protocol_events_receive_longer_reading_beats() -> None:
    app = importlib.import_module("app")
    base = 0.2

    bid = app._playback_delay_for_event("Bid", base)
    announcement = app._playback_delay_for_event("Task Announcement", base)
    award = app._playback_delay_for_event("Task Award", base)
    result = app._playback_delay_for_event("Task Result", base)

    assert bid == base
    assert announcement > bid
    assert award > announcement
    assert result > award
    assert app._playback_delay_for_event("Bid", 0) == 0
