"""Regression tests for the Agent 11 business-first demo presentation layer."""

from __future__ import annotations

import importlib

from auction_coordination.demo import (
    DemoMode,
    llm_runtime_status,
    run_demo,
    safe_run_demo,
)


def test_deterministic_demo_preserves_mission_and_business_metrics() -> None:
    snapshot = run_demo(DemoMode.DETERMINISTIC)

    assert snapshot.mission.metrics.mission_success is True
    assert snapshot.mission.metrics.tasks_completed == 5
    assert snapshot.mission.metrics.total_messages == 50
    assert snapshot.mission.metrics.total_synthetic_execution_cost == 347.0
    assert snapshot.mission.metrics.allocation_efficiency == 1.0
    assert [row.winner for row in snapshot.auction_rows] == [
        "Atlas Research",
        "Ledger Analyst",
        "Sentinel Risk",
        "Veritas Evidence",
        "Quill Synthesis",
    ]
    assert "no semantic manager chose the workers" in snapshot.executive_summary


def test_demo_architecture_view_is_controlled_comparison() -> None:
    snapshot = run_demo()

    assert len(snapshot.architecture_rows) == 2
    distributed, centralized = snapshot.architecture_rows
    assert distributed.architecture == "Distributed auction"
    assert distributed.messages == 50
    assert distributed.execution_cost == "$347"
    assert distributed.allocation_efficiency == "1.00"
    assert centralized.architecture == "Centralized baseline"
    assert centralized.messages == 10
    assert centralized.execution_cost == "$347"
    assert centralized.allocation_efficiency == "1.00"
    assert "same workers" in centralized.allocation_authority


def test_demo_stress_view_preserves_nine_scenario_evaluation() -> None:
    snapshot = run_demo()

    assert len(snapshot.stress_rows) == 9
    assert sum(row.outcome == "Both succeed" for row in snapshot.stress_rows) == 7
    assert sum(row.outcome == "Both escalate" for row in snapshot.stress_rows) == 2
    assert "390 protocol messages" in snapshot.stress_summary
    assert "78" in snapshot.stress_summary
    assert "5.0× coordination traffic" in snapshot.stress_summary
    assert "$2,639" in snapshot.stress_summary


def test_work_products_and_audit_rows_are_derived_from_runtime() -> None:
    snapshot = run_demo()

    assert len(snapshot.work_product_rows) == 5
    assert snapshot.work_product_rows[-1].task == "Executive Synthesis"
    assert snapshot.work_product_rows[-1].peer == "Quill Synthesis"
    assert len(snapshot.event_rows) == 50
    assert snapshot.event_rows[0].event == "Task Announcement"
    assert snapshot.event_rows[-1].event == "Task Result"


def test_llm_status_never_exposes_token_value() -> None:
    env = {
        "HF_TOKEN": "hf_super_secret_test_value",
        "MODEL_ID": "example/model",
    }
    status = llm_runtime_status(env)

    assert status == "configured for `example/model`"
    assert "hf_super_secret_test_value" not in status


def test_safe_llm_demo_reports_missing_configuration_without_starting() -> None:
    snapshot, error = safe_run_demo(DemoMode.LLM_ASSISTED)

    assert snapshot is None
    assert error is not None
    assert "HF_TOKEN" in error or "MODEL_ID" in error


def test_app_imports_and_builds_gradio_blocks() -> None:
    app = importlib.import_module("app")

    assert app.demo is not None
    assert app.DEFAULT_SNAPSHOT.mission.metrics.mission_success is True
    assert len(app.DEFAULT_OUTPUTS) == 9


def test_ui_deterministic_run_returns_business_first_outputs() -> None:
    app = importlib.import_module("app")

    outputs = app._run_from_ui("Deterministic")

    assert len(outputs) == 9
    assert "Mission completed" in outputs[0]
    assert "Executive due-diligence result" in outputs[1]
    assert len(outputs[2]) == 5
    assert len(outputs[3]) == 2
    assert len(outputs[5]) == 9
    assert len(outputs[6]) == 5
    assert len(outputs[7]) == 50
    assert outputs[8]["mission_success"] is True
