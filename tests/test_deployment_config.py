"""Regression tests for Agent 11 Hugging Face deployment plumbing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_declares_native_gradio_space_metadata() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "sdk_version: 6.27.0" in readme
    assert "app_file: app.py" in readme
    assert 'python_version: "3.11"' in readme
    assert "license: mit" in readme


def test_deployment_waits_for_tests_and_never_runs_on_pull_requests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "deploy-hugging-face:" in workflow
    assert "needs: test" in workflow
    assert "github.event_name != 'pull_request'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow


def test_deployment_uses_only_the_dedicated_github_secret() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "secrets.HF_DEPLOY_TOKEN" in workflow
    assert "secrets.HF_TOKEN" not in workflow
    assert "HF_TOKEN:" not in workflow
    assert "deployment will be skipped safely" in workflow


def test_deployment_targets_agent_11_space() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "huggingface.co/spaces/FlyingNunchucks/"
        "11-distributed-auction-task-allocation-agent"
    ) in workflow


def test_example_environment_contains_no_real_secret() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "HF_TOKEN=your_runtime_token" in example
    assert "MODEL_ID=your_model_id" in example
    assert "hf_" not in example
