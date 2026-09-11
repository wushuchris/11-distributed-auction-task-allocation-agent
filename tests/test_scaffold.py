"""Smoke tests for the Agent 11 repository scaffold."""

import auction_coordination


def test_package_imports() -> None:
    assert auction_coordination.__version__ == "0.1.0"
