"""Configuration commune des tests."""

import pytest


@pytest.fixture(autouse=True)
def audit_log_in_tmp(tmp_path, monkeypatch):
    """Le journal d'audit des tests est écrit dans un dossier temporaire."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(path))
    return path
