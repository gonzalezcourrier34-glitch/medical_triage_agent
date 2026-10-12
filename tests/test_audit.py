"""Tests du journal d'audit : une ligne par décision, sans texte patient."""

import json
import os

os.environ["INFERENCE_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from src.api.routes import triage  # noqa: E402

QUESTION = "Patient fictif Jean Dupont, douleur thoracique depuis 20 minutes."
PAYLOAD = {"language": "fr", "question": QUESTION}


def read_entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_success_is_audited(audit_log_in_tmp):
    """Chaque décision produit une ligne reliée à la réponse par request_id."""
    with TestClient(app) as client:
        body = client.post("/triage", json=PAYLOAD).json()

    entries = read_entries(audit_log_in_tmp)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["request_id"] == body["request_id"]
    assert entry["priority"] == body["priority"]
    assert entry["model_version"] == body["model_version"]
    assert entry["prompt_version"] == "triage-classification-v2"
    assert entry["status"] == "success"
    assert entry["language"] == "fr"
    assert len(entry["input_sha256"]) == 64


def test_patient_text_never_written(audit_log_in_tmp):
    """Minimisation RGPD : ni le texte ni le nom du patient dans le journal."""
    with TestClient(app) as client:
        client.post("/triage", json=PAYLOAD)

    raw = audit_log_in_tmp.read_text(encoding="utf-8")
    assert "Dupont" not in raw
    assert "thoracique" not in raw


def test_same_input_same_fingerprint(audit_log_in_tmp):
    """La même entrée donne la même empreinte : une décision est vérifiable."""
    with TestClient(app) as client:
        client.post("/triage", json=PAYLOAD)
        client.post("/triage", json=PAYLOAD)
        client.post("/triage", json={"language": "fr", "question": "Autre cas."})

    first, second, other = read_entries(audit_log_in_tmp)
    assert first["input_sha256"] == second["input_sha256"]
    assert first["input_sha256"] != other["input_sha256"]
    assert first["request_id"] != second["request_id"]


def test_engine_failure_is_audited(audit_log_in_tmp, monkeypatch):
    """Une panne moteur est tracée elle aussi, sans priorité."""

    async def failing_predict(request):
        raise RuntimeError("panne")

    monkeypatch.setattr(triage.inference_service, "predict", failing_predict)

    with TestClient(app) as client:
        response = client.post("/triage", json=PAYLOAD)

    assert response.status_code == 503
    (entry,) = read_entries(audit_log_in_tmp)
    assert entry["status"] == "engine_unavailable"
    assert entry["priority"] is None


def test_audit_write_error_does_not_break_triage(tmp_path, monkeypatch):
    """Journal impossible à écrire : la réponse au patient part quand même."""
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(blocker / "audit.jsonl"))

    with TestClient(app) as client:
        response = client.post("/triage", json=PAYLOAD)

    assert response.status_code == 200


def test_audit_can_be_disabled(audit_log_in_tmp, monkeypatch):
    """AUDIT_LOG_ENABLED=false : aucun fichier écrit."""
    monkeypatch.setenv("AUDIT_LOG_ENABLED", "false")

    with TestClient(app) as client:
        client.post("/triage", json=PAYLOAD)

    assert not audit_log_in_tmp.exists()
