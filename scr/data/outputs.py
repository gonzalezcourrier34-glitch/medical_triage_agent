from pathlib import Path


# Configuration des sorties
# Isole chaque exécution dans un run versionné et conserve les artefacts intermédiaires dans draft.

class OutputConfig:
    """Configuration des sorties du nettoyage."""

    def __init__(self, project_root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = (
            project_root / "data" / "processed"
            / "runs" / run_id
        )
        self.dir = self.run_dir / "draft"

        self.processed = self.dir / "processed_records.jsonl"
        self.clean = self.dir / "clean_records.jsonl"
        self.blocked = self.dir / "blocked_records.jsonl"
        self.pii = self.dir / "pii_candidates.jsonl"
        self.identity = self.dir / "identity_review.jsonl"
        self.quality = self.dir / "quality_signals.json"
        self.summary = self.dir / "cleaning_summary.json"

        self.dir.mkdir(parents=True, exist_ok=True)


def json_ready_record(record: dict) -> dict:
    """Prépare un record pour JSON."""
    return {
        key: sorted(value)
        if isinstance(value, set)
        else value
        for key, value in record.items()
    }


def json_ready_records(
    source_records: list[dict]
) -> list[dict]:
    """Prépare plusieurs records pour JSON."""
    return [
        json_ready_record(record)
        for record in sorted(
            source_records,
            key=lambda record: record["id"]
        )
    ]