from pathlib import Path

from scr.data.semantic import SemanticConfig
from scr.utils.serialization import read_jsonl


def load_semantic_human_decisions(
    queue: list[dict],
    review_path: Path,
    scope: str
) -> dict:
    """Charge et valide les décisions humaines."""
    decisions = (
        {
            decision["id"]: decision
            for decision in read_jsonl(review_path)
        }
        if review_path.is_file()
        else {}
    )

    queue_ids = {
        record["id"]
        for record in queue
    }

    unknown_ids = set(decisions) - queue_ids

    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids):,} décisions humaines "
            f"hors périmètre {scope}."
        )

    invalid_ids = [
        record_id
        for record_id, decision in decisions.items()
        if decision.get("verdict") not in {
            SemanticConfig.PASS,
            SemanticConfig.FAIL
        }
    ]

    if invalid_ids:
        raise ValueError(
            f"{len(invalid_ids):,} décisions humaines "
            f"{scope} avec verdict invalide."
        )

    return decisions


def apply_semantic_human_decisions(
    queue: list[dict],
    decisions: dict
) -> None:
    """Rattache les décisions humaines aux records."""
    for record in queue:
        decision = decisions.get(
            record["id"]
        )

        if decision is not None:
            record["semantic_human_review"] = decision
            
def final_semantic_verdict(
    record: dict
) -> str:
    """Retourne le verdict sémantique final."""
    human = record.get(
        "semantic_human_review"
    )

    if isinstance(human, dict):
        verdict = human.get("verdict")

        if verdict in {
            SemanticConfig.PASS,
            SemanticConfig.FAIL
        }:
            return verdict

    return record[
        "semantic_quality"
    ]["verdict"]