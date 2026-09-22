from json import JSONDecodeError
from pathlib import Path

import pandas as pd

from scr.config import CONFIG
from scr.utils.hashing import file_sha256
from scr.utils.serialization import (
    read_json,
    read_jsonl
)

def project_root() -> Path:
    """Localise la racine du projet."""
    if CONFIG.project_root_override is not None:
        return CONFIG.project_root_override.resolve()

    for path in (
        Path.cwd(),
        *Path.cwd().parents
    ):
        if (path / "data" / "raw").is_dir():
            return path

    raise FileNotFoundError(
        "Projet introuvable : "
        "renseigner CONFIG.project_root_override."
    )


def load_raw(
    path: Path
) -> pd.DataFrame:
    """Charge un dataset brut."""
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False
        )

    try:
        payload = read_json(path)
    except JSONDecodeError:
        payload = read_jsonl(path)

    if (
        not isinstance(payload, list)
        or not all(
            isinstance(item, dict)
            for item in payload
        )
    ):
        raise ValueError(
            f"Schéma JSON inattendu : {path.name}"
        )

    return pd.DataFrame(
        payload,
        dtype=object
    )


class RawSchema:
    """Schémas minimaux attendus par famille."""

    REQUIRED_COLUMNS = {
        "dpo": {
            "prompt",
            "chosen",
            "rejected",
            "metadata",
            "label_type"
        },
        "fmmcqa": {
            "question",
            "answers",
            "correct_answers"
        },
        "mcqu": {
            "question",
            "correct_answers",
            "answer_a",
            "answer_b",
            "answer_c",
            "answer_d"
        },
        "mcqm": {
            "question",
            "correct_answers",
            "answer_a",
            "answer_b",
            "answer_c",
            "answer_d"
        },
        "oeq": {
            "question",
            "answer"
        },
        "open": {
            "Question",
            "Answer"
        }
    }


def validate_raw_dataset(
    name: str,
    family: str,
    path: Path,
    dataframe: pd.DataFrame,
    raw_row_counts: dict[str, int],
    expected_raw_hashes: dict[str, str]
) -> None:
    """Valide une source brute."""
    expected_rows = raw_row_counts.get(name)
    expected_hash = expected_raw_hashes.get(name)
    required = RawSchema.REQUIRED_COLUMNS.get(family)

    if expected_rows is None or expected_hash is None:
        raise ValueError(
            f"{name} : données absentes du manifeste."
        )

    if required is None:
        raise ValueError(
            f"{name} : famille inconnue {family!r}."
        )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{name} : volume différent de l’audit."
        )

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{name} : colonnes manquantes "
            f"{sorted(missing)}."
        )

    if file_sha256(path) != expected_hash:
        raise RuntimeError(
            f"{name} : hash différent "
            f"de la source auditée."
        )