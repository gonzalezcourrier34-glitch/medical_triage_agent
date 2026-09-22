from dataclasses import asdict
import json
from pathlib import Path


# Normalise les objets techniques
# avant leur sérialisation JSON.
def json_safe(
    value
):
    """Normalise une valeur pour JSON."""
    if hasattr(
        value,
        "__dataclass_fields__"
    ):
        value = asdict(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, set):
        return sorted(
            json_safe(item)
            for item in value
        )

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    return value


# Lecture

def read_json(
    path: Path
) -> dict | list:
    """Charge un fichier JSON."""
    with Path(path).open(
        encoding="utf-8-sig"
    ) as stream:
        return json.load(stream)


def read_jsonl(
    path: Path
) -> list[dict]:
    """Charge un fichier JSONL."""
    path = Path(path)

    if not path.exists():
        return []

    with path.open(
        encoding="utf-8-sig"
    ) as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


# Écriture

def write_json(
    path: Path,
    value
) -> None:
    """Écrit un fichier JSON."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as stream:
        json.dump(
            json_safe(value),
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True
        )


def write_jsonl(
    path: Path,
    rows
) -> None:
    """Écrit un fichier JSONL."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    json_safe(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True
                )
                + "\n"
            )