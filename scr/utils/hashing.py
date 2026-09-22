import hashlib
import json
from pathlib import Path

from scr.utils.serialization import (
    json_safe
)


def digest(
    value
) -> str:
    """Calcule une empreinte SHA-256 déterministe."""
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def file_sha256(
    path: Path
) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    file_hash = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b""
        ):
            file_hash.update(block)

    return file_hash.hexdigest()