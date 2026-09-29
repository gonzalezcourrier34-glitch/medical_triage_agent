import hashlib
import json
from pathlib import Path

from scr.utils.serialization import json_safe


# Hash des données
# Sérialise d'abord la valeur de façon stable pour garantir le même SHA-256 à contenu identique.

def digest(value) -> str:
    """Calcule le SHA-256 déterministe d'une valeur."""
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,      # Conserve correctement les caractères Unicode
        sort_keys=True,          # Rend l'ordre des clés sans effet sur le hash
        separators=(",", ":"),   # Supprime les espaces variables du JSON
        allow_nan=False          # Refuse les valeurs JSON non standards
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


# Hash des fichiers
# Lit le fichier par blocs pour éviter de charger un dataset entier en RAM.

def file_sha256(path: Path) -> str:
    """Calcule le SHA-256 d'un fichier."""
    file_hash = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # Blocs de 1 Mio
            file_hash.update(block)

    return file_hash.hexdigest()