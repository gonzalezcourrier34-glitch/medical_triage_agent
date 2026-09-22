from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import file_sha256, read_json, read_jsonl
from .protocol import (
    explicit_branches,
    motifs_without_explicit_branch,
    unresolved_motifs
)


# ## Configuration CLI

# Définit les trois fichiers nécessaires à l'audit de cohérence.
# Cette interface permet aussi d'exécuter le module hors notebook.
def parse_args() -> argparse.Namespace:
    """Lit les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Audite les trois entrées de la pipeline"
    )

    parser.add_argument(
        "--records",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--routing",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        required=True
    )

    return parser.parse_args()


# ## Audit

# Vérifie la cohérence entre les records enrichis, le routage et le protocole.
# Aucun niveau d'urgence ni nouvelle décision clinique n'est produit ici.
def audit(
    records_path: Path,
    routing_path: Path,
    protocol_path: Path
) -> dict:
    """Audite les entrées de la pipeline TRIAGE."""
    records = read_jsonl(
        records_path
    )

    routing = read_jsonl(
        routing_path
    )

    protocol = read_json(
        protocol_path
    )

    record_by_id = {
        row["id"]: row
        for row in records
    }

    route_by_id = {
        row["id"]: row
        for row in routing
    }

    if len(record_by_id) != len(records):
        raise ValueError(
            "Identifiants dupliqués dans le fichier 12"
        )

    if len(route_by_id) != len(routing):
        raise ValueError(
            "Identifiants dupliqués dans le fichier 13"
        )


    # ## Cohérence des identifiants

    missing_in_12 = sorted(
        set(route_by_id)
        - set(record_by_id)
    )

    missing_in_13 = sorted(
        set(record_by_id)
        - set(route_by_id)
    )


    # ## Cohérence des hash

    # Vérifie que le routage référence exactement le contenu source enrichi.
    # Un hash différent signale une rupture de traçabilité entre 12 et 13.
    hash_mismatches = [
        record_id
        for record_id in (
            set(record_by_id)
            & set(route_by_id)
        )
        if (
            record_by_id[
                record_id
            ].get("clean_content_hash")
            != route_by_id[
                record_id
            ].get("clean_content_hash")
        )
    ]


    # ## Protocole

    branches = explicit_branches(
        protocol
    )

    route_counts = Counter(
        row["triage_route"]
        for row in routing
    )

    language_counts = Counter(
        row.get("language") or "UNKNOWN"
        for row in routing
    )


    # ## Rapport

    return {
        "valid": (
            not missing_in_12
            and not missing_in_13
            and not hash_mismatches
        ),
        "rows": {
            "records": len(records),
            "routing": len(routing)
        },
        "missing_in_12": missing_in_12[:20],
        "missing_in_13": missing_in_13[:20],
        "hash_mismatches": hash_mismatches[:20],
        "route_counts": dict(
            sorted(
                route_counts.items()
            )
        ),
        "language_counts": dict(
            sorted(
                language_counts.items()
            )
        ),
        "protocol": {
            "name": protocol["protocol_name"],
            "version": protocol["protocol_version"],
            "motifs": len(
                protocol["motifs"]
            ),
            "explicit_branches": len(
                branches
            ),
            "automatic_branches": sum(
                branch.automation_status
                == "automatic"
                for branch in branches
            ),
            "motifs_without_explicit_branch": (
                motifs_without_explicit_branch(
                    protocol
                )
            ),
            "unresolved_motifs": unresolved_motifs(
                protocol
            )
        },
        "sha256": {
            "records": file_sha256(
                records_path
            ),
            "routing": file_sha256(
                routing_path
            ),
            "protocol": file_sha256(
                protocol_path
            )
        }
    }


# ## Exécution CLI

# Exécute l'audit et retourne un code d'erreur si la cohérence échoue.
# Le notebook peut appeler directement audit() sans passer par cette fonction.
def main() -> None:
    """Exécute l'audit en ligne de commande."""
    args = parse_args()

    report = audit(
        args.records,
        args.routing,
        args.protocol
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()