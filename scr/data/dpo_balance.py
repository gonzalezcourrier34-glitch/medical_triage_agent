"""Complète les paires DPO alignées avec des paires issues du train SFT.

Les paires alignées (scr/data/dpo_alignment.py) sont gardées telles quelles. Pour chaque
cellule langue x priorité, on complète jusqu'à la cible avec des paires construites sur le
train SFT, dont la priorité suit par construction le protocole du SFT :

  chosen   = explication finale du SFT (révisée, spécifique au cas) ;
  rejected = explication antérieure du même cas (final_explanation du corpus maître),
             écrite pour la même priorité, plus générique.

Les cas de sft_validation et sft_test ne sont jamais utilisés. Le découpage 80/20 des
nouvelles paires se fait par groupe de fuite (leakage_group).

Étape suivante : scr/data/dpo_chosen_rewrite.py puis scr/data/dpo_resplit.py.

Exemple (depuis la racine du projet) :
  uv run python scr/data/dpo_balance.py --per-cell 350
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/raw/datasets/splits"
PIPELINE = ROOT / "data/raw/datasets/dpo_pipeline"
ALIGNED = PIPELINE / "aligned"
MASTER = ROOT / "artifacts/sft_dataset_creation/triage_master_18523.jsonl"
OUT_DIR = SPLITS  # écrit splits/dpo_{train,validation}.jsonl
REPORT_DIR = PIPELINE / "reports"
CELLS = [(lang, p) for lang in ("fr", "en") for p in (1, 2, 3)]


def read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Écrit un fichier JSONL UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def priority(value) -> int:
    """Normalise une priorité (1, "1" ou "P1") en entier."""
    return int(str(value).upper().replace("P", "").strip())


def normalized(text: str) -> str:
    """Texte normalisé pour comparer deux explications."""
    return " ".join(str(text).split())


def sft_pair(row: dict, old_explanation: str) -> dict:
    """Paire DPO au format des splits, construite sur un cas du train SFT."""
    pair_id = hashlib.sha1(f"dpo-sft:{row['id']}".encode()).hexdigest()[:24]
    metadata = dict(row.get("metadata", {}))
    metadata["triage_final"] = {
        "priority": row["priority"],
        "decision_method": "sft_protocol_label",
        "confidence": "high",
        "confidence_score": 1.0,
        "rule_ids": [],
        "reference_french_levels": [],
        "reason": "Priorité du dataset SFT (même protocole que l'entraînement du modèle).",
        "review_status": "sft_protocol",
    }
    metadata["dpo_pair_source"] = {
        "origin": "sft_train",
        "sft_id": row["id"],
        "chosen": "sft_final_explanation",
        "rejected": "master_final_explanation_pre_v2",
    }
    return {
        "id": pair_id,
        "language": row["language"],
        "source_split": "sft_train",
        "model_context": row["model_context"],
        "context_hash": row["context_hash"],
        "metadata": metadata,
        "final_split": None,
        "leakage_group": row.get("leakage_group"),
        "sft_input": row.get("sft_input"),
        "sft_input_hash": row.get("sft_input_hash"),
        "chosen_priority": row["priority"],
        "chosen_explanation": row["explanation"],
        "rejected_priority": row["priority"],
        "rejected_explanation": normalized(old_explanation),
    }


def candidates() -> dict[tuple, list[dict]]:
    """Paires possibles depuis le train SFT, par cellule langue x priorité."""
    train = {r["id"]: r for r in read_jsonl(SPLITS / "sft_train.jsonl")}
    by_cell = defaultdict(list)
    with MASTER.open(encoding="utf-8") as file:
        for line in file:
            master = json.loads(line)
            row = train.get(master.get("id"))
            old = master.get("final_explanation")
            if row is None or not isinstance(old, str) or not old.strip():
                continue
            if priority(master["final_priority"]) != row["priority"]:
                continue  # l'ancienne explication défendait une autre priorité
            if normalized(old) == normalized(row["explanation"]):
                continue
            by_cell[(row["language"], row["priority"])].append(sft_pair(row, old))
    return by_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=350)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    aligned = {
        s: read_jsonl(ALIGNED / f"dpo_{s}.jsonl") for s in ("train", "validation")
    }
    for split, rows in aligned.items():
        for row in rows:
            row["final_split"] = split
            row["metadata"]["dpo_pair_source"] = {"origin": "dpo_aligned"}

    held_out = {
        r["context_hash"]
        for s in ("sft_validation", "sft_test")
        for r in read_jsonl(SPLITS / f"{s}.jsonl")
    }
    pool = candidates()
    out = {"train": list(aligned["train"]), "validation": list(aligned["validation"])}
    report = {"per_cell": args.per_cell, "cells": {}}

    for cell in CELLS:
        have = [
            r
            for s in out.values()
            for r in s
            if (r["language"], r["chosen_priority"]) == cell
        ]
        need = max(0, args.per_cell - len(have))
        pairs = [p for p in pool.get(cell, []) if p["context_hash"] not in held_out]
        rng.shuffle(pairs)

        # Découpage par groupe de fuite : un groupe entier va en train ou en validation.
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair["leakage_group"] or pair["id"]].append(pair)
        added = {"train": 0, "validation": 0}
        target_val = round(need * args.validation_ratio)
        for group in groups.values():
            if sum(added.values()) >= need:
                break
            split = "validation" if added["validation"] < target_val else "train"
            for pair in group[: need - sum(added.values())]:
                pair["final_split"] = split
                out[split].append(pair)
                added[split] += 1

        report["cells"][f"{cell[0]}_P{cell[1]}"] = {
            "aligned": len(have),
            "available_from_sft": len(pairs),
            "added_train": added["train"],
            "added_validation": added["validation"],
            "total": len(have) + sum(added.values()),
        }

    # Un groupe de fuite présent en train ne doit pas apparaître en validation.
    train_groups = {r["leakage_group"] for r in out["train"] if r["leakage_group"]}
    moved = [r for r in out["validation"] if r["leakage_group"] in train_groups]
    for row in moved:
        row["final_split"] = "train"
    out["train"] += moved
    out["validation"] = [r for r in out["validation"] if r not in moved]
    report["moved_to_train_for_leakage"] = len(moved)

    # Suivi homogène : chaque paire porte un bloc tracking cohérent avec son split.
    for split, rows in out.items():
        for row in rows:
            tracking = dict(row["metadata"].get("tracking") or {})
            previous = tracking.get("dataset_split")
            tracking["dataset_split"] = split
            if previous and previous != split:
                tracking["split_precedent"] = previous
            tracking["dpo_final"] = {
                "version": "dpo_final_v1",
                "scripts": ["scr/data/dpo_alignment.py", "scr/data/dpo_balance.py"],
                "parameters": {
                    "min_margin": 0.25,
                    "per_cell": args.per_cell,
                    "seed": args.seed,
                },
                "origin": row["metadata"]["dpo_pair_source"]["origin"],
                "review_status": row["metadata"]["triage_final"]["review_status"],
                "priority_protocol": "FRENCH 2018-v1.2 / CHSA_3_LEVELS",
            }
            row["metadata"]["tracking"] = tracking

    for split, rows in out.items():
        rng.shuffle(rows)
        write_jsonl(OUT_DIR / f"dpo_{split}.jsonl", rows)

    report["totals"] = {
        split: {
            "rows": len(rows),
            "by_source": dict(
                Counter(r["metadata"]["dpo_pair_source"]["origin"] for r in rows)
            ),
            "by_review_status": dict(
                Counter(r["metadata"]["triage_final"]["review_status"] for r in rows)
            ),
        }
        for split, rows in out.items()
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
