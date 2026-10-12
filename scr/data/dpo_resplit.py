"""Refait le découpage train / validation des splits DPO finaux.

Règles :
  1. Toute paire construite sur un cas du train SFT (review_status = sft_protocol)
     va en train : le modèle a déjà vu ces cas, ils ne peuvent pas servir à valider.
  2. Une paire alignée quasi identique à un cas du train SFT (>= 60 % de 5-grammes
     communs dans model_context) va aussi en train.
  3. Les groupes de fuite (leakage_group, qui relient notamment la version anglaise
     et la version française d'un même cas) ne sont jamais coupés entre deux splits.
  4. La validation est tirée parmi les groupes restants pour obtenir la même taille
     dans chaque cellule langue x priorité (20 % de 350 = 70 par défaut).

Les 2100 paires et leur contenu ne changent pas : seuls final_split et le suivi
(metadata.tracking) sont mis à jour.

Usage :
    uv run python scr/data/dpo_resplit.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "raw" / "datasets" / "splits"
OUT_DIR = SPLITS  # splits/dpo_{train,validation}.jsonl
PIPELINE = ROOT / "data" / "raw" / "datasets" / "dpo_pipeline"
FULL_DPO = ROOT / "data" / "raw" / "datasets" / "dpo.jsonl"
CELLS = [(lang, p) for lang in ("en", "fr") for p in (1, 2, 3)]
VERSION = "dpo_final_v2"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def priority(value) -> int:
    return int(str(value).upper().replace("P", "").strip())


def cell(row: dict) -> tuple[str, int]:
    return row["language"], priority(row["chosen_priority"])


def shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def near_duplicates_of_sft_train(rows: list[dict], threshold: float) -> set[str]:
    """Ids des paires dont le contexte recopie un cas du train SFT."""
    train = [shingles(r["model_context"]) for r in read_jsonl(SPLITS / "sft_train.jsonl")]
    index: dict[str, list[int]] = defaultdict(list)
    for i, grams in enumerate(train):
        for gram in grams:
            index[gram].append(i)
    found = set()
    for row in rows:
        grams = shingles(row["model_context"])
        if not grams:
            continue
        hits: Counter = Counter()
        for gram in grams:
            hits.update(index.get(gram, ()))
        if any(n / min(len(grams), len(train[i])) >= threshold for i, n in hits.items()):
            found.add(row["id"])
    return found


def pick_validation(groups: dict, target: int, seed: int, tries: int) -> set[str]:
    """Choisit des groupes entiers pour atteindre `target` paires par cellule.

    Les cellules les plus rares passent en premier (sinon un groupe bilingue peut
    remplir la cellule voisine et bloquer la cellule rare). À rareté égale, les
    groupes bilingues sont pris avant les cas isolés ; l'ordre restant est tiré au sort.
    """
    available: Counter = Counter(cell(r) for rows in groups.values() for r in rows)

    def scarcity(key: str) -> float:
        return min(available[cell(r)] / target for r in groups[key])

    best, best_gap = set(), None
    for attempt in range(tries):
        rng = random.Random(seed + attempt)
        order = sorted(
            groups, key=lambda k: (round(scarcity(k), 3), -len(groups[k]), rng.random())
        )
        count: Counter = Counter()
        chosen = set()
        for key in order:
            need = Counter(cell(r) for r in groups[key])
            if all(count[c] + n <= target for c, n in need.items()):
                count.update(need)
                chosen.add(key)
        gap = sum(target - count[c] for c in CELLS)
        if best_gap is None or gap < best_gap:
            best, best_gap = chosen, gap
        if gap == 0:
            break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=350)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--near-dup", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tries", type=int, default=5000)
    args = parser.parse_args()

    rows = sorted(
        read_jsonl(OUT_DIR / "dpo_train.jsonl")
        + read_jsonl(OUT_DIR / "dpo_validation.jsonl"),
        key=lambda r: r["id"],
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["leakage_group"] or row["id"]].append(row)

    aligned = [r for r in rows if r["metadata"]["triage_final"]["review_status"] != "sft_protocol"]
    near_dup = near_duplicates_of_sft_train(aligned, args.near_dup)
    reasons: dict[str, str] = {}
    for key, members in groups.items():
        if any(r["metadata"]["triage_final"]["review_status"] == "sft_protocol" for r in members):
            reasons[key] = "sft_train_origin"
        elif any(r["id"] in near_dup for r in members):
            reasons[key] = "near_duplicate_sft_train"

    target = round(args.per_cell * args.validation_ratio)
    free = {k: v for k, v in groups.items() if k not in reasons}
    validation_keys = pick_validation(free, target, args.seed, args.tries)

    out = {"train": [], "validation": []}
    for key, members in groups.items():
        split = "validation" if key in validation_keys else "train"
        reason = reasons.get(key) or (
            "selected_validation" if split == "validation" else "aligned_train"
        )
        for row in members:
            row["final_split"] = split
            tracking = row["metadata"].setdefault("tracking", {})
            block = tracking.setdefault("dpo_final", {})
            if block.get("version") != VERSION:
                # Premier passage : on garde le split de dpo_final_v1 comme précédent.
                previous = tracking.get("dataset_split")
                tracking.pop("split_precedent", None)
                if previous and previous != split:
                    tracking["split_precedent"] = previous
            tracking["dataset_split"] = split
            block["version"] = VERSION
            block["scripts"] = [
                "scr/data/dpo_alignment.py",
                "scr/data/dpo_balance.py",
                "scr/data/dpo_chosen_rewrite.py",
                "scr/data/dpo_resplit.py",
            ]
            block.setdefault("parameters", {}).update(
                {
                    "validation_per_cell": target,
                    "near_duplicate_threshold": args.near_dup,
                    "resplit_seed": args.seed,
                }
            )
            block["split_reason"] = reason
            out[split].append(row)

    # Contrôles.
    assert not {r["leakage_group"] for r in out["train"]} & {
        r["leakage_group"] for r in out["validation"]
    }, "groupe de fuite partagé"
    assert not any(
        r["metadata"]["triage_final"]["review_status"] == "sft_protocol"
        for r in out["validation"]
    ), "paire issue du train SFT en validation"
    total = Counter(cell(r) for r in rows)
    assert all(total[c] == args.per_cell for c in CELLS), total

    rng = random.Random(args.seed)
    report: dict = {"version": VERSION, "validation_target_per_cell": target}
    for split, split_rows in out.items():
        rng.shuffle(split_rows)
        write_jsonl(OUT_DIR / f"dpo_{split}.jsonl", split_rows)
        report[split] = {
            "rows": len(split_rows),
            "cells": {
                f"{lang}_P{p}": n
                for (lang, p), n in sorted(Counter(cell(r) for r in split_rows).items())
            },
            "review_status": dict(
                Counter(r["metadata"]["triage_final"]["review_status"] for r in split_rows)
            ),
            "split_reason": dict(
                Counter(r["metadata"]["tracking"]["dpo_final"]["split_reason"] for r in split_rows)
            ),
            "bilingual_groups": sum(
                len({r["language"] for r in groups[k]}) > 1
                for k in {r["leakage_group"] or r["id"] for r in split_rows}
            ),
        }
    report["near_duplicates_of_sft_train"] = sorted(near_dup)
    write_jsonl(FULL_DPO, out["train"] + out["validation"])
    (PIPELINE / "reports" / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "near_duplicates_of_sft_train"},
                     ensure_ascii=False, indent=2))
    print("quasi-doublons du train SFT :", len(near_dup))


if __name__ == "__main__":
    main()
