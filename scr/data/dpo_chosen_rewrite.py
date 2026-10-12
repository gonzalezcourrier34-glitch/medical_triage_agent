"""Réinjecte les explications « chosen » réécrites dans splits/dpo_{train,validation}.jsonl.

Les chosen d'origine reprenaient soit l'explication cible du SFT mot pour mot
(paires sft_protocol), soit des formules toutes faites (« impose une prise en charge
immédiate », « dans ce scénario pédagogique »...). Chaque chosen a été réécrit à partir du seul model_context :
deux phrases maximum, dans la langue de l'enregistrement, ton bienveillant,
propre au patient, sans nommer la priorité ni reprendre les phrases types.

Le script est idempotent : l'explication d'origine est conservée dans
metadata.chosen_rewrite.original et n'est jamais écrasée.

Usage :
    uv run python scr/data/dpo_chosen_rewrite.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "raw" / "datasets" / "splits"
PIPELINE = ROOT / "data" / "raw" / "datasets" / "dpo_pipeline"
# v1 : les 1193 paires construites sur le train SFT ; v2 : les 907 paires alignées.
REWRITE_FILES = {
    "chosen_rewrite_v1": PIPELINE / "chosen_rewrites_v1.jsonl",
    "chosen_rewrite_v2": PIPELINE / "chosen_rewrites_v2.jsonl",
}
REPORT_DIR = PIPELINE / "reports"
RULE = (
    "Réécrit depuis model_context uniquement ; 2 phrases max ; langue de "
    "l'enregistrement ; ton bienveillant et individualisé ; sans mention de "
    "priorité ni phrase type de triage."
)
FORBIDDEN = re.compile(r"\bP[123]\b|priorit|triage", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_count(text: str) -> int:
    text = text.replace("E. coli", "E coli")
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def main() -> None:
    rewrites: dict[str, tuple[str, str]] = {}
    for version, path in REWRITE_FILES.items():
        for row in read_jsonl(path):
            if row["id"] in rewrites:
                raise ValueError(f"Réécriture en double : {row['id']}")
            rewrites[row["id"]] = (row["chosen"], version)
    used: set[str] = set()
    report: dict = {"rewrite_files": {v: p.name for v, p in REWRITE_FILES.items()}}

    for split in ("train", "validation"):
        path = SPLIT_DIR / f"dpo_{split}.jsonl"
        rows = read_jsonl(path)
        changed = 0
        for row in rows:
            if row["id"] not in rewrites:
                continue
            new, version = rewrites[row["id"]]
            used.add(row["id"])
            meta = row["metadata"]
            block = meta.get("chosen_rewrite") or {}
            original = block.get("original", row["chosen_explanation"])
            if FORBIDDEN.search(new) or sentence_count(new) > 2:
                raise ValueError(f"Réécriture non conforme : {row['id']}")
            if new.strip() == row["rejected_explanation"].strip():
                raise ValueError(f"chosen == rejected : {row['id']}")
            row["chosen_explanation"] = new
            meta["chosen_rewrite"] = {
                "version": version,
                "original": original,
                "rule": RULE,
            }
            tracking = meta.setdefault("tracking", {})
            tracking.setdefault("dpo_final", {})["chosen_rewrite"] = version
            changed += 1
        write_jsonl(path, rows)

        cells = Counter(
            (r["language"], int(str(r["chosen_priority"]).lstrip("Pp"))) for r in rows
        )
        report[split] = {
            "rows": len(rows),
            "chosen_rewritten": changed,
            "chosen_equals_rejected": sum(
                r["chosen_explanation"].strip() == r["rejected_explanation"].strip()
                for r in rows
            ),
            "cells": {f"{lang}_P{p}": n for (lang, p), n in sorted(cells.items())},
        }

    missing = sorted(set(rewrites) - used)
    if missing:
        raise ValueError(f"{len(missing)} réécritures sans paire correspondante")
    report["total_rewritten"] = len(used)
    (REPORT_DIR / "chosen_rewrite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
