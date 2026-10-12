"""Aligne les paires DPO sur le protocole FRENCH utilisé pour le SFT.

Les priorités des paires DPO n'ont pas été fixées par la même chaîne que le SFT.
Ce script ne garde que les paires dont la priorité est confirmée :

1. par la décision FRENCH automatique, quand elle existe (sinon la paire est exclue
   si FRENCH la contredit) ;
2. à défaut, par le SFT checkpoint-450 (entraîné sur les étiquettes FRENCH du SFT),
   s'il prédit la même priorité avec une marge suffisante.

Les enregistrements sont recopiés à l'identique (contexte, métadonnées, chosen, rejected).
Deux blocs sont ajoutés dans metadata :
  - triage_final : la décision de priorité, au même format que rich_reference.triage_final
    du SFT (priority, decision_method, confidence, confidence_score, rule_ids,
    reference_french_levels, reason), plus review_status ;
  - dpo_alignment : le détail du filtrage (étiquette d'origine, prédiction du SFT, marge).

review_status vaut « french_confirmed » quand le protocole a tranché, et
« pending_french_review » quand la priorité n'est confirmée que par le SFT : ces paires
doivent encore être revues selon FRENCH pour suivre entièrement le protocole du SFT.

Emplacements (data/raw/datasets) :
  splits/old_dpo_{train,validation}_garde.jsonl  paires DPO d'origine (entrée)
  dpo_pipeline/aligned/                           sortie de dpo_alignment.py
  splits/dpo_{train,validation}.jsonl             splits DPO finaux (notebook 06)
  dpo.jsonl                                       les 2100 paires finales en un fichier
  dpo_pipeline/reports/                           rapports de chaque étape

Ordre : dpo_alignment.py -> dpo_balance.py -> dpo_chosen_rewrite.py -> dpo_resplit.py

Exemple (depuis la racine du projet) :
  uv run python scr/data/dpo_alignment.py --min-margin 0.25
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/raw/datasets/splits"
SOURCE = SPLITS / "old_dpo_{split}_garde.jsonl"  # paires DPO d'origine
DECISIONS = ROOT / "data/raw/datasets/dpo_french_decisions.jsonl"
PREDICTIONS = ROOT / "artifacts/finetuning_sft_dpo/metrics/20261009-154320/merged_eval"
OUT_DIR = ROOT / "data/raw/datasets/dpo_pipeline/aligned"
SFT_SOURCE = "sft_20261009-154320_checkpoint-450"


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


def margin(prediction: dict) -> float:
    """Écart de score entre la priorité prédite et la suivante."""
    scores = sorted((prediction[f"score_p{p}"] for p in (1, 2, 3)), reverse=True)
    return scores[0] - scores[1]


def decide(row: dict, french: dict, prediction: dict | None, min_margin: float) -> dict:
    """Décision de conservation d'une paire, avec sa justification."""
    label = priority(row["chosen_priority"])
    info = {
        "label": label,
        "french_priority": french.get("french_priority"),
        "french_method": french.get("french_method"),
        "french_confidence": french.get("french_confidence"),
        "sft_source": SFT_SOURCE,
        "sft_predicted": prediction["predicted"] if prediction else None,
        "sft_margin": round(margin(prediction), 4) if prediction else None,
        "min_margin": min_margin,
    }

    if french.get("french_priority") is not None:
        confirmed = french["french_priority"] == label
        info["rule"] = "french_decision"
        info["status"] = (
            "kept_french_confirmed" if confirmed else "excluded_french_contradicts"
        )
    elif prediction is None:
        info["rule"] = "missing_prediction"
        info["status"] = "excluded_no_arbiter"
    elif prediction["predicted"] != label:
        info["rule"] = "sft_arbiter"
        info["status"] = "excluded_sft_disagrees"
    elif info["sft_margin"] < min_margin:
        info["rule"] = "sft_arbiter"
        info["status"] = "excluded_sft_low_margin"
    else:
        info["rule"] = "sft_arbiter"
        info["status"] = "kept_sft_confirmed"

    info["kept"] = info["status"].startswith("kept")
    return info


CONFIDENCE_SCORE = {"high": 1.0, "medium": 0.75, "low": 0.5}


def triage_final(info: dict, french: dict) -> dict:
    """Décision au format rich_reference.triage_final du SFT."""
    if info["rule"] == "french_decision":
        confidence = french.get("french_confidence") or "low"
        return {
            "priority": info["label"],
            "decision_method": french.get("french_method"),
            "confidence": confidence,
            "confidence_score": CONFIDENCE_SCORE.get(confidence, 0.5),
            "rule_ids": french.get("french_rule_ids", []),
            "reference_french_levels": french.get("french_levels", []),
            "reason": "Priorité confirmée par la décision FRENCH du pipeline DPO.",
            "review_status": "french_confirmed",
        }
    return {
        "priority": info["label"],
        "decision_method": "sft_model_arbitration",
        "confidence": "medium" if (info["sft_margin"] or 0) >= 0.5 else "low",
        "confidence_score": info["sft_margin"],
        "rule_ids": [],
        "reference_french_levels": [],
        "reason": (
            "Aucune décision FRENCH automatique ; priorité confirmée par le SFT "
            f"({SFT_SOURCE}), à revoir selon le protocole."
        ),
        "review_status": "pending_french_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-margin", type=float, default=0.25)
    args = parser.parse_args()

    french = {r["id"]: r for r in read_jsonl(DECISIONS)}
    report = {"min_margin": args.min_margin, "splits": {}}

    for split in ("train", "validation"):
        rows = read_jsonl(Path(str(SOURCE).format(split=split)))
        pred_path = PREDICTIONS / f"hf_dpo_{split}_checkpoint-450" / "predictions.jsonl"
        predictions = {r["id"]: r for r in read_jsonl(pred_path)}

        kept, excluded = [], []
        for row in rows:
            info = decide(
                row,
                french.get(row["id"], {}),
                predictions.get(row["id"]),
                args.min_margin,
            )
            row["chosen_priority"] = row["rejected_priority"] = info["label"]
            metadata = {**row.get("metadata", {}), "dpo_alignment": info}
            if info["kept"]:
                metadata["triage_final"] = triage_final(info, french.get(row["id"], {}))
            row["metadata"] = metadata
            (kept if info["kept"] else excluded).append(row)

        write_jsonl(OUT_DIR / f"dpo_{split}.jsonl", kept)
        write_jsonl(OUT_DIR / f"dpo_{split}_excluded.jsonl", excluded)

        report["splits"][split] = {
            "input": len(rows),
            "kept": len(kept),
            "status": dict(
                Counter(r["metadata"]["dpo_alignment"]["status"] for r in rows)
            ),
            "kept_by_review_status": dict(
                Counter(r["metadata"]["triage_final"]["review_status"] for r in kept)
            ),
            "kept_by_language_priority": dict(
                Counter(f"{r['language']}_P{r['chosen_priority']}" for r in kept)
            ),
        }

    (OUT_DIR / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
