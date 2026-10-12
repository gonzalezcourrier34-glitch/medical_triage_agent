"""Rejoue le test gelé (500 cas) sur le modèle fusionné et le compare à la référence.

Deux modes :
  hf   : Transformers sur GPU, scoring conditionnel P1/P2/P3, identique à l'évaluation
         du notebook (vérifie que la fusion LoRA n'a pas changé les décisions).
  vllm : requêtes au serveur vLLM (génération, comme en production) + mesure de latence.

Exemples (depuis la racine du projet) :
  uv run python benchmarks/eval_test_gele.py --mode hf
  uv run python benchmarks/eval_test_gele.py --mode vllm --vllm-url http://127.0.0.1:8001/v1
  (serveur distant protégé : définir VLLM_API_KEY avant de lancer le script)

Options d'analyse (mode hf) :
  --split dpo_validation   les 420 cas de validation DPO au lieu du test gelé (ou dpo_train)
  --adapter <dossier>      base Qwen/Qwen3-1.7B (BF16) + adapter LoRA, au lieu du modèle fusionné
  ex. SFT seul : --split dpo_validation --adapter artifacts/finetuning_sft_dpo/models/
                 20261009-154320/sft_checkpoints/checkpoint-450

Résultats : artifacts/finetuning_sft_dpo/metrics/20261009-154320/merged_eval/<mode>[_<split>][_<adapter>]/
"""

import argparse
import json
import os
import statistics
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # permet d'importer scr/ depuis benchmarks/

from src.api.services.prompting import (  # noqa: E402
    INPUT_LABELS,
    SYSTEM_PROMPTS,
    parse_completion,
)

RUN_DIR = ROOT / "artifacts/finetuning_sft_dpo/metrics/20261009-154320"
SPLITS = {
    "test": ROOT / "data/raw/datasets/splits/sft_test.jsonl",
    "dpo_validation": ROOT / "data/raw/datasets/splits/dpo_validation.jsonl",
    "dpo_train": ROOT / "data/raw/datasets/splits/dpo_train.jsonl",
}
BASE_MODEL = "Qwen/Qwen3-1.7B"  # base réelle de unsloth/Qwen3-1.7B-unsloth-bnb-4bit
MODEL_PATH = ROOT / "artifacts/deployment/qwen3-triage-20261009-154320"
REFERENCE = RUN_DIR / "dpo_predictions.jsonl"
LABELS = (1, 2, 3)
MAX_LENGTH = 2_048


# ------------------------------------------------------------------ données


def read_jsonl(path: Path) -> list[dict]:
    """Lit un fichier JSONL."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def expected_priority(row: dict) -> int:
    """Priorité de référence (SFT : priority ; paires DPO : chosen_priority)."""
    return int(row["priority"] if "priority" in row else row["chosen_priority"])


def messages(row: dict) -> list[dict]:
    """Conversation identique à l'entraînement (prompt système + cas clinique)."""
    context = str(row["model_context"]).strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[row["language"]]},
        {"role": "user", "content": f"{INPUT_LABELS[row['language']]}:\n{context}"},
    ]


# ------------------------------------------------------------------ mode hf


def predict_hf(rows: list[dict], adapter: Path | None = None) -> list[dict]:
    """Scoring conditionnel des trois priorités, comme dans le notebook 05."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = adapter or MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL if adapter else MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.eval()

    def tokenize(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    targets = {p: tokenize(f"PRIORITY: P{p}") for p in LABELS}
    results = []

    for index, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(
            messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenize(prompt)
        scores = {}

        for priority, target in targets.items():
            ids = prompt_ids[-(MAX_LENGTH - len(target)) :] + target
            start = len(ids) - len(target)
            input_ids = torch.tensor([ids], device=model.device)

            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits[0, start - 1 : len(ids) - 1].float()

            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(target, device=model.device)
            position = torch.arange(len(target), device=model.device)
            scores[priority] = log_probs[position, target_ids].mean().item()

        predicted = max(scores, key=scores.get)
        results.append(
            {
                "id": row["id"],
                "language": row["language"],
                "expected": expected_priority(row),
                "predicted": predicted,
                **{f"score_p{p}": scores[p] for p in LABELS},
            }
        )

        if index % 50 == 0:
            print(f"  {index}/{len(rows)} cas")

    return results


# ------------------------------------------------------------------ mode vllm


def predict_vllm(rows: list[dict], url: str, model: str) -> list[dict]:
    """Génération via vLLM, avec mesure de la latence de chaque requête."""
    import httpx

    results = []
    key = os.getenv("VLLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    with httpx.Client(timeout=120, headers=headers) as client:
        client.get(f"{url}/models").raise_for_status()

        for index, row in enumerate(rows, start=1):
            payload = {
                "model": model,
                "messages": messages(row),
                "temperature": 0,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
                "stop_token_ids": [151645, 151643],
            }
            start = time.perf_counter()
            response = client.post(f"{url}/chat/completions", json=payload)
            latency_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            body = response.json()

            try:
                parsed = parse_completion(body["choices"][0]["message"]["content"])
                predicted = int(parsed["priority"][1])
            except ValueError:
                predicted = 0  # réponse sans priorité valide : comptée comme erreur

            results.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "expected": expected_priority(row),
                    "predicted": predicted,
                    "latency_ms": round(latency_ms, 2),
                    "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                }
            )

            if index % 50 == 0:
                print(f"  {index}/{len(rows)} cas")

    return results


# ------------------------------------------------------------------ métriques


def metrics(results: list[dict]) -> dict:
    """Accuracy, F1 macro, sous-triage, matrice de confusion, scores par langue."""

    def block(rows: list[dict]) -> dict:
        n = len(rows)
        f1s, per_class = [], {}

        for c in LABELS:
            tp = sum(r["expected"] == c and r["predicted"] == c for r in rows)
            fp = sum(r["expected"] != c and r["predicted"] == c for r in rows)
            fn = sum(r["expected"] == c and r["predicted"] != c for r in rows)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1s.append(f1)
            per_class[f"P{c}"] = {"precision": precision, "recall": recall, "f1": f1}

        under = sum(
            (r["expected"] == 1 and r["predicted"] in (2, 3))
            or (r["expected"] == 2 and r["predicted"] == 3)
            for r in rows
        )
        return {
            "rows": n,
            "accuracy": sum(r["expected"] == r["predicted"] for r in rows) / n,
            "f1_macro": sum(f1s) / 3,
            "undertriage_rate": under / n,
            "invalid_outputs": sum(r["predicted"] == 0 for r in rows),
            "per_class": per_class,
        }

    output = block(results)
    output["confusion"] = {
        f"P{e}": {
            f"P{p}": sum(r["expected"] == e and r["predicted"] == p for r in results)
            for p in LABELS
        }
        for e in LABELS
    }
    for language in ("fr", "en"):
        output[language] = block([r for r in results if r["language"] == language])

    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        ordered = sorted(latencies)
        output["latency_ms"] = {
            "mean": statistics.mean(ordered),
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[int(len(ordered) * 0.95) - 1],
            "max": ordered[-1],
            "requests_per_second": 1000 / statistics.mean(ordered),
        }
    return output


def compare(results: list[dict]) -> dict | None:
    """Compare chaque décision à la référence (adapter évalué pendant l'entraînement)."""
    reference = {r["id"]: r for r in read_jsonl(REFERENCE)}
    if not all(r["id"] in reference for r in results):
        return None  # pas de référence pour ce jeu de données
    pairs = [(r, reference[r["id"]]) for r in results]
    same = sum(r["predicted"] == ref["predicted"] for r, ref in pairs)
    new_only = sum(r["predicted"] == r["expected"] != ref["predicted"] for r, ref in pairs)
    ref_only = sum(ref["predicted"] == ref["expected"] != r["predicted"] for r, ref in pairs)
    n, k = new_only + ref_only, min(new_only, ref_only)
    p_value = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {
        "same_decision": same,
        "rows": len(pairs),
        "reference_correct": sum(ref["predicted"] == ref["expected"] for _, ref in pairs),
        "merged_only_correct": new_only,
        "reference_only_correct": ref_only,
        "mcnemar_exact_p": p_value,
    }


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", default="triage-qwen3")
    parser.add_argument("--limit", type=int, default=None, help="Nombre de cas (test rapide)")
    parser.add_argument("--split", choices=tuple(SPLITS), default="test")
    parser.add_argument("--adapter", type=Path, default=None, help="Adapter LoRA (mode hf)")
    args = parser.parse_args()

    rows = read_jsonl(SPLITS[args.split])[: args.limit]
    print(f"{args.split} : {len(rows)} cas · mode {args.mode}")

    if args.mode == "hf":
        results = predict_hf(rows, args.adapter)
    else:
        results = predict_vllm(rows, args.vllm_url.rstrip("/"), args.vllm_model)

    served = str(args.adapter or MODEL_PATH) if args.mode == "hf" else args.vllm_url
    report = {"mode": args.mode, "model": served, **metrics(results)}
    report["comparison_with_reference"] = compare(results)

    name = args.mode
    if args.split != "test":
        name += f"_{args.split}"
    if args.adapter:
        name += f"_{args.adapter.name}"
    out_dir = RUN_DIR / "merged_eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    cmp = report["comparison_with_reference"]
    p1 = report["per_class"]["P1"]["recall"]
    print(
        f"\nAccuracy {report['accuracy']:.2%} · F1 macro {report['f1_macro']:.2%} · "
        f"rappel P1 {p1:.2%} · sous-triage {report['undertriage_rate']:.2%} · "
        f"FR {report['fr']['accuracy']:.2%} · EN {report['en']['accuracy']:.2%}"
    )
    if cmp:
        print(
            f"Mêmes décisions que la référence : {cmp['same_decision']}/{cmp['rows']} "
            f"(référence {cmp['reference_correct']}/{cmp['rows']} justes, McNemar p = "
            f"{cmp['mcnemar_exact_p']:.3f})"
        )
    if "latency_ms" in report:
        lat = report["latency_ms"]
        print(
            f"Latence : moyenne {lat['mean']:.0f} ms · P50 {lat['p50']:.0f} ms · "
            f"P95 {lat['p95']:.0f} ms"
        )
    print(f"\nRésultats : {out_dir}")


if __name__ == "__main__":
    main()
