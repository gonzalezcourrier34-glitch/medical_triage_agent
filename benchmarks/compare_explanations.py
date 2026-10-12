"""Compare les explications générées par le SFT, le DPO livré et le DPO réaligné.

Le DPO ne change pas les priorités (même priorité dans chosen et rejected) : ce
script mesure donc ce qu'il est censé améliorer, la forme des explications, sur un
échantillon stratifié du test gelé (même cas pour les trois modèles, génération
gloutonne, base Qwen/Qwen3-1.7B BF16 + adapter LoRA, un modèle à la fois en VRAM).

Indicateurs par modèle :
  template_sft   : part reprenant un gabarit du SFT (« Your reported clinical
                   information includes… », « Vous présentez une situation… »)
  stock_phrases  : part contenant une formule toute faite de triage
  priority_named : part citant P1/P2/P3 ou « priorité » dans l'explication
  addresses_you  : part qui s'adresse au patient (vous / you)
  unique_rate    : part d'explications distinctes
  copy_target    : part identique mot pour mot à la cible SFT
  words_mean     : longueur moyenne (mots)

Exemple (depuis la racine du projet) :
  uv run python benchmarks/compare_explanations.py --per-cell 15
Résultats : artifacts/finetuning_sft_dpo/metrics/20261009-154320/explanations_compare/
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.services.prompting import INPUT_LABELS, SYSTEM_PROMPTS  # noqa: E402

RUN = "20261009-154320"
MODELS_DIR = ROOT / f"artifacts/finetuning_sft_dpo/models/{RUN}"
ADAPTERS = {
    "sft": MODELS_DIR / "sft_checkpoints/checkpoint-450",
    "dpo_livre": MODELS_DIR / "dpo_adapter_livre",
    "dpo_realigne": MODELS_DIR / "dpo_aligned_adapter",
}
BASE_MODEL = "Qwen/Qwen3-1.7B"
TEST = ROOT / "data/raw/datasets/splits/sft_test.jsonl"
OUT = ROOT / f"artifacts/finetuning_sft_dpo/metrics/{RUN}/explanations_compare"

TEMPLATES = re.compile(
    r"reported clinical information includes|these findings warrant|"
    r"vous présentez une situation|les éléments décrits évoquent|"
    r"vous êtes pris en charge pour|votre situation nécessite",
    re.IGNORECASE,
)
STOCK = re.compile(
    r"prompt care required|immediate care required|deferred care|workflow|usual order|"
    r"prise en charge (immédiate|rapide|différée)|impose une|urgence (vitale|modérée|"
    r"différée)|délais habituels|organisation du service|scénario|scenario|pédagogique",
    re.IGNORECASE,
)
PRIORITY = re.compile(r"\bP[123]\b|priorit", re.IGNORECASE)
YOU = re.compile(r"\b(vous|votre|vos|you|your)\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sample(rows: list[dict], per_cell: int, seed: int) -> list[dict]:
    """Même nombre de cas par langue x priorité."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["language"], int(row["priority"]))].append(row)
    rng = random.Random(seed)
    picked = []
    for key in sorted(cells):
        picked += rng.sample(cells[key], min(per_cell, len(cells[key])))
    return picked


def explanation(text: str) -> str:
    match = re.search(r"EXPLANATION:\s*(.*)", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def generate(adapter: Path, rows: list[dict], max_new_tokens: int) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
    stop = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    outputs = []
    for i, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[row["language"]]},
            {
                "role": "user",
                "content": f"{INPUT_LABELS[row['language']]}:\n{row['model_context'].strip()}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop
            )
        text = tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        outputs.append(text.strip())
        print(f"\r  {adapter.name}: {i}/{len(rows)}", end="", flush=True)
    print()
    del model
    torch.cuda.empty_cache()
    return outputs


def summarize(rows: list[dict], texts: list[str]) -> dict:
    exps = [explanation(t) for t in texts]
    n = len(exps)
    rate = lambda pattern: round(sum(bool(pattern.search(e)) for e in exps) / n, 3)  # noqa: E731
    return {
        "n": n,
        "template_sft": rate(TEMPLATES),
        "stock_phrases": rate(STOCK),
        "priority_named": rate(PRIORITY),
        "addresses_you": rate(YOU),
        "unique_rate": round(len(set(exps)) / n, 3),
        "copy_target": round(
            sum(e.strip() == r["explanation"].strip() for e, r in zip(exps, rows)) / n, 3
        ),
        "words_mean": round(sum(len(e.split()) for e in exps) / n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--models", nargs="+", default=list(ADAPTERS), choices=list(ADAPTERS))
    args = parser.parse_args()

    rows = sample(read_jsonl(TEST), args.per_cell, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    generations = {name: generate(ADAPTERS[name], rows, args.max_new_tokens) for name in args.models}

    summary = {name: summarize(rows, texts) for name, texts in generations.items()}
    with (OUT / "generations.jsonl").open("w", encoding="utf-8") as file:
        for i, row in enumerate(rows):
            record = {
                "id": row["id"],
                "language": row["language"],
                "priority": row["priority"],
                "context": row["model_context"],
                "sft_target": row["explanation"],
            }
            record.update({name: generations[name][i] for name in generations})
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(
        json.dumps({"sample": len(rows), "models": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
