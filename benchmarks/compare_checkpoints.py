"""Choisit le meilleur checkpoint DPO sur la génération (SFT vs checkpoints DPO).

Le DPO ne change pas les priorités (matrice de confusion identique au SFT) : le choix
du checkpoint se fait donc sur la réponse générée, mêmes cas pour tous les modèles,
génération gloutonne, base BF16 + adapter LoRA (+ jetons spéciaux), un modèle à la fois.

Indicateurs ajoutés à ceux de compare_explanations.py :
  ended         : part des réponses terminées par un jeton de fin (pas de coupure)
  repeated      : part d'explications qui répètent une même phrase (dégénérescence)
  format_ok     : part au format PRIORITY + EXPLANATION
  priority_ok   : part dont la priorité générée est la priorité attendue
  grounding     : part moyenne des mots médicaux de l'explication (≥ 7 lettres) présents
                  dans le contexte patient ; proxy d'hallucination (plus haut = plus ancré)

Exemple (depuis la racine du projet) :
  uv run python benchmarks/compare_checkpoints.py --per-cell 10
Résultats : artifacts/finetuning_sft_dpo/metrics/<RUN>/<DPO_TAG>/checkpoints_compare/
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.compare_explanations import explanation, read_jsonl, sample, summarize  # noqa: E402
from src.api.services.prompting import INPUT_LABELS, SYSTEM_PROMPTS  # noqa: E402

RUN = "20261012-162715"
DPO_TAG = "dpo-ckpt550-20261012-195026"
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
MODELS_DIR = ROOT / f"artifacts/finetuning_sft_dpo/models/{RUN}"
DPO_DIR = MODELS_DIR / f"{DPO_TAG}_checkpoints"
TEST = ROOT / "data/raw/datasets/splits/sft_test.jsonl"
OUT = ROOT / f"artifacts/finetuning_sft_dpo/metrics/{RUN}/{DPO_TAG}/checkpoints_compare"

FORMAT = re.compile(r"^PRIORITY: P[123]\s*\nEXPLANATION:\s*\S")
PRIORITY = re.compile(r"PRIORITY:\s*P([123])")


def candidates() -> dict[str, Path]:
    """SFT source + tous les checkpoints DPO du run."""
    models = {"sft-550": MODELS_DIR / "sft_checkpoints/checkpoint-550"}
    for path in sorted(DPO_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        models[f"dpo-{path.name.split('-')[1]}"] = path
    return models


WORD = re.compile(r"[a-zà-öø-ÿ]{7,}", re.IGNORECASE)
GENERIC = {
    "assessment", "immediate", "emergency", "department", "reported", "clinical", "information",
    "includes", "findings", "warrant", "patient", "requires", "prompt", "deferred", "evaluation",
    "immédiate", "urgences", "nécessite", "rapide", "évaluation", "patiente", "situation",
    "symptômes", "symptoms", "condition", "together", "because", "without", "suggest", "suggests",
    "needs", "scheduled", "consultation", "surveillance", "prise", "charge", "medical", "médicale",
}


def grounding(text: str, context: str) -> float:
    words = [w.lower() for w in WORD.findall(text) if w.lower() not in GENERIC]
    if not words:
        return 1.0
    ctx = context.lower()
    return sum(w[:6] in ctx for w in words) / len(words)


def repeated(text: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]\s+", text) if len(s.split()) >= 4]
    return len(sentences) != len(set(sentences))


def generate(adapter: Path, rows: list[dict], max_new_tokens: int) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
    stop = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop[1]
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
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=stop, pad_token_id=pad,
            )
        gen = out[0, ids["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        outputs.append({"text": text, "ended": any(t in stop for t in gen), "tokens": len(gen)})
        print(f"\r  {adapter.name}: {i}/{len(rows)}", end="", flush=True)
    print()
    del model
    torch.cuda.empty_cache()
    return outputs


def score(rows: list[dict], outs: list[dict]) -> dict:
    n = len(outs)
    texts = [o["text"] for o in outs]
    summary = summarize(rows, texts)
    summary.update({
        "ended": round(sum(o["ended"] for o in outs) / n, 3),
        "repeated": round(sum(repeated(explanation(t)) for t in texts) / n, 3),
        "format_ok": round(sum(bool(FORMAT.search(t)) for t in texts) / n, 3),
        "priority_ok": round(sum(
            (m := PRIORITY.search(t)) is not None and int(m.group(1)) == int(r["priority"])
            for t, r in zip(texts, rows)
        ) / n, 3),
        "grounding": round(sum(
            grounding(explanation(t), r["model_context"]) for t, r in zip(texts, rows)
        ) / n, 3),
        "tokens_mean": round(sum(o["tokens"] for o in outs) / n, 1),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--models", nargs="*", help="sous-ensemble, ex. sft-550 dpo-100 dpo-210")
    args = parser.parse_args()

    models = candidates()
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    rows = sample(read_jsonl(TEST), args.per_cell, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} cas | modèles : {', '.join(models)}")

    generations = {name: generate(path, rows, args.max_new_tokens) for name, path in models.items()}
    summary = {name: score(rows, outs) for name, outs in generations.items()}

    with (OUT / "generations.jsonl").open("w", encoding="utf-8") as file:
        for i, row in enumerate(rows):
            record = {"id": row["id"], "language": row["language"], "priority": row["priority"],
                      "context": row["model_context"], "sft_target": row["explanation"]}
            record.update({name: generations[name][i]["text"] for name in generations})
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(
        json.dumps({"sample": len(rows), "models": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    keys = ["ended", "repeated", "format_ok", "priority_ok", "grounding", "template_sft", "copy_target",
            "unique_rate", "words_mean", "tokens_mean"]
    print(f"\n{'modèle':10s} " + " ".join(f"{k[:11]:>11s}" for k in keys))
    for name, s in summary.items():
        print(f"{name:10s} " + " ".join(f"{s[k]:>11}" for k in keys))
    print(f"\n✓ {OUT}")


if __name__ == "__main__":
    main()
