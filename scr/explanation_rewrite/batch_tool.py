"""Outils pour la réécriture du champ explanation (dump de lots, QC)."""
import json, sys, random, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/datasets/splits/old/sft_triage_5000_metadata_corrigees.jsonl"
OUT_DIR = ROOT / "data/raw/datasets/splits/old/explanation_rewrite"
RESULTS = OUT_DIR / "results.jsonl"


def load():
    return [json.loads(l) for l in SRC.open(encoding="utf-8")]


def done_ids():
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["id"] for l in RESULTS.open(encoding="utf-8") if l.strip()}


def dump(ids):
    recs = {r["id"]: r for r in load()}
    for i in ids:
        r = recs[i]
        ctx = r["model_context"].replace("Contexte médical:\n", "").strip()
        print(f"### {r['id']} | {r['language']} | P{r['priority']}\n{ctx}\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":  # échantillon stratifié langue x priorité
        rnd = random.Random(14)
        R = [r for r in load() if r["id"] not in done_ids()]
        n = int(sys.argv[2])
        groups = collections.defaultdict(list)
        for r in R:
            groups[(r["language"], r["priority"])].append(r["id"])
        ids = [i for g in sorted(groups) for i in rnd.sample(groups[g], n // len(groups))]
        dump(ids)
    elif cmd == "next":  # prochains lots dans l'ordre du fichier
        n = int(sys.argv[2])
        d = done_ids()
        dump([r["id"] for r in load() if r["id"] not in d][:n])
    elif cmd == "append":  # ajoute un fichier JSONL de résultats après validation
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        valid = {r["id"] for r in load()}
        d = done_ids()
        bad = re.compile(r"\bP[123]\b|priorit|niveau d'urgence|classification|triage|\btri\b|synth|fictif|projet|projected|metadata|métadonn", re.I)
        new = [json.loads(l) for l in Path(sys.argv[2]).open(encoding="utf-8") if l.strip()]
        with RESULTS.open("a", encoding="utf-8") as f:
            for x in new:
                assert x["id"] in valid, x["id"]
                if x["id"] in d:
                    continue
                e = x.get("explanation")
                if e:
                    assert not bad.search(e), (x["id"], e)
                    w = len(e.split())
                    if w > 75:
                        print("TROP LONG", x["id"], w)
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                d.add(x["id"])
        print("total traités:", len(d))
