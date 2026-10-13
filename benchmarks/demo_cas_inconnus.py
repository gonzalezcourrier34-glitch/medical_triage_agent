"""Teste le modèle sur 10 cas patients inédits et cherche le format d'entrée minimal.

Les 10 cas ont été écrits pour la démo : ils ne viennent d'aucun dataset (5 en
français, 5 en anglais ; 4 P1, 3 P2, 3 P3, priorité attendue fixée à la main).
Chaque cas est envoyé sous trois formes, pour voir ce dont le modèle a besoin :

  complet : en-tête + âge + symptômes + signes cliniques + constantes + antécédents
            (le format structuré vu à l'entraînement) ;
  minimal : en-tête + âge + symptômes, sans constantes ni antécédents ;
  libre   : une seule phrase rédigée, sans en-tête ni rubriques.

Le format d'entrée minimal est le plus court qui garde la bonne priorité sur les
10 cas ; le script l'indique à la fin.

Modes :
  local : modèle fusionné chargé sur le GPU (même service que l'API locale)
  vllm  : serveur vLLM (VLLM_BASE_URL, VLLM_API_KEY si protégé)
  api   : API FastAPI déployée (POST <url>/triage), comme la démo

Exemples (depuis la racine du projet) :
  uv run python benchmarks/demo_cas_inconnus.py --mode local
  uv run python benchmarks/demo_cas_inconnus.py --mode api --api-url http://<hote>:8000
Résultats : artifacts/finetuning_sft_dpo/metrics/20261012-162715/demo_cas_inconnus/
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/finetuning_sft_dpo/metrics/20261012-162715/demo_cas_inconnus"
HEADERS = {"fr": "Informations patient:", "en": "Patient information:"}
VARIANTS = ("complet", "minimal", "libre")

# Priorité attendue fixée à la main selon la gravité actuelle et le risque d'aggravation.
CASES = [
    {
        "id": "en_p1_douleur_thoracique",
        "language": "en",
        "expected": "P1",
        "age": 67,
        "symptoms": "crushing chest pain for 40 minutes radiating to the left arm, sweating",
        "signs": "pale, clammy skin",
        "vitals": "heart_rate: 110, systolic_bp: 92, diastolic_bp: 60, spo2: 94",
        "history": "type 2 diabetes, smoker",
        "free": "A 67-year-old diabetic smoker has had crushing chest pain radiating to his "
        "left arm for 40 minutes, with sweating.",
    },
    {
        "id": "fr_p1_anaphylaxie",
        "language": "fr",
        "expected": "P1",
        "age": 31,
        "symptoms": "gonflement des lèvres et de la langue, sifflements respiratoires, "
        "urticaire diffuse 15 minutes après une piqûre de guêpe",
        "signs": "gêne respiratoire, voix rauque",
        "vitals": "heart_rate: 128, systolic_bp: 82, diastolic_bp: 45, spo2: 89",
        "history": "allergie connue aux hyménoptères",
        "free": "Femme de 31 ans piquée par une guêpe il y a 15 minutes : lèvres et langue "
        "gonflées, sifflements, urticaire sur tout le corps.",
    },
    {
        "id": "en_p1_avc",
        "language": "en",
        "expected": "P1",
        "age": 74,
        "symptoms": "sudden weakness of the right arm and slurred speech for one hour",
        "signs": "right facial droop",
        "vitals": "systolic_bp: 178, diastolic_bp: 96, heart_rate: 88",
        "history": "atrial fibrillation, hypertension",
        "free": "A 74-year-old woman suddenly lost strength in her right arm and has had "
        "slurred speech for one hour.",
    },
    {
        "id": "fr_p1_purpura_fievre",
        "language": "fr",
        "expected": "P1",
        "age": 19,
        "symptoms": "fièvre à 39,8 °C, maux de tête, taches violacées qui s'étendent sur "
        "les jambes, somnolence depuis ce matin",
        "signs": "purpura extensif, raideur de nuque",
        "vitals": "temperature: 39.8, heart_rate: 132, systolic_bp: 86, diastolic_bp: 50",
        "history": "aucun",
        "free": "Étudiant de 19 ans, 39,8 °C, maux de tête et taches violacées qui "
        "s'étendent sur les jambes, de plus en plus somnolent.",
    },
    {
        "id": "en_p2_appendicite",
        "language": "en",
        "expected": "P2",
        "age": 42,
        "symptoms": "right lower abdominal pain for 18 hours, nausea, loss of appetite",
        "signs": "tenderness in the right lower quadrant",
        "vitals": "temperature: 38.1, heart_rate: 98, systolic_bp: 128, diastolic_bp: 78",
        "history": "none",
        "free": "A 42-year-old man has had right lower abdominal pain with nausea for "
        "18 hours and a mild fever.",
    },
    {
        "id": "fr_p2_bpco_exacerbation",
        "language": "fr",
        "expected": "P2",
        "age": 58,
        "symptoms": "essoufflement aggravé depuis 3 jours, toux grasse avec crachats verts",
        "signs": "sibilants diffus, parle en phrases complètes",
        "vitals": "spo2: 91, respiratory_rate: 24, temperature: 38.4, heart_rate: 102",
        "history": "BPCO, ancien fumeur",
        "free": "Homme de 58 ans avec une BPCO, plus essoufflé depuis 3 jours, tousse et "
        "crache vert, a un peu de fièvre.",
    },
    {
        "id": "fr_p2_fracture_poignet",
        "language": "fr",
        "expected": "P2",
        "age": 9,
        "symptoms": "chute de trottinette, douleur et déformation du poignet gauche",
        "signs": "pouls radial présent, doigts chauds et mobiles",
        "vitals": "heart_rate: 105",
        "history": "aucun",
        "free": "Enfant de 9 ans tombé de trottinette, poignet gauche douloureux et déformé.",
    },
    {
        "id": "en_p3_mal_de_gorge",
        "language": "en",
        "expected": "P3",
        "age": 29,
        "symptoms": "sore throat for two days, eating and drinking normally",
        "signs": "no difficulty breathing or swallowing",
        "vitals": "temperature: 37.4, heart_rate: 76",
        "history": "none",
        "free": "A 29-year-old has had a sore throat for two days, without fever, and is "
        "eating and drinking normally.",
    },
    {
        "id": "fr_p3_lombalgie",
        "language": "fr",
        "expected": "P3",
        "age": 45,
        "symptoms": "douleur du bas du dos depuis 3 jours après un déménagement, sans "
        "irradiation, sans trouble urinaire",
        "signs": "marche normale, force normale des jambes",
        "vitals": "temperature: 36.8, heart_rate: 72",
        "history": "aucun",
        "free": "Homme de 45 ans, mal au bas du dos depuis un déménagement il y a 3 jours, "
        "marche normalement, pas de problème urinaire.",
    },
    {
        "id": "en_p3_entorse_cheville",
        "language": "en",
        "expected": "P3",
        "age": 35,
        "symptoms": "twisted right ankle yesterday playing football, mild swelling",
        "signs": "able to walk four steps, no bone tenderness",
        "vitals": "heart_rate: 70",
        "history": "none",
        "free": "A 35-year-old twisted her ankle yesterday playing football; it is a little "
        "swollen but she can walk.",
    },
]


def build_case(case: dict, variant: str) -> str:
    """Texte du cas, au format de l'entraînement (rubriques en anglais, comme le SFT)."""
    header = HEADERS[case["language"]]
    if variant == "libre":
        return case["free"]
    lines = [header, f"Age: {case['age']}", "", f"Reported symptoms: {case['symptoms']}"]
    if variant == "complet":
        lines += [
            f"Clinical signs: {case['signs']}",
            f"Vital signs: {case['vitals']}",
            f"Medical history: {case['history']}",
        ]
    return "\n".join(lines)


async def run_local(requests: list) -> list[dict]:
    from src.api.services.inference import InferenceService

    service = InferenceService()
    service.load()
    return [await timed(service.predict(r)) for r in requests]


async def run_vllm(requests: list) -> list[dict]:
    from src.api.services.inference_vllm import VLLMInferenceService

    service = VLLMInferenceService()
    service.load()
    return [await timed(service.predict(r)) for r in requests]


async def timed(coroutine) -> dict:
    start = time.perf_counter()
    try:
        result = await coroutine
    except Exception as error:  # le test continue même si un cas échoue
        result = {"priority": "ERREUR", "explanation": str(error)}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return result


def run_api(payloads: list[dict], url: str) -> list[dict]:
    import httpx

    results = []
    with httpx.Client(timeout=120) as client:
        for payload in payloads:
            start = time.perf_counter()
            try:
                response = client.post(f"{url.rstrip('/')}/triage", json=payload)
                response.raise_for_status()
                data = response.json()
                result = {"priority": data["priority"], "explanation": data["explanation"]}
            except Exception as error:
                result = {"priority": "ERREUR", "explanation": str(error)}
            result["latency_ms"] = round((time.perf_counter() - start) * 1000)
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("local", "vllm", "api"), default="local")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    jobs = [(case, variant) for case in CASES for variant in VARIANTS]
    payloads = [
        {"language": case["language"], "question": build_case(case, variant)}
        for case, variant in jobs
    ]

    if args.mode == "api":
        outputs = run_api(payloads, args.api_url)
    else:
        from src.api.schemas import TriageRequest

        requests = [TriageRequest(**p) for p in payloads]
        runner = run_local if args.mode == "local" else run_vllm
        outputs = asyncio.run(runner(requests))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (case, variant), payload, output in zip(jobs, payloads, outputs):
        rows.append(
            {
                "id": case["id"],
                "variant": variant,
                "language": case["language"],
                "expected": case["expected"],
                "predicted": output["priority"],
                "correct": output["priority"] == case["expected"],
                "latency_ms": output["latency_ms"],
                "input": payload["question"],
                "explanation": output["explanation"],
            }
        )

    summary = {"mode": args.mode, "variants": {}}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        latencies = sorted(r["latency_ms"] for r in subset)
        summary["variants"][variant] = {
            "correct": sum(r["correct"] for r in subset),
            "total": len(subset),
            "undertriage": sum(
                r["predicted"] in ("P1", "P2", "P3") and r["predicted"] > r["expected"]
                for r in subset
            ),
            "latency_ms_median": latencies[len(latencies) // 2],
        }
    sufficient = [v for v in ("libre", "minimal", "complet")
                  if summary["variants"][v]["correct"] == len(CASES)]
    summary["format_minimal_suffisant"] = sufficient[0] if sufficient else None

    with (OUT / f"results_{args.mode}.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / f"summary_{args.mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'cas':28} {'attendu':7} " + " ".join(f"{v:8}" for v in VARIANTS))
    for case in CASES:
        preds = {r["variant"]: r for r in rows if r["id"] == case["id"]}
        cells = " ".join(
            f"{preds[v]['predicted']}{'✓' if preds[v]['correct'] else '✗'}      " for v in VARIANTS
        )
        print(f"{case['id']:28} {case['expected']:7} {cells}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
