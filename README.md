<div align="center">

# Medical Triage Agent

**Un petit modèle de langage spécialisé pour classer un patient en P1, P2 ou P3 et justifier sa décision, en français et en anglais.**

Qwen3-1.7B · QLoRA · SFT + DPO · vLLM · FastAPI · Docker · GitHub Actions

[![CI](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent/actions/workflows/ci.yml)
[![CD](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent/actions/workflows/cd.yml/badge.svg)](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/badge/documentation-GitHub%20Pages-2f9e8a)](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/)
![Python](https://img.shields.io/badge/python-3.13-3776ab)
![uv](https://img.shields.io/badge/deps-uv-de5fe9)
![Ruff](https://img.shields.io/badge/lint-ruff-d7ff64)

[Documentation technique](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/) ·
[Expériences](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/experiments.html) ·
[Architecture](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/architecture.html) ·
[Synthèse et feuille de route](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/synthese.html)

</div>

> [!WARNING]
> **Preuve de concept académique, pas un dispositif médical.** Le modèle n'a fait l'objet d'aucune validation clinique.
> Il ne doit jamais servir à décider de la prise en charge d'un patient réel. Les exemples de ce dépôt sont fictifs.

---

## Sommaire

- [Le projet](#le-projet)
- [Résultats](#résultats)
- [Architecture](#architecture)
- [Démarrage rapide](#démarrage-rapide)
- [Utiliser l'API](#utiliser-lapi)
- [Configuration](#configuration)
- [Reproduire le modèle](#reproduire-le-modèle)
- [Structure du dépôt](#structure-du-dépôt)
- [Qualité, CI/CD et déploiement](#qualité-cicd-et-déploiement)
- [Données et confidentialité](#données-et-confidentialité)
- [Limites](#limites)

## Le projet

À partir d'une description clinique libre, le modèle renvoie :

- une **priorité** : `P1` (urgence maximale), `P2` (urgence modérée) ou `P3` (urgence différée) ;
- une **explication** courte, fondée uniquement sur les éléments du cas ;
- une **recommandation** standardisée, ajoutée de façon déterministe par l'API.

Contraintes du projet : un modèle imposé (Qwen3-1.7B), un entraînement sur une seule RTX 4050 Laptop (6 Go de VRAM),
des données publiques anonymisées et un protocole d'évaluation qui sépare strictement sélection et mesure.

| Étape | Choix |
|---|---|
| Données | 4 sources publiques FR/EN, audit, détection de données personnelles (Presidio), déduplication, étiquetage selon le protocole **FRENCH v1.2** |
| Dataset | 5 000 cas équilibrés (50 % FR / 50 % EN, un tiers par priorité) : 4 000 train · 500 validation · 500 test gelé |
| SFT | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`, LoRA r=16 / α=32 sur 7 projections, loss pondérée sur la priorité |
| DPO | 2 100 paires centrées sur l'explication (la priorité est identique dans les deux réponses), β = 0,1 |
| Service | LoRA fusionné dans Qwen3-1.7B (BF16), servi par vLLM derrière une API FastAPI |

## Résultats

Test gelé de **500 cas** (250 FR, 250 EN), jamais utilisé pour choisir un modèle :

| Modèle | Cas justes | F1 macro | Sous-triage | Accuracy FR · EN |
|---|---:|---:|---:|---:|
| Adapter SFT + DPO (évaluation de référence, base 4 bits) | 459 / 500 | 91,86 % | 5,80 % | 87,6 % · 96,0 % |
| **Modèle fusionné livré** (BF16, servi par vLLM) | **453 / 500** | **90,66 %** | **7,20 %** | 84,8 % · 96,4 % |

- La fusion garde **492 décisions sur 500** identiques à la référence (McNemar exact, p = 0,070) ; les écarts concernent des cas limites.
- Sans les cas du test trop proches d'un cas d'entraînement, le F1 macro de référence est de 90,83 % : pas de fuite massive, un effet de gabarit d'environ 1 point.
- Le DPO ne change aucune décision de triage : il agit sur la formulation des explications.

Le détail des runs, des erreurs et des décisions est dans la [page Expériences](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/experiments.html).

## Architecture

```mermaid
flowchart LR
    C[Client] -->|POST /triage| A[API FastAPI<br/>validation · recommandation · métriques]
    A -->|INFERENCE_BACKEND| B{Moteur}
    B -->|mock| M[Réponse simulée<br/>tests et CI]
    B -->|local| L[Transformers + GPU]
    B -->|vllm| V[Serveur vLLM<br/>modèle fusionné]
    A -->|/metrics| P[Prometheus] --> G[Grafana]
```

L'API ne dépend pas du modèle : le même code sert la CI (moteur simulé), un poste avec GPU (moteur local)
et le déploiement (vLLM, éventuellement distant et protégé par une clé).

## Démarrage rapide

Prérequis : [uv](https://docs.astral.sh/uv/) et Python 3.13 (installé par uv si besoin). Docker pour la partie conteneurs.

```bash
git clone https://github.com/gonzalezcourrier34-glitch/medical_triage_agent.git
cd medical_triage_agent
cp .env.example .env            # puis compléter les valeurs

uv sync                         # dépendances de l'API
uv run uvicorn main:app --reload
```

L'API démarre avec le moteur simulé : la documentation interactive est sur <http://127.0.0.1:8000/docs>.

**Avec Docker Compose** (profils activés à la demande) :

```bash
docker compose up -d                          # API seule
docker compose --profile monitoring up -d     # + Prometheus (9090) et Grafana (3000)
docker compose --profile gpu up -d            # + vLLM (8001), GPU NVIDIA requis
```

Le profil `gpu` attend le modèle fusionné dans `artifacts/deployment/qwen3-triage-20261009-154320/`
(voir [Reproduire le modèle](#reproduire-le-modèle)). Tous les ports sont liés à `127.0.0.1`.

## Utiliser l'API

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/triage` | Priorité, explication et recommandation pour un cas |
| `GET` | `/health` | `ok` si le moteur est prêt, `degraded` sinon |
| `GET` | `/version` | Versions de l'API, du modèle et du prompt |
| `GET` | `/metrics` | Métriques Prometheus (aucune donnée médicale) |
| `GET` | `/docs` | Documentation OpenAPI interactive |

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"language": "fr", "question": "Homme de 68 ans, douleur thoracique oppressive depuis 40 minutes, sueurs."}'
```

Requête : `language` (`fr` ou `en`) et `question` sont obligatoires ; `context` et `answer` sont facultatifs.
Tout autre champ est refusé (`422`), pour éviter de recevoir des données non prévues.

Réponse (forme) :

```json
{
  "request_id": "uuid",
  "timestamp": "2026-10-11T15:00:00Z",
  "model_version": "qwen3-1.7b-sft154320-ckpt450-dpo-ckpt150",
  "priority": "P1",
  "explanation": "…",
  "recommendation": "Prise en charge immédiate par une équipe médicale.",
  "latency_ms": 0.0,
  "status": "success"
}
```

Si le moteur d'inférence est injoignable, `/triage` renvoie `503` sans détail technique.

## Configuration

Variables lues par l'API et Docker Compose (modèle : [`.env.example`](.env.example)) :

| Variable | Défaut | Rôle |
|---|---|---|
| `INFERENCE_BACKEND` | `mock` | Moteur : `mock`, `local` ou `vllm` |
| `VLLM_BASE_URL` | `http://vllm:8000/v1` | URL du serveur vLLM (compatible OpenAI) |
| `VLLM_API_KEY` | vide | Clé du serveur vLLM, obligatoire dès qu'il est exposé |
| `VLLM_MODEL` | `triage-qwen3` | Nom du modèle servi par vLLM |
| `MODEL_PATH` | modèle fusionné | Moteur `local` : modèle fusionné ou adapter LoRA |
| `MODEL_VERSION` | `qwen3-1.7b-sft154320-ckpt450-dpo-ckpt150` | Version renvoyée par l'API |
| `GRAFANA_ADMIN_PASSWORD` | aucun | Obligatoire pour le profil `monitoring` |

## Reproduire le modèle

L'entraînement demande un GPU NVIDIA (testé sur 6 Go) et les dépendances des groupes `data` et `training` :

```bash
uv sync --group data --group training
```

| Étape | Notebook | Sortie |
|---|---|---|
| Audit des sources | `01_audit_raw_data.ipynb` | rapports d'audit |
| Nettoyage, anonymisation | `02_cleaning_pipeline.ipynb` | corpus nettoyé |
| Découpage SFT | `03_decoupage_datasets_sft.ipynb` | `sft_train` / `sft_validation` / `sft_test` |
| Découpage DPO | `04_decoupage_datasets_dpo.ipynb` | `dpo_train` / `dpo_validation` |
| SFT | `05_finetuning_sft.ipynb` | adapter SFT, métriques, manifeste |
| DPO | `06_finetuning_dpo.ipynb` | adapter DPO, métriques, manifeste |
| Fusion pour vLLM | `07_fusion_lora.ipynb` | modèle fusionné BF16 |

Les splits sont identifiés par leur empreinte SHA-256 et chaque run écrit un manifeste
(configuration, environnement, données) ; le suivi d'expériences utilise Weights & Biases.

Vérifier le modèle fusionné sur le test gelé, puis mesurer la latence via vLLM :

```bash
uv run python benchmarks/eval_test_gele.py --mode hf
uv run python benchmarks/eval_test_gele.py --mode vllm --vllm-url http://127.0.0.1:8001/v1
```

Les données, adapters et modèles (`data/`, `artifacts/`) ne sont pas versionnés.

## Structure du dépôt

```text
.
├── main.py                  # application FastAPI (routes, métriques Prometheus)
├── src/api/                 # code de l'API, seul code embarqué dans l'image Docker
│   ├── routes/              #   /triage, /health, /version
│   ├── schemas.py           #   contrats Pydantic (champs inconnus refusés)
│   └── services/            #   prompt, moteurs mock / local / vLLM, recommandations
├── scr/                     # chaîne de données : nettoyage, extraction, triage FRENCH
├── notebooks/               # 01 → 07 : audit, données, SFT, DPO, fusion
├── tests/                   # tests de l'API (pytest)
├── benchmarks/              # test gelé et latence du modèle servi
├── docker/Dockerfile.api    # image de l'API (Python 3.13, utilisateur non root)
├── docker-compose.yml       # API, vLLM, Prometheus, Grafana (profils)
├── monitoring/              # configuration Prometheus et dashboard Grafana
├── deploy/runpod.md         # procédure de déploiement cloud de la démo
├── docs/                    # documentation technique (GitHub Pages)
└── .github/workflows/       # CI (qualité, tests) et CD (image GHCR)
```

## Qualité, CI/CD et déploiement

```bash
uv run ruff check main.py src tests
uv run ruff format --check main.py src tests
uv run pytest tests -q --cov=src --cov=main
```

- **Tests** : 34 fonctions de test, dont la validation des entrées, la non-exposition des données saisies,
  le prompt identique à l'entraînement, le choix du moteur et les échanges avec vLLM simulés.
  La CI exige au moins 90 % de couverture ; le moteur GPU local, exclu de ce calcul, est validé par le test gelé.
- **CI** (`ci.yml`) : Ruff puis pytest à chaque push et pull request sur `master`.
- **CD** (`cd.yml`) : après une CI réussie, construction de l'image de l'API et publication sur GitHub Container Registry
  (`ghcr.io/gonzalezcourrier34-glitch/medical_triage_agent/api:sha-<commit>`).
  Un déploiement SSH vers un serveur pilote est prévu et s'active avec la variable de dépôt `DEPLOY_ENABLED=true`.
- **Démo cloud** : vLLM sur un GPU RunPod en Europe et l'API dans un second pod, voir [`deploy/runpod.md`](deploy/runpod.md).

## Données et confidentialité

- Sources publiques uniquement ; détection des données personnelles avec Presidio (FR/EN), puis revue des cas signalés.
  Le corpus final ne contient ni e-mail, ni téléphone, ni numéro de sécurité sociale, ni URL.
- L'API ne journalise pas le contenu des requêtes, ne le renvoie pas et ne l'expose pas dans les métriques.
- Un usage réel exigerait un hébergement certifié HDS, une analyse d'impact (AIPD) et une authentification des clients :
  la démo RunPod n'est pas conçue pour des données de santé réelles.

Détails : [page Données](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/data.html).

## Limites

- Pas de validation clinique ; les étiquettes viennent d'une application automatisée du protocole FRENCH.
- Le français reste moins bien classé que l'anglais (84,8 % contre 96,4 % d'accuracy sur le modèle livré).
- La qualité des explications n'est pas encore mesurée par une grille d'évaluation.
- L'API de démonstration n'a pas d'authentification.

Recommandation, feuille de route vers un pilote au CHSA et analyse des risques : [page Synthèse](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/synthese.html).

---

<div align="center">
<sub>Projet réalisé dans le cadre d'une formation OpenClassrooms · documentation complète sur
<a href="https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/">GitHub Pages</a></sub>
</div>
