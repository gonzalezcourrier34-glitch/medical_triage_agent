# Medical Triage AI | Documentation technique et expérimentale

Site HTML statique qui documente le POC de triage médical bilingue (Qwen3-1.7B-Base, QLoRA, SFT, DPO, API FastAPI).
Ouvrir `index.html` dans un navigateur : aucune dépendance externe, aucun CDN, aucun JavaScript tiers.

## Pages

| Page | Contenu |
|---|---|
| `index.html` | Projet, tâche, contraintes, parcours en cinq étapes, résultats du modèle final, limites |
| `data.html` | Sources, audits, confidentialité (Presidio, RGPD), attribution des priorités FRENCH, sélection, dataset final, empreintes |
| `experiments.html` | Six runs SFT historiques, reconstruction des données, choix du modèle final, test gelé, analyse de fuite, reproductibilité |
| `dpo.html` | Mécanisme du DPO sous 6 Go, paires de préférences, sept essais, saturation des préférences, choix final |
| `base.html` | Comparaison instruct / Base : diagnostic du jeton de fin, meilleurs runs SFT et DPO, modèles servis, cas inédits, choix de la Base |
| `architecture.html` | Chaîne de données, entraînement, API, Docker, CI/CD, monitoring, sécurité, corrections et reste à faire |
| `resultats.html` | Résultats finaux, essai du DPO réaligné, comparaison des explications, test sur dix cas inédits, format d'entrée, limites et conclusion |
| `synthese.html` | Bilan, réponses à la mission, recommandation pour le CHSA, feuille de route, passage à l'échelle, cadre réglementaire, risques |
| `rapport_technique.md` | Version texte du rapport technique (le PDF mis en page, 17 pages est remis à part avec les livrables) : synthèse, données, méthodologie, résultats, mesures cloud, go / no-go, risques, feuille de route |
| `demo.html` | Préparation de la démo : checklist, lancement du service et du monitoring, onglets à ouvrir, déroulé, cas à coller, questions probables, plan de secours |

## Modèle final documenté

- Base : `Qwen/Qwen3-1.7B-Base` (le pipeline a d'abord été mené sur la version instruct, voir `base.html`)
- SFT : run `20261012-162715`, `checkpoint-550` (F1 macro de validation 90,93 %), lignes des jetons spéciaux entraînées
- DPO : `dpo-ckpt550-20261012-195026`, `checkpoint-150`, choisi sur la génération (`benchmarks/compare_checkpoints.py`)
- Modèle fusionné livré, test gelé de 500 cas : 453/500, F1 macro 90,62 %, sous-triage 5,40 %, rappel P1 93,41 % ; 492 décisions identiques à l'adapter, 500/500 via vLLM
- Datasets `sft.jsonl` et `dpo.jsonl` (découpage dans le champ `final_split`) : https://huggingface.co/datasets/SteGONZALEZ/medical_triage_agent
- Poids : `artifacts/deployment/qwen3-triage-20261012-162715-base-dpo150` (BF16, 3,3 Go), produit par `notebooks/07_fusion_lora.ipynb`, publié sur le dépôt Hugging Face privé `SteGONZALEZ/qwen3-triage-chsa` et servi par vLLM. Le dossier `artifacts/` est exclu de Git.
- Version précédente (instruct) : run `20261009-154320`, SFT `checkpoint-450` + DPO réaligné `checkpoint-200`, 453/500, sous-triage 7,20 %.

## Provenance des chiffres

Tous les chiffres viennent des fichiers du projet, pas d'estimations :

- métriques des runs : `artifacts/finetuning_sft_dpo/metrics/<run>/` (métriques, prédictions, manifestes) et `liste run/` pour les six runs historiques ;
- courbes de validation et checkpoints : `trainer_state.json` des checkpoints ;
- audits de données : `data/audits_optimized/` et `audit_manifest.json` ;
- splits et empreintes SHA-256 : `data/raw/datasets/splits/` ;
- récit des runs 01 à 06 : notes de travail de l'auteur.

Les runs historiques ont été évalués sur des versions différentes des données : leur comparaison avec le modèle final est indicative.

## Organisation

- `css/style.css` : style commun
- `js/main.js` : navigation et ancres
- `assets/` : schémas SVG (également intégrés dans les pages) et données structurées

Les schémas sont des SVG statiques accessibles (titre et description pour les lecteurs d'écran), lisibles en thème sombre et sur mobile (défilement horizontal sous 700 px).
