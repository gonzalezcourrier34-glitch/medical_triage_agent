# Medical Triage AI | Documentation technique et expérimentale

Site HTML statique qui documente le POC de triage médical bilingue (Qwen3-1.7B, QLoRA, SFT, DPO, API FastAPI).
Ouvrir `index.html` dans un navigateur : aucune dépendance externe, aucun CDN, aucun JavaScript tiers.

## Pages

| Page | Contenu |
|---|---|
| `index.html` | Projet, tâche, contraintes, parcours en cinq étapes, résultats du modèle final, limites |
| `data.html` | Sources, audits, confidentialité (Presidio, RGPD), attribution des priorités FRENCH, sélection, dataset final, empreintes |
| `experiments.html` | Six runs SFT historiques, reconstruction des données, choix du modèle final, test gelé, analyse de fuite, reproductibilité |
| `dpo.html` | Mécanisme du DPO sous 6 Go, paires de préférences, sept essais, saturation des préférences, choix final |
| `architecture.html` | Chaîne de données, entraînement, API, Docker, CI/CD, monitoring, sécurité, corrections et reste à faire |
| `resultats.html` | Résultats finaux, essai du DPO réaligné, comparaison des explications, test sur dix cas inédits, format d'entrée, limites et conclusion |
| `synthese.html` | Bilan, réponses à la mission, recommandation pour le CHSA, feuille de route, passage à l'échelle, cadre réglementaire, risques |
| `demo.html` | Préparation de la démo : checklist, lancement du service et du monitoring, onglets à ouvrir, déroulé, cas à coller, questions probables, plan de secours |

## Modèle final documenté

- SFT : run `20261009-154320`, `checkpoint-450` (sélection sur le F1 macro de validation, 89,51 %)
- DPO : DPO réaligné, adapter `dpo_aligned_adapter`, identique à `dpo_aligned_checkpoints/checkpoint-200` (plus basse loss de validation DPO) ; le premier DPO (`dpo_adapter_livre`, checkpoint-150) prend les mêmes décisions et sert de retour arrière
- Test gelé (500 cas) : F1 macro 91,86 %, accuracy 91,80 %, sous-triage 5,80 % ; 90,83 % de F1 sans les cas trop proches du train
- Modèle fusionné livré, même test : F1 macro 90,66 %, accuracy 90,60 %, sous-triage 7,20 % ; 492 décisions sur 500 identiques à l'adapter (McNemar p = 0,070), contrôle par `benchmarks/eval_test_gele.py`
- Poids finaux livrés : modèle fusionné `artifacts/deployment/qwen3-triage-20261009-154320-aligned` (Qwen3-1.7B + LoRA SFT/DPO, BF16, 3,3 Go), produit par `notebooks/07_fusion_lora.ipynb` et servi par vLLM. Ce dossier est exclu de Git (`artifacts/`) : il est transmis séparément.

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
