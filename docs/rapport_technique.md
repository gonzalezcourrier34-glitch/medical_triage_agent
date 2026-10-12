# Rapport technique — Agent de triage médical pour le CHSA

12 octobre 2026

Preuve de concept d'un agent de triage bilingue (Qwen3-1.7B, SFT + LoRA + DPO, servi par vLLM) : méthodologie, métriques, analyse et feuille de route.

## 1. Synthèse

**Un modèle de 1,7 milliard de paramètres, entraîné sur une carte graphique de 6 Go, classe correctement 453 cas sur 500 d'un test gelé bilingue (90,66 % de F1 macro). Il sous-trie encore 7,2 % des cas : je recommande un pilote encadré comme second avis, pas un usage autonome.**

Le Centre Hospitalier Saint-Aurélien (CHSA) cherche à aider ses infirmiers d'accueil et d'orientation (IOA) à trier les patients des urgences en trois niveaux : urgence maximale (P1), modérée (P2) ou différée (P3), avec une explication claire. J'ai construit une preuve de concept complète, de la donnée publique jusqu'à un service déployé dans le cloud.

| Indicateur (test gelé, 500 cas, 250 FR / 250 EN) | Modèle livré |
| --- | --- |
| Cas correctement classés | 453 / 500 |
| F1 macro | 90,66 % |
| Sous-triage (patient classé trop bas) | 7,20 % |
| Rappel des P1 | 92,2 % |
| Accuracy français · anglais | 84,8 % · 96,4 % |
| Mémoire GPU d'entraînement | 6 Go (RTX 4050 Laptop) |

**Ce qui est livré :**

- un dataset bilingue de 5 000 cas de triage, nettoyé, dédoublonné et anonymisé, étiqueté selon le protocole FRENCH ;
- un modèle Qwen3-1.7B spécialisé par SFT avec LoRA, puis aligné par DPO, fusionné et vérifié sur le test gelé ;
- une API FastAPI testée, conteneurisée, avec journal d'audit, servie par vLLM ;
- une chaîne CI/CD GitHub Actions, un monitoring Prometheus et Grafana ;
- une grille go / no-go et une feuille de route.

**Ce que le POC ne démontre pas :** une performance sur de vrais patients, une concordance avec des soignants, ni la qualité clinique des explications. Les données sont publiques et les priorités viennent d'une application automatisée du protocole, pas d'IOA.

**Recommandation :** engager un pilote en quatre phases, dont la première consolide le modèle (filet de sécurité par règles, urgences dépendantes du temps) et la deuxième la conformité (AIPD, hébergement HDS, cas réels étiquetés par des IOA). Le modèle n'intervient dans la prise en charge qu'en phase 3, et toujours comme second avis.

## 2. Contexte, objectifs et périmètre

La mission demandait de valider en quatre semaines la faisabilité d'un agent de triage bilingue, sur un modèle compact et du matériel modeste, avec une chaîne industrialisable.

### Objectifs fixés

| Objectif de la mission | Ce que j'ai livré |
| --- | --- |
| Dataset médical bilingue, nettoyé, anonymisé (RGPD) | 5 000 cas SFT et 2 100 paires DPO, FR/EN à parts égales, Presidio, empreintes SHA-256 |
| Fine-tuning supervisé (SFT) avec LoRA | Six runs documentés, puis deux candidats finaux ; QLoRA 4 bits sous 6 Go |
| Alignement par préférences (DPO) | Sept essais, puis un DPO réaligné sur le protocole, livré |
| Endpoint optimisé avec vLLM | Modèle fusionné BF16 servi par vLLM, en local et sur GPU RunPod en Europe |
| CI/CD GitHub Actions | Ruff, 40 tests, couverture ≥ 90 %, image publiée sur GHCR |
| Tests de latence, pertinence et traçabilité | Test gelé rejoué sur le service vLLM (500 décisions identiques, P95 1,03 s), cas inédits, journal d'audit |
| Rapport et feuille de route | Ce document, la grille go / no-go et le site de documentation |

### Contraintes

- **Matériel :** une RTX 4050 Laptop de 6 Go. Un fine-tuning complet est impossible ; tout l'entraînement passe par QLoRA.
- **Données :** aucune donnée patient réelle. Les sources sont des QCM, des questions-réponses et des préférences publiques, pas des dossiers de triage.
- **Langues :** le français et l'anglais doivent être traités à égalité, et mesurés séparément.
- **Sécurité :** l'erreur la plus coûteuse est le sous-triage, un patient grave classé trop bas. C'est mon deuxième critère après le F1 macro.

### Écarts assumés à la mission

| Écart | Décision et justification |
| --- | --- |
| Qwen3-1.7B post-entraîné au lieu de Qwen3-1.7B-Base | Il connaît déjà le format de conversation utilisé par l'API et vLLM, et suit des consignes en français comme en anglais : avec 4 000 exemples, l'entraînement sert au triage plutôt qu'au format. Les deux variantes n'ont pas été comparées à protocole égal ; c'est une expérience à ajouter. Elles ne sont pas interchangeables : fusionner l'adapter dans la Base fait tomber le modèle de 459 à 417 cas justes. |
| Questionnaire adaptatif non réalisé | Le POC classe un cas déjà décrit. Le format d'entrée structuré (âge, symptômes, signes, constantes, antécédents) en est la première brique ; le questionnaire est dans la feuille de route. |
| Paires DPO de triage plutôt qu'UltraMedical-Preference brut | UltraMedical-Preference a été audité (112 362 paires, 11 830 retenues), mais ses réponses ne sont pas des explications de triage. Les paires livrées portent sur des cas de triage du projet, pour aligner ce que l'agent produit réellement. |
| Déploiement automatique du modèle par la CD | La CD publie l'image de l'API ; le changement de modèle reste manuel, conditionné au test gelé rejoué sur le modèle fusionné (section 7). |

## 3. Données

De 164 477 lignes publiques, j'ai tiré 5 000 cas de triage équilibrés par langue et par priorité : la difficulté n'était pas le volume, mais de trouver des cas réellement triables.

### 3.1 Sources

| Source | Langue · licence | Lignes | Usage |
| --- | --- | --- | --- |
| MediQAl | FR · CC-BY-4.0 | 32 603 | QCM et questions ouvertes, parfois avec un cas clinique : principale source française |
| FrenchMedMCQA | FR · Apache-2.0 | 3 105 | QCM de pharmacie, peu de situations patient |
| MedQuAD | EN · CC-BY-4.0 | 16 407 | Fiches maladies, projetées en situations patient |
| UltraMedical-Preference | EN · MIT | 112 362 | Paires préférée / rejetée, base du travail sur les préférences |

### 3.2 Chaîne de traitement

1. **Audit** de chaque source avant transformation : structure, langue, longueur en tokens Qwen3, doublons, contradictions. Environ 61 % des QCM MediQAl n'ont pas de cas clinique, donc pas de patient à trier.
2. **Dédoublonnage** : 25 771 lignes identiques retirées, puis 9 803 paires de quasi-doublons regroupées (MinHash LSH à 0,80, confirmation RapidFuzz ≥ 95).
3. **Contradictions** : 124 groupes de QCM identiques aux réponses divergentes (315 lignes) exclus.
4. **Anonymisation** avec Presidio en français et en anglais (seuil 0,60), complété par des détecteurs maison (NIR, identifiant patient, date de naissance). E-mails, téléphones, URL et identifiants sont masqués automatiquement ; les noms détectés partent en revue, car beaucoup sont des termes médicaux. Le corpus final ne contient aucun e-mail, téléphone, NIR ni URL détectable.
5. **Étiquetage** par un moteur de règles appuyé sur la grille FRENCH v1.2 (115 motifs) : FRENCH 1–2 → P1, 3A–3B → P2, 4–5 → P3, avec un niveau de confiance. Le corpus maître compte 18 523 cas.
6. **Sélection** de 5 000 cas parmi 16 879 candidats. Quatre stratégies ont été comparées avec un classifieur rapide : privilégier les cas décrivant un vrai patient donne 0,750 de F1, contre 0,572 au hasard.
7. **Reconstruction (dataset v2, 9 octobre)** : explications réécrites pour relier le cas à la priorité, P1 révisés par lots, patients trop courts enrichis, 136 cas remplacés. Les priorités validées n'ont pas été modifiées silencieusement.

### 3.3 Jeux finaux

| Jeu | Taille | Composition | Rôle |
| --- | --- | --- | --- |
| sft\_train | 4 000 | 50 % FR / 50 % EN, un tiers par priorité | Apprentissage supervisé |
| sft\_validation | 500 | Idem | Choix du checkpoint |
| sft\_test (gelé) | 500 | 250 FR / 250 EN, environ un tiers par priorité | Mesure finale, utilisée une seule fois |
| dpo\_train | 1 680 | 280 paires par langue et par priorité | Apprentissage des préférences |
| dpo\_validation | 420 | 70 paires par langue et par priorité | Loss DPO et choix du checkpoint |

Chaque exemple SFT contient un contexte patient (23 mots en médiane), une priorité et une explication (22,5 mots). Le découpage respecte des groupes de fuite : aucun contexte, entrée ou source n'est partagé entre train, validation et test. Les paires DPO reprennent volontairement 1 193 cas du train SFT, mais aucune ne recoupe la validation ou le test SFT.

**Limites des données :**

- les priorités sont des étiquettes construites par règles, pas un avis médical ; plus de la moitié du corpus maître a une confiance faible ;
- 679 cas « patient fictif » viennent des mêmes modèles de phrase ; des cas presque identiques se retrouvent entre train et test et gonflent le score d'environ 1 point (section 5) ;
- une détection automatique ne garantit pas une anonymisation complète : une revue humaine exhaustive serait nécessaire pour la certifier.

## 4. Méthodologie d'entraînement

Le modèle livré est le checkpoint-450 du run SFT 20261009-154320, complété par le DPO réaligné (checkpoint-200) : il a été choisi sur la validation, puis mesuré une seule fois sur le test gelé.

### 4.1 Protocole d'évaluation

- **Critère principal : le F1 macro**, qui donne le même poids à P1, P2 et P3. Mon premier run m'a appris qu'une métrique mal choisie trompe : 94,5 % de tokens corrects pour seulement 51,8 % de bonnes priorités.
- **Critère de sécurité : le sous-triage**, (P1 prédits P2 ou P3 + P2 prédits P3) / cas évalués, détaillé par transition (P1→P3 est le plus grave).
- **Accuracy par langue**, pour qu'une moyenne ne masque pas une langue moins bien traitée.
- **Scoring conditionnel** : le modèle compare la vraisemblance des trois réponses « PRIORITY: P1/P2/P3 », ce qui mesure la décision indépendamment d'une génération libre.
- **Règle :** la validation choisit, le test gelé mesure. Les comparaisons entre modèles sont appariées (test de McNemar).

### 4.2 QLoRA sous 6 Go

| Paramètre | Valeur | Raison |
| --- | --- | --- |
| Base | unsloth/Qwen3-1.7B-unsloth-bnb-4bit (Qwen3-1.7B, NF4) | Tenir dans 6 Go ; 1,40 Go alloués après chargement |
| LoRA | r = 16, α = 32, dropout 0,05, 7 projections (attention + MLP) | Apprendre la décision et la forme des explications |
| Optimisation | AdamW 8 bits, LR 2 × 10⁻⁴, cosine, clipping 1,0 | Mémoire réduite, gradients bornés |
| Batch | 1 sur GPU, accumulation 16 | Batch effectif de 16 malgré la mémoire |
| Supervision | Réponse assistant seule, poids 5 sur la priorité, P1 et le token de fin | La priorité ne fait que quelques tokens face à l'explication |
| Longueur, seed | 2 048 tokens, seed 42 | Couvrir les contextes enrichis ; reproductibilité |

### 4.3 Parcours SFT

| Run | Changement principal | F1 macro | Sous-triage | Décision |
| --- | --- | --- | --- | --- |
| 01 (25 sept.) | Extraction large, priorité seule | 51,97 % | 21,23 % | Non retenu |
| 02 (27 sept.) | Poids forts sur la priorité | 78,04 % | 10,80 % | Piste gardée |
| 03 (29 sept.) | Poids réduits, token de fin, warmup | 64,05 % | 17,60 % | Régression |
| 04 (1er oct.) | Patients projetés, Unsloth | 66,27 % | 18,00 % | Non retenu |
| 05 (2 oct.) | Réextraction des champs | 71,60 % | 17,20 % | Référence de l'époque |
| 06 (3 oct.) | Plus d'époques | 66,30 % | 19,60 % | Non retenu |
| 8 oct. | Nouvelle version des données | 94,22 % | 3,00 % | Écarté : fuite suspectée |
| 20261009-154320 | Dataset v2, loss pondérée, 2 époques | 91,86 % | 5,80 % | **Retenu** |
| 20261009-190331 | Dataset v2, loss non pondérée | 88,60 % | 6,40 % | Comparé, non retenu |

Scores sur le test de chaque version des données : seuls les deux runs du 9 octobre sont strictement comparables. Les deux gros gains viennent de la supervision (run 02) et de la reconstruction des données, pas des hyperparamètres. Un run à +19,5 points en une journée a été écarté pour suspicion de fuite plutôt que célébré.

**Choix final.** Le run pondéré est meilleur en validation (F1 89,51 % contre 86,83 %, sous-triage 6,4 % contre 7,2 %). J'avais d'abord préféré le run non pondéré, jugé plus neutre pour le DPO ; l'expérience DPO a montré que cet argument ne tenait pas, et j'ai appliqué la règle : choisir sur la validation. Je connaissais alors déjà les scores de test des deux candidats ; je le signale.

### 4.4 DPO

Le DPO apprend à préférer une explication à une autre pour un même cas. Pour tenir dans 6 Go, les log-probabilités du modèle de référence (le SFT) sont pré-calculées une fois, puis la mémoire est libérée. Réglages : β = 0,1, LR 5 × 10⁻⁶, une époque, batch effectif 8, priorité pondérée 0,2 et explication 1,0.

**La priorité est identique dans la réponse préférée et la réponse rejetée : sa contribution s'annule dans la loss, et le DPO ne peut pas changer le triage.** Les huit essais le confirment : aucun ne déplace le F1 de plus d'un point, et sur le SFT retenu aucune des 500 décisions ne change.

**Réalignement (12 octobre).** Les paires du premier DPO venaient d'une réserve dont les priorités ne suivaient le protocole FRENCH que dans 42 % des cas vérifiables. Je les ai reconstruites en quatre scripts : paires confirmées par la décision FRENCH ou par le SFT, complétion à 350 paires par langue et par priorité, réécriture des 2 100 explications préférées (deux phrases au plus, adressées au patient, sans formule de triage), puis nouveau découpage sans aucun cas vu par le SFT en validation. Le DPO réentraîné avec les mêmes réglages (52 minutes) est celui qui est livré.

## 5. Résultats

Le modèle servi classe 453 cas sur 500 (F1 macro 90,66 %) ; l'adapter évalué pendant l'entraînement en classait 459, et la différence vient de la fusion en BF16.

### 5.1 Test gelé

| Modèle (test gelé, 500 cas) | Cas justes | F1 macro | Sous-triage | FR · EN | Statut |
| --- | --- | --- | --- | --- | --- |
| SFT ckpt-450 + DPO réaligné, adapter sur base 4 bits | 459 | 91,86 % | 5,80 % | 87,6 · 96,0 % | Référence d'évaluation |
| **Modèle livré : adapter fusionné dans Qwen3-1.7B (BF16)** | **453** | **90,66 %** | **7,20 %** | **84,8 · 96,4 %** | **Livré** |
| Première fusion, sur la mauvaise base (Qwen3-1.7B-Base) | 417 | 83,34 % | 14,00 % | 76,8 · 90,0 % | Erreur corrigée |
| Premier DPO (checkpoint-150), fusionné de la même façon | 453 | 90,66 % | 7,20 % | 84,8 · 96,4 % | Remplacé |

**Matrice de confusion du modèle livré**

| Réel ↓ / Prédit → | P1 | P2 | P3 | Rappel |
| --- | --- | --- | --- | --- |
| P1 (167) | 154 | 7 | 6 | 92,2 % |
| P2 (166) | 1 | 142 | 23 | 85,5 % |
| P3 (167) | 0 | 10 | 157 | 94,0 % |

Le modèle servi sous-trie 36 cas, dont 13 P1 ; la frontière P2/P3 reste la principale source d'erreur (23 P2 classés P3). Sur l'adapter de référence, 23 des 29 sous-triages sont français, et les 6 P1 classés P3 le sont tous.

**Fusion.** Passer de l'adapter sur base 4 bits au modèle fusionné en BF16 change 8 décisions sur 500 (McNemar p = 0,070, non significatif). Les 8 vont vers une priorité plus basse et portent surtout sur des cas limites : je rapporte donc les chiffres du modèle servi, pas ceux de l'adapter. Les deux DPO, une fois fusionnés, prennent exactement les mêmes 500 décisions avec des poids différents.

**Fuite.** Aucun identifiant, contexte, entrée ou source n'est partagé entre train et test. En retirant les 76 cas du test trop proches d'un cas du train (similarité de Jaccard ≥ 0,6), le F1 de référence passe de 91,86 % à 90,83 % : l'effet de gabarit vaut environ 1 point.

### 5.2 Explications

Le DPO ne pouvant changer que les explications, je les ai comparées sur 90 cas du test (15 par langue et par priorité), en génération déterministe.

| Indicateur (90 explications) | SFT seul | Premier DPO | DPO réaligné (livré) |
| --- | --- | --- | --- |
| Reprend un gabarit du SFT (« Your reported clinical information includes… ») | 21,1 % | 21,1 % | 21,1 % |
| Contient une formule toute faite de triage | 17,8 % | 15,6 % | 13,3 % |
| Copie mot pour mot la cible du SFT | 14,4 % | 12,2 % | 10,0 % |
| S'adresse au patient | 92 % | 93 % | 92 % |
| Longueur moyenne | 22 mots | 22 mots | 22 mots |

Le DPO va dans le bon sens, mais l'écart ne porte que sur deux à quatre cas sur 90 : je ne le présente pas comme une amélioration démontrée. Sur le gabarit, il ne change parfois qu'un mot (« prompt medical assessment » devient « prompt medical care »). La raison : les réponses rejetées sont des formules que le modèle n'écrit jamais, il les distingue à 100 % dès une trentaine de pas, et son propre gabarit n'est jamais rejeté.

### 5.3 Cas inédits

J'ai écrit dix cas qui ne viennent d'aucun dataset (5 FR, 5 EN ; 4 P1, 3 P2, 3 P3), chacun soumis sous trois formes.

| Forme de l'entrée | Priorités justes | Sous-triages |
| --- | --- | --- |
| Complète : âge, symptômes, signes cliniques, constantes, antécédents | 7 / 10 | 2 |
| Minimale : âge et symptômes | 5 / 10 | 4 |
| Libre : une phrase rédigée | 6 / 10 | 4 |

- **Urgences dépendantes du temps manquées :** l'infarctus et l'AVC sont classés P2 sous toutes les formes, alors que l'explication écrit « suggests a heart attack » ou « suggests a stroke ».
- **Les constantes font la différence :** l'anaphylaxie et le purpura fébrile ne passent en P1 que lorsque la tension basse, la saturation ou la raideur de nuque figurent dans l'entrée.
- **Le bas de l'échelle est fiable :** P2 et P3 sont reconnus dans 16 réponses sur 18 ; les deux erreurs sont des sur-triages.
- **Explications :** toujours dans la langue du cas, en deux phrases au plus, sans redire la priorité. Mais on relève 4 à 5 faits inventés (une chute devient « frappé par la trottinette »), deux reprises du gabarit et cinq fautes, toutes en français.

Ce test fixe le format d'entrée de l'API : la forme complète, avec les constantes. Dix cas ne font pas une statistique, et les priorités attendues sont les miennes : c'est une alerte, pas une mesure.

## 6. Analyse des erreurs

Le modèle a appris l'urgence à partir de mots alarmants, pas à partir du délai de prise en charge : c'est pourquoi il nomme un infarctus ou un AVC mais les classe P2. La cause est dans les données d'entraînement, ni dans le DPO (décisions identiques sans lui), ni dans la fusion (492 décisions sur 500 inchangées).

| Cause | Ce que montrent les 4 000 cas du train SFT | Effet observé |
| --- | --- | --- |
| Des mots-clés plutôt qu'un raisonnement | 58 % des P1 contiennent un signe de gravité écrit en toutes lettres : convulsion (25 %), inconscience (16 %), détresse respiratoire ou cyanose (15 %), hémorragie (9 %). 52 % des P1 sont de courtes fiches « maladie + symptôme ». | Sans mot alarmant, un infarctus ou un AVC tombe en P2. |
| Des urgences dépendantes du temps trop rares | 30 infarctus et 25 AVC sur 4 000 cas, environ 1,4 %. Les 25 AVC sont tous étiquetés P1 : c'est le nombre d'exemples qui manque, pas l'étiquette. | Le modèle n'a pas appris qu'un déficit neurologique d'une heure est une urgence vitale. |
| Des étiquettes contradictoires | La douleur thoracique apparaît 48 fois en P1, 13 fois en P2 et 8 fois en P3 ; une dissection aortique est étiquetée P2. La tournure « X suggests Y. You need a prompt assessment » est fréquente en P2. | La douleur thoracique devient ambiguë ; l'explication de l'infarctus reprend ce gabarit de P2. |
| Peu de constantes | Seuls 23 % des P1, 20 % des P2 et 8 % des P3 contiennent des constantes. | Le modèle réagit aux valeurs très anormales (SpO2 89 %, tension 82/45), pas aux valeurs modérément anormales (92/60, pouls 110). |
| Un format différent | Contextes courts (médiane 24 mots), souvent issus de fiches maladies. | Les cas rédigés comme en pratique sortent de ce que le modèle connaît le mieux. |

Comptages par recherche de mots-clés en français et en anglais : des ordres de grandeur, pas une annotation clinique.

**Deux autres faiblesses :**

- **L'écart entre langues.** Le français reste en retrait (84,8 % contre 96,4 %). Après la reconstruction, les cas anglais sont des situations patient structurées et homogènes ; les cas français, plus proches de vraies notes cliniques, sont plus variés et plus difficiles.
- **L'explication suit la priorité, elle ne la contrôle pas.** Pour l'infarctus classé P2, elle écrit « prompt » et non « immediate ». Elle peut aussi ajouter des faits absents du cas : ce sont des hallucinations, qu'aucun contrôle automatique ne détecte encore. En revanche, nommer « heart attack » ou « stroke » avec une priorité P2 est une contradiction qu'une règle simple peut repérer.

**Corrections proposées, par ordre de coût :**

1. Un filet de sécurité par règles dans l'API : passer en P1 tout cas qui présente un drapeau rouge explicite (saturation sous 90 %, tension systolique sous 90, signes d'AVC, douleur thoracique avec sueurs, purpura fébrile), ou dont l'explication nomme une urgence vitale, en conservant la priorité du modèle pour la traçabilité.
2. Quelques centaines de cas d'urgences dépendantes du temps (infarctus, AVC, embolie pulmonaire, dissection), étiquetés P1 selon FRENCH avec leurs constantes, et les étiquettes contradictoires corrigées.
3. Un jeu de test ciblé sur ces urgences pour suivre le rappel P1, et des cas français difficiles en priorité.

## 7. Architecture et industrialisation

L'entraînement et le service sont deux chaînes séparées : un commit sur l'API ne relance jamais un entraînement, et le seul lien entre les deux est le modèle fusionné, vérifié avant chaque livraison.

### 7.1 Vue d'ensemble

| Brique | Technologie | Rôle | État |
| --- | --- | --- | --- |
| Chaîne de données | Python, Presidio, MinHash, RapidFuzz, moteur de règles FRENCH | Sources publiques → cas de triage tracés | Opérationnel |
| Entraînement | Unsloth, PEFT, Transformers, W&B, notebooks 05 à 07 | SFT, DPO, fusion ; un manifeste par run | Opérationnel |
| API | FastAPI, Pydantic | Validation, inférence, recommandation, journal d'audit | Opérationnel |
| Service du modèle | vLLM (API compatible OpenAI) | Modèle fusionné BF16, contexte 2 048 tokens | Opérationnel en local, déployé sur RunPod pour la démo |
| Conteneurs | Docker, Compose à profils | API seule, monitoring, GPU à la demande | Opérationnel |
| CI/CD | GitHub Actions, GHCR | Lint, tests, couverture, image versionnée | Opérationnel |
| Monitoring | Prometheus, Grafana | Trafic, latence, erreurs, disponibilité | Opérationnel |

### 7.2 API

| Route | Rôle |
| --- | --- |
| `POST /triage` | Renvoie la priorité, l'explication, une recommandation, l'identifiant de requête, l'horodatage, la version du modèle et la latence |
| `GET /health` | Indique si le moteur d'inférence est prêt (`ok` ou `degraded`) |
| `GET /version` | Versions de l'API, du modèle et du prompt |
| `GET /metrics` | Métriques HTTP pour Prometheus, sans aucune donnée patient |

- **Trois moteurs**, choisis par une variable d'environnement : `mock` (sans GPU, pour la CI et les tests), `local` (modèle chargé sur GPU, scoring conditionnel) et `vllm` (serveur distant). Le prompt et le format d'entrée sont centralisés et identiques à l'entraînement, vérifié caractère par caractère.
- **Recommandation déterministe** : une règle fixe par priorité et par langue, hors du LLM, qui ne peut pas inventer de consigne.
- **Validation stricte** : langue `fr` ou `en`, question non vide, champ inconnu refusé (422).
- **Panne du moteur** : réponse 503 sans détail technique ni priorité inventée.
- **Journal d'audit** : chaque appel ajoute une ligne JSON (identifiant, horodatage, versions de l'API, du modèle et du prompt, moteur, langue, priorité, latence, statut), y compris en cas de panne. Le texte du patient n'est jamais écrit, seulement son empreinte SHA-256 : on peut vérifier qu'une entrée donnée a produit une décision donnée. Une empreinte simple reste une donnée pseudonymisée ; en production, il faudrait une empreinte à clé secrète (HMAC).
- **Tests** : 40 fonctions de test (routes, validation, recommandations, journal d'audit, non-exposition des données, prompt, échanges vLLM simulés, panne), couverture minimale de 90 % imposée en CI.

### 7.3 Préparer le modèle pour vLLM

L'adapter LoRA est fusionné dans Qwen/Qwen3-1.7B en BF16 (3,3 Go, 1,72 milliard de paramètres), pour que vLLM serve un modèle autonome. Une première fusion, faite par erreur dans la variante Base, est passée de 459 à 417 cas justes sans aucune erreur visible : architecture et tokenizer identiques. Seul un test sur les prédictions l'a révélée. **Le test gelé rejoué sur le modèle exact qui sera servi conditionne désormais toute livraison** (`benchmarks/eval_test_gele.py`).

### 7.4 Conteneurs, CI/CD et déploiement

- **Image de l'API** : `python:3.13-slim`, dépendances verrouillées par uv, sans PyTorch ni outils d'entraînement, utilisateur sans privilèges, ports limités à la machine locale.
- **CI** à chaque push et pull request : Ruff, puis les tests avec couverture ≥ 90 %. **CD** : après une CI réussie, l'image est construite et publiée sur GHCR avec l'étiquette du commit, ce qui relie le service à un code précis.
- **Ce que la CD n'automatise pas** : la mise en service d'un nouveau modèle reste manuelle (poids publiés sur Hugging Face, version déclarée dans `MODEL_VERSION`), après le test gelé. C'est volontaire : sans contrôle automatique, la fusion à 417/500 serait passée en production. L'étape suivante est un workflow qui lance le test gelé sur GPU et ne déploie que si la grille go / no-go passe.
- **Déploiement de la démo** : deux pods RunPod en Europe. L'API tourne sur CPU à partir de l'image publiée par la CD ; vLLM tourne sur GPU (RTX A5000 ou RTX PRO 4000, 24 Go, 0,27 à 0,57 $/h) et télécharge les poids depuis un dépôt Hugging Face privé. L'API n'appelle vLLM qu'avec une clé ; les secrets vivent dans RunPod, jamais dans le code. Les pods sont arrêtés en dehors des démonstrations.

### 7.5 Monitoring

Prometheus collecte toutes les 15 secondes les métriques de l'API et de vLLM. Un tableau de bord Grafana versionné suit le volume, le débit, la latence moyenne et P95, la part d'erreurs 5xx et la disponibilité de l'API et de vLLM. Limite : il suit le service, pas le modèle ; la répartition des priorités prédites et la dérive des cas entrants ne sont pas encore mesurées.

## 8. Mesures en conditions réelles

**Le service vLLM déployé sur GPU prend exactement les mêmes 500 décisions que le modèle fusionné évalué en local, sans aucune réponse invalide, avec une latence P95 d'environ 1 seconde.** Le 12 octobre, les 500 cas du test gelé ont été envoyés au pod vLLM depuis le poste, à travers le proxy RunPod, comme le ferait l'API ; la priorité est choisie en générant le texte, comme en production.

| Mesure | Résultat |
| --- | --- |
| Environnement | RTX 4000 Ada (20 Go, 0,28 $/h), datacenter européen, vLLM 0.31.0, BF16, contexte 2 048 tokens |
| Démarrage | Environ 2 minutes du lancement du conteneur au service prêt (image en cache) ; moteur initialisé en 40 s ; 3,5 Go de mémoire GPU utilisés |
| Latence par requête | Moyenne 792 ms · P50 801 ms · P95 1 028 ms · maximum 1 733 ms |
| Génération | 44 tokens en moyenne par réponse, environ 56 tokens/s |
| Cas justes, F1 macro, sous-triage | 453 / 500 · 90,66 % · 7,20 % ; FR 84,8 %, EN 96,4 % |
| Décisions identiques au modèle évalué par scoring | 500 / 500 |
| Réponses sans priorité lisible | 0 / 500 |
| Capacité théorique | Cache de 148 000 tokens, soit environ 72 requêtes simultanées à 2 048 tokens |

Les requêtes imposent `temperature 0` : la génération est déterministe, malgré les réglages d'échantillonnage par défaut du modèle signalés par vLLM au démarrage. Le cache de préfixe réutilise le prompt système commun à 85-90 %, ce qui réduit la latence. Limite : les requêtes ont été envoyées une par une ; un test de charge avec des requêtes simultanées reste à faire. Une séance de mesure coûte moins d'un dollar.

## 9. Grille go / no-go

**Verdict : go pour la démonstration sur des cas fictifs, no-go pour tout usage sur des données réelles.** Les critères techniques sont atteints ; les critères cliniques et réglementaires ne le sont pas.

Cette grille se rejoue avant chaque mise en service, sur le modèle exact qui sera servi. Les seuils sont les miens et donnent un point de départ à valider avec les urgentistes du CHSA ; pour le sous-triage, je reprends l'ordre de grandeur visé en traumatologie (5 % au plus).

| Critère | Seuil | Modèle livré | Verdict |
| --- | --- | --- | --- |
| Fidélité du modèle servi | ≥ 98 % de décisions identiques à l'adapter évalué | 492 / 500 (98,4 %) | Atteint |
| Performance globale | F1 macro ≥ 85 % | 90,66 % | Atteint |
| Rappel des P1 | ≥ 95 % | 92,2 % | Non atteint |
| P1 classés P3 | Aucun | 6 sur 167 | Non atteint |
| Sous-triage | ≤ 5 % | 7,2 % | Non atteint |
| Écart entre langues | ≤ 5 points d'accuracy | 11,6 points | Non atteint |
| Réponses invalides via vLLM | Aucune sur 500 | 0 sur 500 | Atteint |
| Latence | P95 ≤ 3 s par requête à l'API | P95 1,03 s, moyenne 0,79 s (vLLM, réseau compris) | Atteint |
| Qualité du code | CI verte, couverture ≥ 90 % | CI verte | Atteint |
| Traçabilité | Chaque décision journalisée avec les versions du modèle et du prompt | Journal d'audit de l'API | Atteint |
| Conformité | AIPD validée, hébergement HDS | Non engagées | Non atteint |

Les critères non atteints correspondent exactement aux phases 0 et 1 de la feuille de route (section 11) : sous-triage et écart entre langues d'abord, conformité ensuite.

## 10. Risques et cadre réglementaire

Un outil de tri aux urgences relève des usages à haut risque : le contrôle humain, la traçabilité et la conformité des données ne sont pas des options.

### 10.1 Cadre

| Texte | Ce qu'il impose | Ce que le POC prépare |
| --- | --- | --- |
| RGPD et hébergement HDS | Données de santé sensibles : base légale, minimisation, analyse d'impact (AIPD), hébergeur certifié HDS dès qu'un prestataire héberge ces données | Données publiques anonymisées ; API qui ne renvoie ni n'expose les données saisies ; journal d'audit sans texte patient ; pods en Europe |
| Règlement sur les dispositifs médicaux (MDR) | Un logiciel qui aide à décider d'une prise en charge relève probablement du MDR : marquage CE avant tout usage clinique hors recherche | Positionnement en second avis, documentation des performances et des limites |
| AI Act | Le tri des patients aux urgences figure parmi les usages à haut risque : gestion des risques, qualité des données, traçabilité, contrôle humain | Empreintes des données, manifestes de run, versions tracées dans chaque réponse et dans le journal, décision laissée à l'IOA |

### 10.2 Risques et garde-fous

| Risque | Ce qu'on observe déjà | Garde-fou proposé |
| --- | --- | --- |
| Sous-triage d'un patient grave | 7,2 % sur le test, dont 13 P1 ; infarctus et AVC classés P2 sur les cas inédits | Décision humaine ; filet de sécurité par règles ; alerte quand le modèle est plus urgent que l'IOA |
| Écart entre langues | 84,8 % d'accuracy en français contre 96,4 % en anglais | Données françaises réelles en priorité ; métriques toujours suivies par langue |
| Explication trompeuse | Faits parfois inventés ; le DPO modifie peu les explications | Grille de relecture clinique ; l'explication est une aide, jamais une preuve |
| Dérive des données | Le modèle a appris sur des cas publics, courts et structurés | Suivi de la répartition des priorités et des cas entrants ; réévaluation périodique |
| Confiance excessive des soignants | Risque connu des aides à la décision | Formation ; pilote silencieux d'abord ; mesure de l'effet du second avis |
| Erreur de livraison du modèle | Une fusion sur la mauvaise base est passée de 459 à 417 cas justes sans erreur visible | Test gelé rejoué sur le modèle servi avant chaque mise en service |
| Fuite de données dans le service | Aucune donnée patient dans les métriques ni le journal, mais API sans authentification | Authentification des clients, réseau privé entre API et modèle, empreinte à clé secrète |

## 11. Recommandations, feuille de route et passage à l'échelle

**Oui à un pilote encadré, non à un usage autonome.** Le modèle peut devenir un second avis : il lit le cas, propose une priorité justifiée et alerte quand sa priorité est plus urgente que celle de l'IOA, qui garde la décision.

Rien ne doit toucher un vrai patient avant trois conditions : un hébergement certifié HDS, une AIPD validée avec le délégué à la protection des données, et une évaluation sur des cas réels du CHSA étiquetés par des IOA.

### 11.1 Feuille de route

Quatre phases sur environ un an pour une petite équipe ; on ne passe à la suivante que si les critères de sortie sont atteints.

| Phase | Objectif | Critère de sortie |
| --- | --- | --- |
| 0 · Consolider le POC | Filet de sécurité par règles ; cas d'infarctus, d'AVC et d'autres urgences dépendantes du temps ajoutés au SFT ; DPO dont les réponses rejetées sont les propres sorties du SFT ; grille de relecture des explications ; authentification de l'API ; versions d'images figées ; tests de charge | Critères cliniques de la grille go / no-go atteints sur le test gelé ; explications relues et jugées fidèles au cas |
| 1 · Conformité et données | AIPD et base légale ; hébergement HDS ; extraction de cas réels pseudonymisés ; étiquetage par des IOA selon FRENCH | AIPD validée ; quelques milliers de cas étiquetés par au moins deux IOA, désaccords arbitrés |
| 2 · Pilote silencieux | Le modèle tourne en parallèle des IOA sur les vrais flux, sans que personne ne voie ses réponses | Concordance et sous-triage mesurés sur plusieurs semaines, comparés à la variabilité entre IOA |
| 3 · Pilote assisté | Intégration au logiciel des urgences comme second avis ; questionnaire adaptatif qui demande les constantes et antécédents manquants ; alerte de sous-triage ; ré-entraînement encadré | Pas d'augmentation du sous-triage, adoption par les équipes, statut réglementaire clarifié |

En continu : l'IOA ou le médecin garde la décision, le sous-triage est suivi à chaque version, et le test gelé est rejoué avant toute mise en service.

### 11.2 Passage à l'échelle

| Dimension | Dans le POC | Pour un pilote au CHSA |
| --- | --- | --- |
| Taille du modèle | Qwen3-1.7B : 4 bits à l'entraînement, BF16 au service | Modèles de 8 à 32 milliards de paramètres et plus, seulement si le gain est mesuré sur des cas réels. Un 32B occupe environ 64 Go en BF16 (un GPU de 80 Go) ou une vingtaine de Go en 4 bits ; la chaîne LoRA, DPO, vLLM et test gelé reste la même |
| Entrée du patient | Cas décrit en texte structuré | Questionnaire adaptatif : sur les cas inédits, un cas incomplet passe de 7 à 5 bonnes réponses sur 10 |
| Hébergement | RunPod en Europe, 0,27 à 0,57 $/h | Hébergeur certifié HDS ou serveur GPU interne |
| Inférence | Un serveur vLLM, un GPU | Plusieurs réplicas vLLM derrière un répartiteur ; un modèle de cette taille tient sur des GPU d'entrée de gamme |
| Fidélité du modèle servi | Adapter fusionné en BF16 (453 / 500) | Servir la base 4 bits avec l'adapter non fusionné pour retrouver les décisions évaluées (459 / 500), si la latence le permet |
| Sécurité | Clé entre l'API et vLLM, journal d'audit sans texte patient, API sans authentification | Authentification des clients, réseau privé, chiffrement, journal centralisé avec empreinte à clé secrète et durée de conservation fixée avec le DPD |
| Supervision | Trafic, latence, erreurs | Indicateurs métier : répartition des priorités, écart avec les IOA, sous-triage, dérive des cas entrants |
| Cycle de vie | Manifestes SHA-256, W&B ; CD de l'image de l'API, changement de modèle manuel | Registre de modèles ; la CD déploie un nouveau modèle seulement si la grille go / no-go passe sur le modèle servi |

## 12. Limites et conclusion

### 12.1 Limites

- **Validité clinique :** les priorités viennent d'une application automatisée du protocole FRENCH, et les cas de données publiques en partie reformulées. Aucune concordance avec des IOA n'est mesurée.
- **Urgences dépendantes du temps :** infarctus et AVC sans constantes alarmantes passent en P2. C'est la limite principale.
- **Explications :** mesurées automatiquement sur 90 cas, pas relues par des cliniciens ; elles peuvent contenir des faits inventés.
- **DPO :** des paires trop faciles (100 % de préférences correctes) qui ne corrigent pas le gabarit du SFT.
- **Évaluation :** l'effet de gabarit gonfle le score d'environ 1 point ; le test gelé n'était plus tout à fait vierge au moment du choix entre les deux SFT finaux ; dix cas inédits ne font pas une statistique.
- **Historique :** les premiers runs changeaient plusieurs variables à la fois ; ce n'est pas une ablation.
- **Variante du modèle :** la version post-entraînée et la variante Base n'ont pas été comparées à protocole égal.

### 12.2 Conclusion

Le POC répond à la question de départ : un petit modèle, entraîné sur une carte de 6 Go, apprend à classer des cas cliniques en P1, P2 ou P3 et à justifier sa décision, avec une chaîne complète, testée et tracée jusqu'au service vLLM. Il ne répond pas encore à la question qui compte pour un hôpital : aide-t-il vraiment une IOA sans lui faire prendre de risque ?

Trois leçons structurent la suite :

1. **Les données décident.** Les gains nets sont venus de la supervision et de la reconstruction du dataset, pas des hyperparamètres ; les erreurs restantes s'expliquent par ce que le modèle a vu.
2. **Un chiffre se vérifie sur ce qui est servi.** Un score trop beau a été écarté, et seul le test gelé rejoué sur le modèle fusionné a révélé une erreur de livraison.
3. **Comprendre sa loss évite des essais inutiles.** Le DPO ne pouvait pas changer le triage, parce que la priorité s'annulait entre les deux réponses.

La prochaine étape n'est pas technique : c'est l'évaluation sur des cas réels du CHSA, étiquetés par des IOA, dans un cadre conforme.

## Annexe — Reproductibilité

Chaque résultat de ce rapport est rattaché à des fichiers identifiés par leur empreinte. Le détail de chaque expérience est sur le [site de documentation](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/) et le code dans le [dépôt GitHub](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent).

| Élément | Valeur |
| --- | --- |
| sft\_train · sft\_validation · sft\_test | 4 000 · 500 · 500 cas ; SHA-256 0c15e682…b61ea0 · 5c1f6bda…286e65 · 01795834…b2abe7 |
| dpo\_train · dpo\_validation (réalignés) | 1 680 · 420 paires ; SHA-256 a3c496d7…04a1fc · a17b738d…2a6b3f |
| Protocole FRENCH v1.2 | 115 motifs ; SHA-256 11b70614…185e44d |
| Checkpoint SFT | Run 20261009-154320, checkpoint-450 |
| Adapter DPO livré | dpo\_aligned\_adapter = checkpoint-200 ; SHA-256 fed4587b…5ff477 |
| Modèle fusionné | qwen3-triage-20261009-154320-aligned, Qwen/Qwen3-1.7B en BF16, 3,3 Go ; dépôt Hugging Face privé |
| Version exposée par l'API | qwen3-1.7b-sft154320-ckpt450-dpoaligned-ckpt200 · prompt triage-classification-v2 |
| Environnement d'entraînement | Python 3.13.2, PyTorch 2.11.0, CUDA 13.0, Windows 11, RTX 4050 Laptop 6 Go, seed 42 |
| Suivi | W&B, projet medical-triage-qwen3 ; un manifeste JSON par run |

**Rejouer les mesures :**

```bash
uv run python benchmarks/eval_test_gele.py --mode hf
uv run python benchmarks/eval_test_gele.py --mode vllm --vllm-url <url du serveur>/v1
uv run python benchmarks/compare_explanations.py
uv run python benchmarks/demo_cas_inconnus.py --mode local
```

La chaîne complète va des notebooks 01 à 07 (audit, nettoyage, découpage, SFT, DPO, fusion), complétés par les scripts de réalignement des paires DPO (`scr/data/dpo_alignment.py`, `dpo_balance.py`, `dpo_chosen_rewrite.py`, `dpo_resplit.py`). La procédure de déploiement cloud est décrite pas à pas dans `deploy/runpod.md`.
