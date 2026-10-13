> Version texte du rapport. La version mise en page, avec ses graphiques, est le PDF livré (`livrables/rapport_technique.pdf`).

RAPPORT TECHNIQUE · PREUVE DE CONCEPT

# Agent de triage médical bilingue pour les urgences du CHSA

Qwen3-1.7B-Base spécialisé par SFT et DPO avec QLoRA sur une carte de 6 Go, servi par vLLM derrière une API FastAPI tracée et surveillée : méthode, runs, chiffres, décisions et feuille de route.

**453 / 500** · cas du test gelé bien classés (90,6 %)

**90,62 %** · F1 macro, trois priorités à poids égal

**5,4 %** · sous-triage (7,2 % avec l'instruct)

**93,4 %** · des urgences maximales (P1) reconnues

**84,8 · 96,4 %** · accuracy français · anglais

**6 Go** · de mémoire GPU pour tout l'entraînement

13 octobre 2026 · Centre Hospitalier Saint-Aurélien (CHSA) · données publiques anonymisées, aucun patient réel

## 1. Synthèse

**Un modèle de 1,7 milliard de paramètres, entraîné sur une carte graphique de 6 Go, classe correctement 453 cas sur 500 d'un test gelé bilingue et n'envoie que 5,4 % des patients vers une priorité trop basse. Les objectifs du POC sont atteints ; avant déploiement, je fixe des cibles plus exigeantes et recommande un pilote encadré, en second avis.**

Le CHSA veut aider ses infirmiers d'accueil et d'orientation (IOA) à trier les patients des urgences en trois niveaux, urgence maximale (P1), modérée (P2) ou différée (P3), avec une explication. J'ai construit une preuve de concept complète : de 164 477 lignes de données publiques jusqu'à un service vLLM déployé sur GPU, appelé par une API testée, journalisée et surveillée.

J'ai mené le pipeline deux fois. D'abord sur la version post-entraînée de Qwen3-1.7B (« instruct »), plus simple à démarrer ; puis sur **Qwen3-1.7B-Base**, comme le demande la mission. Le premier essai Base ne terminait aucune réponse : j'ai mesuré pourquoi (section 6), corrigé, réentraîné. À performance globale égale, la Base sous-trie un quart de cas en moins : **c'est elle que je livre.**

| Modèle servi (fusionné en BF16, test gelé 500 cas) | Instruct      | Base (livré)  |
|----------------------------------------------------|---------------|---------------|
| Cas correctement classés                           | 453           | 453           |
| F1 macro                                           | 90,66 %       | 90,62 %       |
| Sous-triage (patient classé trop bas)              | 7,2 %         | 5,4 %         |
| Rappel des P1 · P1 classés P3                      | 92,2 % · 6    | 93,4 % · 5    |
| Accuracy français · anglais                        | 84,8 · 96,4 % | 84,8 · 96,4 % |
| Sur-triage (patient classé trop haut)              | 11            | 20            |

#### Ce que le POC démontre

Un petit modèle apprend le triage FRENCH et sa justification en deux langues, sur du matériel modeste, avec une chaîne reproductible (empreintes SHA-256, manifestes, W&B), testée (CI, couverture ≥ 90 %) et vérifiée sur le modèle réellement servi.

#### Ce qu'il ne démontre pas

Une performance sur de vrais patients, une concordance avec des IOA, ni la qualité clinique des explications. Les priorités viennent d'une application automatisée du protocole, pas de soignants.

#### Recommandation

Engager un pilote en quatre phases : consolider le modèle (filet de sécurité par règles, urgences dépendantes du temps), puis la conformité (AIPD, hébergement HDS, cas réels étiquetés par des IOA), puis un pilote silencieux. Le modèle n'intervient dans la prise en charge qu'en dernière phase, et toujours comme second avis.

## 2. Contexte, objectifs et périmètre

La mission demandait de valider en quatre semaines la faisabilité d'un agent de triage bilingue, sur un modèle compact et du matériel modeste, avec une chaîne industrialisable.

| Objectif de la mission                                    | Ce que j'ai livré                                                                          |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Dataset médical bilingue nettoyé et anonymisé (RGPD)      | 5 000 cas SFT et 2 100 paires DPO, FR/EN à parts égales, Presidio, empreintes SHA-256      |
| Fine-tuning supervisé (SFT) avec LoRA sur Qwen3-1.7B-Base | Six runs exploratoires, deux candidats instruct, puis la Base ; QLoRA 4 bits sous 6 Go     |
| Alignement par préférences (DPO)                          | Sept essais, des paires reconstruites selon le protocole, puis un DPO sur la Base          |
| Endpoint optimisé avec vLLM                               | Modèle fusionné BF16 servi par vLLM 0.31.0, en local et sur GPU RunPod en Europe           |
| CI/CD GitHub Actions                                      | Ruff, 40 tests, couverture ≥ 90 %, image de l'API publiée sur GHCR à chaque commit validé  |
| Tests de latence, pertinence et traçabilité               | Test gelé rejoué sur le modèle servi et sur vLLM, cas inédits, journal d'audit par requête |
| Rapport, grille go / no-go, feuille de route              | Ce document et le site de documentation du projet                                          |

### Contraintes

- **Matériel :** une RTX 4050 Laptop de 6 Go. Un fine-tuning complet est impossible ; tout l'entraînement passe par QLoRA.
- **Données :** aucune donnée patient réelle. Les sources sont des QCM, des questions-réponses et des préférences publiques, pas des dossiers de triage.
- **Langues :** le français et l'anglais sont traités à égalité et mesurés séparément.
- **Sécurité :** l'erreur la plus coûteuse est le sous-triage. C'est mon second critère, juste après le F1 macro.

### Écarts à la mission, et comment je les traite

| Écart                                                       | Décision                                                                                                                                                                                        |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Modèle de départ                                            | Pipeline d'abord mené sur la version instruct, puis repris sur **Qwen3-1.7B-Base** : écart corrigé, la Base est livrée (section 6).                                                             |
| Questionnaire adaptatif non réalisé                         | Le POC classe un cas déjà décrit. Le format d'entrée structuré (âge, symptômes, signes, constantes, antécédents) en est la première brique ; le questionnaire est dans la feuille de route.     |
| Paires DPO de triage plutôt qu'UltraMedical-Preference brut | UltraMedical a été audité (112 362 paires) mais ses réponses ne sont pas des explications de triage ; les paires portent sur les cas du projet, pour aligner ce que l'agent produit réellement. |
| Changement de modèle non automatisé par la CD               | La CD publie l'image de l'API ; la mise en service d'un modèle reste conditionnée au test gelé rejoué sur le modèle fusionné (section 10).                                                      |

## 3. Données

Le volume n'a jamais été le problème : de 164 477 lignes publiques, la difficulté était de trouver des cas réellement triables, puis de leur attribuer une priorité fondée sur un protocole.

**Sources :** MediQAl (FR, CC-BY-4.0, 32 603 lignes, principale source française) · FrenchMedMCQA (FR, Apache-2.0, 3 105) · MedQuAD (EN, CC-BY-4.0, 16 407 fiches maladies projetées en situations patient) · UltraMedical-Preference (EN, MIT, 112 362 paires préférée / rejetée).

### Chaîne de traitement

| Étape                    | Ce que j'ai fait et trouvé                                                                                                                                                         |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Audit                    | Structure, langue, longueur en tokens Qwen3, doublons, contradictions. Environ 61 % des QCM MediQAl n'ont pas de cas clinique, donc pas de patient à trier.                        |
| Doublons, contradictions | 25 771 lignes identiques retirées ; 9 803 paires de quasi-doublons regroupées (MinHash LSH 0,80, RapidFuzz ≥ 95) ; 124 groupes de QCM contradictoires (315 lignes) exclus.         |
| Anonymisation            | Presidio FR et EN (seuil 0,60) et détecteurs maison (NIR, identifiant patient, date de naissance). Le corpus final ne contient aucun e-mail, téléphone, NIR ni URL détectable.     |
| Étiquetage               | Moteur de règles sur la grille FRENCH v1.2 (115 motifs) : FRENCH 1–2 → P1, 3A–3B → P2, 4–5 → P3, avec un niveau de confiance.                                                      |
| Sélection                | 5 000 cas parmi 16 879 : privilégier les cas décrivant un vrai patient donne 0,750 de F1 avec un classifieur rapide, contre 0,572 au hasard.                                       |
| Dataset v2 (9 oct.)      | Explications réécrites pour relier le cas à la priorité, P1 révisés par lots, patients trop courts enrichis, 136 cas remplacés ; aucune priorité validée modifiée silencieusement. |

| Jeu             | Taille | Rôle                    |
|-----------------|--------|-------------------------|
| sft_train       | 4 000  | Apprentissage           |
| sft_validation  | 500    | Choix du checkpoint     |
| sft_test (gelé) | 500    | Mesure finale, une fois |
| dpo_train       | 1 680  | Préférences             |
| dpo_validation  | 420    | Loss DPO                |

50 % FR / 50 % EN et un tiers par priorité dans chaque jeu SFT ; 70 à 280 paires par langue et par priorité côté DPO.

#### Limites des données

- Des étiquettes construites par règles, pas un avis médical ; plus de la moitié du corpus maître a une confiance faible.
- 679 cas « patient fictif » issus des mêmes gabarits : des quasi-doublons entre train et test gonflent le score d'environ 1 point.
- Une détection automatique ne certifie pas l'anonymisation : une revue humaine exhaustive serait nécessaire.

## 4. Méthode d'entraînement et paramètres

La validation choisit, le test gelé mesure. Chaque run a son manifeste (configuration, environnement, empreintes des données, checkpoint retenu), suivi dans Weights & Biases.

### Protocole d'évaluation

- **F1 macro**, critère principal : P1, P2 et P3 pèsent autant. Mon premier run m'a appris qu'une métrique mal choisie trompe : 94,5 % de jetons corrects pour 51,8 % de bonnes priorités.
- **Sous-triage** = (P1 prédits P2 ou P3 + P2 prédits P3) / cas évalués, détaillé par transition ; P1 → P3 est la plus grave.
- **Accuracy par langue**, pour qu'une moyenne ne masque pas une langue moins bien traitée.
- **Scoring conditionnel** : le modèle compare la vraisemblance de « PRIORITY: P1 / P2 / P3 », ce qui mesure la décision sans dépendre d'une génération libre. Les comparaisons entre modèles sont appariées (test de McNemar).

### Paramètres du modèle livré

| Paramètre        | Valeur                                                                                                              | Pourquoi                                                                             |
|------------------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Modèle de départ | `Qwen/Qwen3-1.7B-Base`, 4 bits NF4, calcul BF16                                                                     | Demandé par la mission ; 4 bits pour tenir dans 6 Go                                 |
| LoRA             | r 16 · α 32 · dropout 0,05 · q, k, v, o, gate, up, down                                                             | Apprendre la décision et la forme des explications, ≈ 1 % des paramètres             |
| Jetons spéciaux  | Lignes de `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` entraînées (6 144 paramètres)                               | Jamais entraînées dans la Base ; sans elles, pas de fin de réponse                   |
| Supervision SFT  | Réponse assistant seule ; poids 5 sur la priorité, P1 et les jetons de fin                                          | La priorité ne fait que quelques jetons face à l'explication                         |
| Optimisation SFT | LR 2 × 10⁻⁴ cosine, warmup 5 % · batch 1 × 16 · AdamW 8 bits · clipping 1,0 · 2 048 jetons                          | Valeurs usuelles pour LoRA ; batch effectif 16 dans 6 Go                             |
| Sélection SFT    | Évaluation tous les 50 pas · meilleur F1 de validation · arrêt après 3 évaluations sans progrès                     | Choisir sans regarder le test ; arrêter quand rien ne progresse                      |
| DPO              | β 0,1 · LR 5 × 10⁻⁶ · 1 époque (210 pas) · batch 1 × 8 · référence pré-calculée · poids priorité 0,2, explication 1 | Ajuster l'explication sans déplacer le triage ; une seule copie du modèle en mémoire |
| Service          | Fusion BF16 · vLLM, contexte 2 048 · température 0 · arrêt sur `<|im_end|>` et `<|endoftext|>`                      | Modèle autonome, réponses déterministes                                              |
| Reproductibilité | Seed 42 · Python 3.13.2 · PyTorch 2.11.0 · CUDA 13.0 · Unsloth 2026.9.12 · PEFT 0.19.1                              | Rejouer chaque run à l'identique                                                     |

## 5. Le parcours des runs

Les deux grands gains viennent de la supervision (run 02) et de la reconstruction des données (9 octobre), pas des hyperparamètres. Chaque run devait répondre à une question précise.

| Run               | Ce que je testais                                        | F1      | Sous-tri. | Constat et décision                                                              |
|-------------------|----------------------------------------------------------|---------|-----------|----------------------------------------------------------------------------------|
| 01 · 25 sept.     | Extraction large, priorité seule                         | 51,97 % | 21,2 %    | Bonne précision par jeton, mauvaise décision → juger sur le F1 et le sous-triage |
| 02 · 27 sept.     | Poids 15 / 10 / 10 sur la priorité                       | 78,04 % | 10,8 %    | +26 points mais réponses sans fin → garder une pondération plus équilibrée       |
| 03 · 29 sept.     | Poids 3, jeton de fin, warmup                            | 64,05 % | 17,6 %    | Régression, trois changements à la fois → une variable à la fois                 |
| 04–06 · 1–3 oct.  | Patients projetés, Unsloth, réextraction, plus d'époques | 66–72 % | 17–20 %   | Garder Unsloth ; plus d'époques n'apporte rien → retravailler les données        |
| 8 oct.            | Nouvelle version des données                             | 94,22 % | 3,0 %     | écarté +19 points en un jour : fuite suspectée, contrôles ajoutés                |
| 9 oct. · instruct | Dataset v2, loss pondérée × 5                            | 91,86 % | 5,8 %     | Meilleur instruct (validation 89,51 % contre 86,83 % sans pondération)           |
| 12 oct. · Base 1  | Même pipeline sur la Base                                | 90,47 % | 4,0 %     | rejeté bon triage mais aucune réponse terminée (section 6)                       |
| 12 oct. · Base 2  | Jetons de fin corrigés, lignes spéciales entraînées      | 91,25 % | 4,8 %     | retenu checkpoint-550, 9 / 9 réponses terminées                                  |

#### Ce que je retiens du parcours

Une métrique se choisit avant de lire les résultats ; une seule variable change à la fois ; un score trop beau se vérifie avant d'être célébré. Je signale aussi un biais : lors du choix entre les deux SFT instruct du 9 octobre, je connaissais déjà leurs scores de test. La décision s'appuie sur la validation, mais le test n'était plus tout à fait vierge pour ce choix.

## 6. Instruct puis Base : diagnostic et comparaison

Le premier SFT sur la Base classait bien, mais ne terminait aucune réponse : après une explication correcte, il écrivait un caractère rare puis bouclait jusqu'à 512 jetons. La cause se mesure en deux chiffres.

#### Pourquoi LoRA ne suffisait pas

LoRA modifie les couches d'attention et du MLP, jamais la ligne d'un jeton. Le modèle ne pouvait donc pas apprendre à écrire `<|im_end|>`. En cherchant quand même à l'atteindre, le SFT a écrasé les fins plausibles et rendu la suite presque uniforme. Le DPO n'y pouvait rien : le jeton de fin est le même dans les deux réponses comparées.

#### La correction

La réponse se termine par `<|im_end|>` puis `<|endoftext|>`, un jeton que la Base maîtrise ; les trois lignes des jetons du chat sont entraînées (PEFT `trainable_token_indices`, réactivées après Unsloth qui gèle tout ce qui n'est pas LoRA) ; poids 5 sur la priorité et la fin. Résultat : 9 réponses sur 9 terminées. Les trois changements ont été faits ensemble.

| Même dataset, même test, mêmes réglages      | Instruct               | Base                   |
|----------------------------------------------|------------------------|------------------------|
| SFT · F1 de validation (checkpoint retenu)   | 89,51 % (450)          | 90,93 % (550)          |
| SFT · test : cas justes · F1 · sous-triage   | 459 · 91,86 % · 5,8 %  | 456 · 91,25 % · 4,8 %  |
| SFT · test : P1 → P3 · sur-triage · FR / EN  | 6 · 12 · 87,6 / 96,0 % | 4 · 20 · 85,6 / 96,8 % |
| Durée SFT · DPO (RTX 4050)                   | 2 h 33 · 52 min        | 2 h 35 · 1 h 07        |
| Modèle servi : cas justes · F1 · sous-triage | 453 · 90,66 % · 7,2 %  | 453 · 90,62 % · 5,4 %  |

## 7. Le DPO

Le DPO apprend à préférer une explication à une autre pour un même cas. La priorité étant identique dans les deux réponses, sa contribution s'annule dans la loss : le DPO ne peut pas changer le triage, et il ne l'a jamais changé.

Figure 5 · Loss DPO sur les 420 paires de validation (gauche) et triage suivi sur la validation SFT (droite) ; violet : instruct, vert : Base. La préférence sature en une trentaine de pas, le triage reste plat.

### Ce que j'ai changé au fil des essais

| Quand           | Constat                                                                                     | Décision                                                                         |
|-----------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| 25 sept.–8 oct. | Jusqu'à 11 h par essai, jamais plus d'un point d'écart sur le F1                            | Référence pré-calculée (une seule copie du modèle en mémoire), une époque suffit |
| 10 oct.         | Les priorités des paires ne suivaient FRENCH que dans 42 % des cas vérifiables              | Suivre le triage sur la validation SFT ; reconstruire les paires                 |
| 12 oct.         | Paires reconstruites, explications préférées réécrites (deux phrases, adressées au patient) | DPO réaligné : même triage, formules toutes faites 17,8 → 13,3 % sur 90 cas      |
| 13 oct. · Base  | Loss saturée (0,0038 → 0,0008), triage identique pour tous les checkpoints                  | Choisir le checkpoint sur ce qu'il écrit : 60 cas, 6 modèles comparés            |

| Base · 60 réponses générées    | SFT 550   | DPO 50    | DPO 100   | DPO 150    | DPO 200   | DPO 210   |
|--------------------------------|-----------|-----------|-----------|------------|-----------|-----------|
| Terminées · répétitions        | 100 % · 0 | 100 % · 0 | 100 % · 0 | 100 % · 0  | 100 % · 0 | 100 % · 0 |
| Ancrage dans le contexte       | 0,548     | 0,545     | 0,539     | **0,550**  | 0,520     | 0,539     |
| Formules toutes faites         | 10,0 %    | 11,7 %    | 15,0 %    | **10,0 %** | 11,7 %    | 11,7 %    |
| Copie mot pour mot de la cible | 10,0 %    | 8,3 %     | 6,7 %     | **6,7 %**  | 6,7 %     | 6,7 %     |

Ancrage : part des mots médicaux de l'explication retrouvés dans le contexte du patient, un indicateur d'invention et non une relecture clinique. Un cas vaut 1,7 point : les écarts sont dans le bruit. Je retiens le checkpoint-150, seul meilleur à la fois sur les trois indicateurs.

#### Limite principale

Préférences correctes à 100 % sur la validation (marge 6,1) : les réponses rejetées sont des formules que le modèle n'écrit jamais. Le DPO retouche un mot ou une tournure dans environ la moitié des réponses, sans corriger le gabarit du SFT (25 % des réponses). La suite : prendre comme réponses rejetées les propres sorties du SFT.

## 8. Résultats

Je rapporte les chiffres du modèle réellement servi, fusionné en BF16 : 453 cas justes sur 500, autant que l'instruct, mais avec des erreurs orientées vers le haut plutôt que vers le bas.

Figure 7 · Modèles servis : scores (gauche) et erreurs en nombre de cas (droite).

#### Ce qu'il faut lire derrière ces chiffres

- **Fusion :** passer de l'adapter (base 4 bits) au modèle fusionné BF16 change 8 décisions sur 500 pour les deux modèles (456 → 453 pour la Base) : je rapporte le modèle servi, pas l'adapter.
- **Pas de gain global démontré :** les deux modèles diffèrent sur 26 cas, 12 justes seulement pour la Base et 12 seulement pour l'instruct (McNemar p = 1,0) ; la différence est dans la direction des erreurs.
- **Fuite :** aucun identifiant, contexte ou source partagé entre train et test ; sans les 76 cas trop proches d'un cas du train, le F1 de référence perdait environ 1 point (effet de gabarit).

## 9. Analyse des erreurs

Le modèle a appris l'urgence à partir de mots alarmants, pas à partir du délai de prise en charge. La cause est dans les données d'entraînement, communes aux deux modèles : ni le DPO, ni la fusion, ni le choix instruct / Base ne l'expliquent.

| Cause                                   | Ce que montrent les 4 000 cas du train SFT                                                                                                                                                             | Effet observé                                                              |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Des mots-clés plutôt qu'un raisonnement | 58 % des P1 contiennent un signe de gravité écrit : convulsion (25 %), inconscience (16 %), détresse respiratoire (15 %), hémorragie (9 %). 52 % des P1 sont de courtes fiches « maladie + symptôme ». | Sans mot alarmant, un infarctus ou un AVC tombe en P2.                     |
| Urgences dépendantes du temps rares     | 30 infarctus et 25 AVC sur 4 000 cas (≈ 1,4 %), pourtant tous les AVC étiquetés P1.                                                                                                                    | Un déficit neurologique d'une heure n'est pas appris comme urgence vitale. |
| Étiquettes contradictoires              | Douleur thoracique 48 fois en P1, 13 en P2, 8 en P3 ; une dissection aortique en P2.                                                                                                                   | La douleur thoracique devient ambiguë.                                     |
| Peu de constantes                       | Seuls 23 % des P1, 20 % des P2 et 8 % des P3 en contiennent.                                                                                                                                           | Réaction aux valeurs très anormales, pas aux valeurs modérément anormales. |
| Un format différent                     | Contextes courts (médiane 24 mots), souvent issus de fiches maladies.                                                                                                                                  | Les cas rédigés comme en pratique sortent du connu.                        |

#### L'écart entre langues

84,8 % en français contre 96,4 % en anglais, pour les deux modèles. Les cas anglais sont des situations patient structurées et homogènes ; les cas français, plus proches de vraies notes cliniques, sont plus variés. La Base réduit le sous-triage français (23 cas contre 30) sans changer l'accuracy.

#### L'explication suit la priorité

Elle justifie la décision mais ne la contrôle pas, et peut ajouter des faits absents du cas : des hallucinations qu'aucun contrôle automatique ne détecte encore. Nommer « heart attack » avec une priorité P2 est en revanche une contradiction qu'une règle simple repère.

### Corrections proposées, par ordre de coût

1.  **Filet de sécurité par règles dans l'API :** passer en P1 tout cas avec un drapeau rouge explicite (SpO₂ \< 90 %, systolique \< 90, signes d'AVC, douleur thoracique avec sueurs, purpura fébrile) ou dont l'explication nomme une urgence vitale, en gardant la priorité du modèle pour la traçabilité.
2.  **Données :** quelques centaines de cas d'urgences dépendantes du temps (infarctus, AVC, embolie, dissection) étiquetés P1 avec leurs constantes ; étiquettes contradictoires corrigées ; plus de cas français difficiles.
3.  **Évaluation :** un jeu de test ciblé sur ces urgences pour suivre le rappel P1.

## 10. Architecture et industrialisation

Deux chaînes séparées : l'entraînement sur le poste, le service dans des conteneurs. Un commit sur l'API ne relance jamais un entraînement, et un modèle n'est mis en service qu'après le test gelé rejoué sur sa version fusionnée.

| Brique                 | Ce qui existe et pourquoi                                                                                                                                                                                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| API FastAPI            | `POST /triage` (priorité, explication, recommandation, identifiant, horodatage, version du modèle, latence), `/health`, `/version`, `/metrics`. Validation stricte (langue fr / en, champ inconnu refusé → 422) ; panne du moteur → 503 sans priorité inventée.                                           |
| Trois moteurs          | `mock` pour la CI, `local` (GPU, scoring conditionnel), `vllm` (serveur distant). Prompt et format d'entrée identiques à l'entraînement, vérifiés caractère par caractère.                                                                                                                                |
| Recommandation         | Règle fixe par priorité et par langue, hors du LLM : elle ne peut pas inventer de consigne.                                                                                                                                                                                                               |
| Journal d'audit        | Une ligne JSON par appel, y compris en panne : identifiant, versions de l'API, du modèle et du prompt, langue, priorité, latence, statut. Le texte patient n'est jamais écrit, seulement son empreinte SHA-256 (HMAC à clé secrète en production).                                                        |
| Préparation vLLM       | Adapter et lignes des jetons spéciaux fusionnés en BF16 (3,3 Go), jetons d'arrêt déclarés, contrôle des normes après fusion. Une première fusion sur la mauvaise base était passée de 459 à 417 cas justes sans erreur visible : **le test gelé rejoué sur le modèle servi conditionne toute livraison.** |
| Conteneurs             | Image `python:3.13-slim`, dépendances verrouillées par uv sans PyTorch, utilisateur sans privilèges, ports limités à la machine locale ; Docker Compose à profils (API, monitoring, GPU).                                                                                                                 |
| CI/CD                  | À chaque push : Ruff, 40 tests, couverture ≥ 90 %. Après une CI verte, l'image est publiée sur GHCR avec le SHA du commit. Le changement de modèle reste manuel, après le test gelé : sans ce contrôle, la fusion à 417 / 500 serait partie en production.                                                |
| Déploiement de la démo | Deux pods RunPod en Europe : l'API sur CPU depuis l'image GHCR, vLLM sur GPU (RTX 4000 Ada, 20 Go) qui télécharge les poids d'un dépôt Hugging Face privé. Appels protégés par une clé ; secrets dans RunPod, jamais dans le code ; pods arrêtés hors démonstration.                                      |
| Monitoring             | Prometheus (toutes les 15 s) et un tableau de bord Grafana versionné : volume, débit, latence moyenne et P95, erreurs 5xx, disponibilité. Limite : il suit le service, pas encore le modèle (répartition des priorités, dérive).                                                                          |

### Les décisions d'architecture, et pourquoi

| Décision                                   | Raison                                                                                                   |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Fusionner plutôt que servir base + adapter | Modèle autonome pour vLLM, pas de couche LoRA par requête, lancement plus simple                         |
| vLLM 0.31.0 figé                           | Une mise à jour silencieuse de l'image ne doit pas changer le service                                    |
| Un dossier par DPO                         | Un nouveau DPO sur le même SFT écrasait le précédent : chaque run garde ses checkpoints et ses métriques |
| Dépendances en groupes uv                  | L'image de l'API n'embarque ni PyTorch ni les outils d'entraînement                                      |

## 11. Mesures en conditions réelles : service et cas inédits

Le 12 octobre, les 500 cas du test gelé ont été envoyés au pod vLLM depuis le poste, à travers le proxy RunPod, comme le ferait l'API : la priorité est choisie en générant le texte, comme en production. Je l'ai refaite le 13 octobre avec les poids Base : mêmes décisions que le modèle fusionné, aucune réponse illisible, et le modèle s'arrête bien sur `<|im_end|>`.

| Mesure (vLLM, RunPod, 500 cas)          | Instruct (12 octobre)                                                                                  | Base (livré)                                                                             |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Environnement                           | RTX 4000 Ada (20 Go, 0,28 \$/h), Europe                                                                | **[GPU à préciser]**, Europe                                                       |
| Service                                 | vLLM 0.31.0, BF16, contexte 2 048 jetons, température 0, arrêt sur \<\|im_end\|\> et \<\|endoftext\|\> |                                                                                          |
| Démarrage                               | ≈ 2 min jusqu'au service prêt · moteur initialisé en 40 s · 3,5 Go de mémoire GPU                      | ≈ 1 min 30 s du lancement de vLLM au service prêt, téléchargement des poids Base compris |
| Latence par requête                     | moyenne 792 ms · P50 801 ms · P95 1 028 ms · max 1 733 ms                                              | **moyenne 816 ms · P50 815 ms · P95 1 126 ms · max 1 842 ms**                            |
| Génération                              | 44 jetons par réponse en moyenne, ≈ 56 jetons/s                                                        | 46 jetons par réponse en moyenne, ≈ 56 jetons/s                                          |
| Cas justes · F1 · sous-triage           | 453 / 500 · 90,66 % · 7,2 %                                                                            | **453 / 500 · 90,62 % · 5,4 %**                                                          |
| Décisions identiques au modèle fusionné | 500 / 500                                                                                              | **500 / 500**                                                                            |
| Réponses sans priorité lisible          | 0 / 500                                                                                                | **0 / 500**                                                                              |

### Dix cas inédits, écrits pour la démonstration

J'ai écrit dix cas qui ne viennent d'aucun dataset (5 FR, 5 EN ; 4 P1, 3 P2, 3 P3), chacun soumis sous trois formes : complète (âge, symptômes, signes, constantes, antécédents), minimale (âge et symptômes) et libre (une phrase).

| Forme de l'entrée | Instruct · justes | Instruct · sous-triages | Base · justes | Base · sous-triages |
|-------------------|-------------------|-------------------------|---------------|---------------------|
| Complète          | 7 / 10            | 2                       | 9 / 10        | 0                   |
| Minimale          | 5 / 10            | 4                       | 6 / 10        | 3                   |
| Libre             | 6 / 10            | 4                       | 7 / 10        | 3                   |

Avec la forme complète, la Base reconnaît les 4 urgences maximales, dont l'infarctus et l'AVC que l'instruct classait P2 ; seule l'entorse est sur-triée. L'AVC reste P1 sous les trois formes ; l'infarctus, l'anaphylaxie et le purpura retombent en P2 sans constantes. Les explications restent le point faible : des faits inventés sur 5 réponses sur 30 (« paludisme et tétanos » pour un purpura fébrile, « angiocholite » pour une anaphylaxie, infection pelvienne chez un homme), une boucle de répétition et deux reprises du gabarit du SFT. Ce test fixe le format d'entrée de l'API, la forme complète ; dix cas ne font pas une statistique.

#### Ce que la mesure vLLM vérifie, et ce qu'elle ne vérifie pas

Que le modèle servi en génération prend les mêmes décisions que le modèle évalué par scoring, sans réponse illisible, et que la Base s'arrête bien sur `<|im_end|>`. Les requêtes partent une par une : un test de charge reste à faire.

## 12. Grille go / no-go

**Les objectifs du POC sont atteints. Les cibles de déploiement, plus exigeantes, fixent la marge de progression : tant qu'elles ne sont pas atteintes, le modèle ne traite pas de données réelles.**

| Critère                     | Objectif POC                   | Modèle livré (Base)                      | POC     | Cible de déploiement          |
|-----------------------------|--------------------------------|------------------------------------------|---------|-------------------------------|
| Fidélité du modèle servi    | ≥ 98 % de décisions identiques | 98,4 % après fusion ; 500 / 500 via vLLM | atteint | 100 %                         |
| Performance globale         | F1 macro ≥ 85 %                | 90,62 %                                  | atteint | ≥ 92 %                        |
| Rappel des P1               | ≥ 90 %                         | 93,4 %                                   | atteint | ≥ 95 %                        |
| P1 classés P3               | ≤ 5 % des P1                   | 3,0 % (5 sur 167)                        | atteint | aucun                         |
| Sous-triage                 | ≤ 10 %                         | 5,4 %                                    | atteint | ≤ 5 %                         |
| Bilinguisme                 | ≥ 80 % par langue              | FR 84,8 % · EN 96,4 %                    | atteint | écart ≤ 5 points              |
| Réponses invalides via vLLM | aucune sur 500                 | 0 sur 500                                | atteint | aucune sous charge            |
| Latence                     | P95 ≤ 3 s                      | P95 1,13 s                               | atteint | P95 ≤ 2 s sous charge         |
| Qualité du code             | CI verte, couverture ≥ 90 %    | CI verte, 40 tests                       | atteint | tests de charge en CI         |
| Traçabilité                 | chaque décision journalisée    | journal d'audit                          | atteint | journal centralisé, HMAC      |
| Données et conformité       | données publiques anonymisées  | Presidio FR / EN                         | atteint | AIPD validée, hébergement HDS |

La mission demande de définir ces seuils sans les fixer : ce sont les miens. Les objectifs POC marquent le passage à un modèle exploitable (mes premiers runs étaient à 52 % de F1 et 21 % de sous-triage) ; les cibles de déploiement reprennent pour le sous-triage l'ordre de grandeur visé en traumatologie (5 % au plus) et devront être validées avec les urgentistes du CHSA.

## 13. Risques et cadre réglementaire

Un outil de tri aux urgences relève des usages à haut risque : contrôle humain, traçabilité et conformité des données ne sont pas des options.

#### RGPD et HDS

Données de santé : base légale, minimisation, AIPD et hébergeur certifié HDS. Le POC n'utilise que des données publiques anonymisées ; l'API ne renvoie ni n'expose les données saisies et son journal ne garde qu'une empreinte ; les pods sont en Europe.

#### MDR et AI Act

Une aide à la décision de prise en charge relève probablement du règlement sur les dispositifs médicaux (marquage CE) et, dans l'AI Act, d'un usage à haut risque. Le POC trace données, runs et versions, et laisse la décision à l'IOA.

| Risque                           | Ce qu'on observe déjà                                                                                                                                | Garde-fou proposé                                                                                  |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Sous-triage d'un patient grave   | 5,4 % sur le test, dont 11 P1 ; sur les cas inédits, infarctus, anaphylaxie et purpura classés P2 quand les constantes manquent                      | Décision humaine ; filet de sécurité par règles ; alerte quand le modèle est plus urgent que l'IOA |
| Sur-triage                       | 20 cas classés trop haut avec la Base, contre 11 avec l'instruct                                                                                     | Suivi en pilote silencieux : un sur-triage massif encombrerait les urgences                        |
| Écart entre langues              | 84,8 % en français contre 96,4 % en anglais                                                                                                          | Données françaises réelles en priorité ; métriques toujours suivies par langue                     |
| Explication trompeuse            | Faits parfois inventés ; le DPO modifie peu les explications                                                                                         | Grille de relecture clinique ; l'explication est une aide, jamais une preuve                       |
| Dérive des données               | Appris sur des cas publics, courts et structurés                                                                                                     | Suivi de la répartition des priorités et des cas entrants ; réévaluation périodique                |
| Confiance excessive              | Risque connu des aides à la décision                                                                                                                 | Formation ; pilote silencieux d'abord ; mesure de l'effet du second avis                           |
| Erreur de livraison              | Une fusion sur la mauvaise base est passée de 459 à 417 cas justes sans erreur visible ; la Base sans lignes spéciales ne terminait pas ses réponses | Test gelé et test de génération rejoués sur le modèle servi avant chaque mise en service           |
| Fuite de données dans le service | Aucune donnée patient dans les métriques ni le journal, mais API sans authentification                                                               | Authentification des clients, réseau privé entre API et modèle, empreinte à clé secrète            |

## 14. Recommandations, feuille de route et passage à l'échelle

**Prochaine étape : un pilote encadré.** Le modèle peut devenir un second avis : il lit le cas, propose une priorité justifiée et alerte quand sa priorité est plus urgente que celle de l'IOA, qui garde la décision. Rien ne touche un vrai patient sans hébergement HDS, AIPD validée et évaluation sur des cas réels étiquetés par des IOA.

| Phase                     | Objectif                                                                                                                                                                                                | Critère de sortie                                                                              |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| 0 · Consolider            | Filet de sécurité par règles ; urgences dépendantes du temps ajoutées au SFT ; DPO dont les réponses rejetées sont les sorties du SFT ; relecture des explications ; authentification ; tests de charge | Cibles cliniques de la grille atteintes sur le test gelé ; explications jugées fidèles         |
| 1 · Conformité et données | AIPD et base légale ; hébergement HDS ; cas réels pseudonymisés étiquetés par des IOA selon FRENCH                                                                                                      | AIPD validée ; quelques milliers de cas étiquetés par deux IOA, désaccords arbitrés            |
| 2 · Pilote silencieux     | Le modèle tourne en parallèle des IOA sur les vrais flux, sans que ses réponses soient vues                                                                                                             | Concordance et sous-triage mesurés sur plusieurs semaines, comparés à la variabilité entre IOA |
| 3 · Pilote assisté        | Second avis dans le logiciel des urgences ; questionnaire adaptatif qui demande constantes et antécédents manquants ; alerte de sous-triage                                                             | Pas d'augmentation du sous-triage, adoption par les équipes, statut réglementaire clarifié     |

| Dimension         | Dans le POC                                                             | Pour un pilote au CHSA                                                                                                                                                                                                   |
|-------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Taille du modèle  | Qwen3-1.7B-Base : 4 bits à l'entraînement, BF16 au service              | 8 à 32 milliards de paramètres seulement si le gain est mesuré sur des cas réels ; un 32B occupe ≈ 64 Go en BF16 (un GPU de 80 Go) ou une vingtaine de Go en 4 bits ; la chaîne LoRA, DPO, vLLM, test gelé reste la même |
| Entrée du patient | Cas décrit en texte structuré                                           | Questionnaire adaptatif                                                                                                                                                                                                  |
| Hébergement       | RunPod en Europe, 0,27 à 0,57 \$/h                                      | Hébergeur HDS ou GPU interne, plusieurs réplicas vLLM derrière un répartiteur                                                                                                                                            |
| Sécurité          | Clé entre API et vLLM, journal sans texte patient                       | Authentification, réseau privé, chiffrement, journal centralisé à clé secrète                                                                                                                                            |
| Supervision       | Trafic, latence, erreurs                                                | Répartition des priorités, écart avec les IOA, sous-triage, dérive des cas entrants                                                                                                                                      |
| Cycle de vie      | Manifestes SHA-256, W&B, CD de l'image, modèle mis en service à la main | Registre de modèles ; déploiement automatique seulement si la grille passe sur le modèle servi                                                                                                                           |

## 15. Limites et conclusion

- **Validité clinique :** priorités issues d'une application automatisée de FRENCH, cas publics en partie reformulés ; aucune concordance avec des IOA mesurée.
- **Urgences dépendantes du temps :** sans constantes, l'infarctus repasse en P2 ; c'est la principale marge avant déploiement, avec la fidélité des explications (faits inventés sur 5 réponses sur 30 aux cas inédits).
- **Explications :** mesurées automatiquement, pas relues par des cliniciens ; elles peuvent contenir des faits inventés.
- **DPO :** des paires trop faciles (100 % de préférences correctes), qui ne corrigent pas le gabarit du SFT.
- **Évaluation :** effet de gabarit d'environ 1 point ; écart instruct / Base non significatif (McNemar p = 1,0) ; dix cas inédits ne font pas une statistique ; les premiers runs changeaient plusieurs variables à la fois ; les trois corrections de la Base ont été faites ensemble.

#### Conclusion

Le POC répond à la question de départ : un modèle de 1,7 milliard de paramètres, entraîné sur une carte de 6 Go, apprend à classer des cas cliniques en P1, P2 ou P3 et à justifier sa décision, avec une chaîne complète, testée et tracée jusqu'au service vLLM. Mené sur la Base comme le demande la mission, il atteint le même niveau que la version instruct et sous-trie moins.

Trois leçons structurent la suite. **Les données décident :** les gains nets sont venus de la supervision et de la reconstruction du dataset, et les erreurs restantes s'expliquent par ce que le modèle a vu. **Un chiffre se vérifie sur ce qui est servi :** un score trop beau a été écarté, une fusion défaillante n'a été révélée que par le test gelé, et une Base qui ne s'arrêtait pas n'a été comprise qu'en mesurant ses jetons. **Comprendre sa loss évite des essais inutiles :** ni le DPO ni LoRA ne pouvaient corriger ce que leur loss ou leurs paramètres ne touchaient pas.

La prochaine étape n'est pas technique : c'est l'évaluation sur des cas réels du CHSA, étiquetés par des IOA, dans un cadre conforme.

## A. Annexe : outils et reproductibilité

| Usage                | Outils et versions                                                                                                                                                             |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Environnement        | Python 3.13.2, uv (groupes data, training, dev, ci), Windows 11, RTX 4050 Laptop 6 Go                                                                                          |
| Données              | pandas 3.0.6, datasets 4.3.0, RapidFuzz 3.14.6, langdetect, Presidio 2.2.362, spaCy 3.8.16, grille FRENCH 1.2                                                                  |
| Entraînement         | PyTorch 2.11.0 (CUDA 13.0), Transformers 5.5.0, PEFT 0.19.1, Unsloth 2026.9.12, bitsandbytes 0.49.2, Accelerate 1.15.0, Weights & Biases 0.23.1                                |
| Service              | vLLM 0.31.0, FastAPI 0.142.1, Pydantic 2.13.5, Uvicorn 0.54.0, Docker Compose, Prometheus, Grafana                                                                             |
| Qualité et livraison | GitHub Actions, Ruff 0.17.0, pytest 9.1.1, GHCR, Hugging Face Hub (dépôt privé `SteGONZALEZ/qwen3-triage-chsa`), RunPod                                                        |
| Modèle livré         | SFT run 20261012-162715, checkpoint-550 (adapter 35fe9253…) · DPO `dpo-ckpt550-20261012-195026`, checkpoint-150 (5318247…) · fusion `qwen3-triage-20261012-162715-base-dpo150` |
| Données (SHA-256)    | SFT 0c15e682… · 5c1f6bda… · 01795834… ; paires DPO a3c496d7… · a17b738d… ; protocole FRENCH 11b70614… ; seed 42                                                                |

### Glossaire

| Terme                  | Définition                                                                                                                                       |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| IOA · FRENCH           | Infirmier d'accueil et d'orientation, qui trie les patients ; FRENCH est la grille de triage française (5 niveaux), regroupée ici en P1, P2, P3. |
| SFT                    | Fine-tuning supervisé : le modèle apprend à reproduire des réponses cibles (priorité et explication).                                            |
| DPO                    | Optimisation par préférences : pour un même cas, le modèle apprend à rendre une réponse préférée plus probable qu'une réponse rejetée.           |
| LoRA · QLoRA           | On n'entraîne que de petites matrices ajoutées au modèle (≈ 1 % des paramètres) ; QLoRA garde la base gelée et compressée en 4 bits.             |
| Instruct · Base        | Base : modèle seulement pré-entraîné sur du texte ; instruct : version déjà entraînée par l'éditeur à converser et suivre des consignes.         |
| Jeton · embedding      | Le texte est découpé en jetons ; chaque jeton a une ligne de nombres (embedding) apprise au pré-entraînement.                                    |
| F1 macro · sous-triage | Moyenne des F1 de P1, P2, P3 à poids égal ; part des patients classés à une priorité trop basse.                                                 |
| Scoring conditionnel   | Mesurer la priorité en comparant la probabilité des trois réponses possibles, sans génération libre.                                             |
| Fusion · vLLM          | Intégrer les matrices LoRA dans les poids du modèle ; vLLM est le serveur d'inférence qui sert ce modèle fusionné.                               |
| Test gelé · McNemar    | 500 cas jamais utilisés pour un choix, mesurés une seule fois ; test statistique qui compare deux modèles cas par cas.                           |
| AIPD · HDS             | Analyse d'impact sur la protection des données ; hébergement certifié pour les données de santé.                                                 |

Détail des expériences, des scripts de mesure et du déploiement : site de documentation [gonzalezcourrier34-glitch.github.io/medical_triage_agent](https://gonzalezcourrier34-glitch.github.io/medical_triage_agent/) · code : [github.com/gonzalezcourrier34-glitch/medical_triage_agent](https://github.com/gonzalezcourrier34-glitch/medical_triage_agent).
