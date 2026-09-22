# Documentation technique — Audit des datasets médicaux SFT et DPO

## Préparation des données pour le fine-tuning d’un modèle de triage médical

**Projet :** POC de fine-tuning d’un modèle de langage médical  
**Modèle cible :** Qwen3-1.7B-Base  
**Approche :** SFT puis DPO  
**Objectif final :** produire un modèle capable d’estimer un niveau d’urgence clinique et de générer des instructions adaptées à partir d’un questionnaire patient.

---

## 🧩 Sommaire

1. Objectif et rôle de l’audit
2. Cadre, documentation et configuration des corpus
3. Chargement et table canonique d’audit
4. Audit structurel
5. Audit des QCM
6. Audit textuel
7. Répétitions, quasi-doublons et contradictions
8. Audit de diversité
9. Audit médical, sécurité clinique et triage
10. Audit RGPD et confidentialité
11. Audit spécifique DPO
12. Audit des fuites entre splits
13. Traçabilité, intégrité et export des rapports
14. Décisions issues de l’audit
15. Projection vers le nettoyage et la construction du SFT/DPO
16. Limites de l’audit
17. Conclusion

---

# 1. Objectif et rôle de l’audit

## Point

Les données destinées au fine-tuning proviennent de plusieurs corpus médicaux hétérogènes. Elles ne peuvent pas être utilisées directement pour l’entraînement sans vérifier leur structure, leur qualité, leur cohérence, leur confidentialité et les risques de fuite entre jeux de données.

L’audit constitue donc une étape distincte du nettoyage. Il **observe, mesure et signale**, mais ne modifie pas les données sources.

## Décision

Mettre en place un notebook d’audit unique avant toute sélection ou transformation afin de disposer d’un état initial reproductible des données.

L’audit doit permettre de répondre à quatre questions :

- les fichiers sont-ils techniquement exploitables ?
- leur contenu est-il suffisamment cohérent et pertinent pour le projet ?
- quels exemples nécessitent une exclusion, une correction ou une revue ?
- quelles règles devront être appliquées dans le notebook de nettoyage ?

## Mise en place

Le notebook :

- charge les fichiers bruts ;
- contrôle leur schéma ;
- construit une représentation canonique commune ;
- exécute des audits spécialisés ;
- conserve les résultats dans des rapports séparés ;
- calcule des empreintes SHA-256 ;
- produit un manifeste final de traçabilité.

## Résultats

L’audit permet de conserver les corpus dans un environnement d’analyse commun sans perdre leurs particularités. Les anomalies identifiées deviennent des **décisions à prendre**, et non des suppressions automatiques.

## Projection pour le nettoyage

Le notebook de nettoyage utilisera les rapports produits par l’audit comme règles d’entrée. Il appliquera ensuite les décisions de déduplication, confidentialité, sélection, équilibrage et construction des datasets finaux.

---

# 2. Cadre, documentation et configuration des corpus

## Point

Quatre familles de données sont étudiées :

| Corpus | Langue principale | Structure | Usage envisagé |
|---|---|---|---|
| FrenchMedMCQA | Français | QCM | SFT |
| MediQAl | Français | MCQU, MCQM, questions ouvertes | SFT / évaluation |
| MedQuad-KV | Anglais | Questions-réponses ouvertes | SFT sous réserve |
| UltraMedical-Preference | Anglais | Paires `chosen` / `rejected` | DPO |

Les sources n’ont ni le même format, ni le même volume, ni nécessairement le même rôle dans le projet.

## Décision

Documenter chaque corpus avant son exploitation : provenance, licence, langue, format, splits disponibles, usage envisagé et restrictions.

La provenance et la licence de la variante locale **MedQuad-KV** doivent notamment être confirmées avant son utilisation définitive pour l’entraînement.

## Mise en place

Une configuration centralisée décrit les datasets attendus et leurs caractéristiques. Elle permet d’éviter de disperser les règles de chargement et de validation dans le notebook.

Les rôles SFT, DPO et évaluation sont également distingués afin de ne pas mélanger des données ayant des objectifs d’apprentissage différents.

## Résultats

Les corpus sont complémentaires :

- FrenchMedMCQA apporte des QCM médicaux francophones ;
- MediQAl apporte plusieurs formats de raisonnement et des cas contextualisés ;
- MedQuad-KV apporte des questions-réponses ouvertes anglophones ;
- UltraMedical-Preference fournit un grand volume de préférences pour le DPO.

Le volume brut n’est pas considéré comme un critère suffisant de qualité ou de représentativité.

## Projection pour le nettoyage

La sélection finale ne reproduira pas mécaniquement les proportions des corpus bruts. Les sources seront pondérées selon leur utilité pour la tâche cible, leur qualité et leur statut documentaire.

---

# 3. Chargement et table canonique d’audit

## Point

Les datasets utilisent des structures incompatibles entre elles : QCM, réponses ouvertes, contexte clinique ou préférences DPO.

Auditer directement chaque schéma séparément rendrait les contrôles difficiles à comparer et multiplierait le code spécifique.

## Décision

Construire une **table canonique d’audit** sans modifier les fichiers sources.

Cette table ne remplace pas les données originales. Elle fournit uniquement une représentation commune pour les analyses.

## Mise en place

Le chargement prend en charge les formats utilisés par les corpus, puis vérifie la présence des colonnes obligatoires.

Chaque observation est ensuite convertie vers des champs communs tels que :

- identifiant ;
- dataset et source ;
- famille de tâche ;
- langue attendue ;
- texte d’entrée ;
- contexte ;
- cible ;
- réponses `chosen` et `rejected` pour le DPO ;
- métadonnées utiles ;
- texte complet destiné aux audits textuels.

Des fonctions dédiées normalisent également les choix QCM et les positions de réponses.

## Résultats

La table canonique permet d’exécuter les mêmes contrôles sur des corpus différents tout en conservant la source et la nature de chaque observation.

Les volumes chargés sont conservés afin de pouvoir vérifier ultérieurement qu’aucune ligne n’a été perdue pendant l’audit.

## Projection pour le nettoyage

La table canonique sert de couche d’observation. Le nettoyage repartira des données sources et des décisions d’audit afin de produire les formats SFT et DPO réellement destinés au modèle.

---

# 4. Audit structurel

## Point

Avant d’évaluer le contenu, il faut vérifier que les fichiers respectent leur structure attendue et identifier les valeurs manquantes ou incohérentes.

Une valeur manquante n’est pas automatiquement une erreur. Certains champs peuvent être optionnels selon la source ou le type de tâche.

## Décision

Contrôler séparément :

- les colonnes obligatoires ;
- les valeurs manquantes ;
- les identifiants ;
- les types internes ;
- les différences de schéma entre splits ;
- les champs structurellement optionnels.

## Mise en place

Des contrôles automatiques comparent les colonnes présentes aux spécifications déclarées pour chaque famille de dataset.

Les valeurs manquantes sont comptabilisées par champ et par dataset. Les différences entre splits sont également analysées.

## Résultats

Les corpus sont globalement exploitables. Les absences observées dans certains champs MediQAl, notamment le contexte clinique, peuvent être structurelles et ne justifient pas une suppression automatique.

Les différences entre questions ouvertes et QCM correspondent à des différences de tâche et ne sont pas traitées comme des anomalies.

## Projection pour le nettoyage

Le nettoyage ne supprimera une observation pour valeur manquante que si le champ est réellement indispensable à la tâche cible. Aucun contexte absent ne sera inventé ou imputé artificiellement.

---

# 5. Audit des QCM

## Point

Les corpus QCM utilisent des réponses uniques ou multiples. Une mauvaise interprétation des positions correctes produirait directement des cibles d’entraînement erronées.

## Décision

Vérifier la structure des choix avant toute conversion en données SFT.

Les contrôles portent sur :

- le nombre de propositions ;
- les positions A à E ;
- le nombre de réponses correctes ;
- la cohérence entre QCM unique et multiple ;
- les positions invalides ;
- la distribution des réponses.

## Mise en place

Les choix sont convertis vers une représentation commune A-E.

L’audit distingue :

- **MCQU** : une seule réponse correcte ;
- **MCQM** : plusieurs réponses correctes possibles ;
- **FrenchMedMCQA** : une ou plusieurs positions correctes selon le format source.

La distribution des positions est également calculée pour repérer un éventuel biais structurel.

## Résultats

Les contrôles permettent de confirmer que les réponses peuvent être interprétées de manière homogène et de repérer les rares particularités de format avant transformation.

Le nombre de choix n’est pas supposé identique pour tous les corpus sans vérification.

## Projection pour le nettoyage

Les réponses seront normalisées en positions A-E tout en préservant la nature unique ou multiple de la question. Une observation dont la réponse ne peut pas être interprétée de façon fiable devra être exclue ou revue.

---

# 6. Audit textuel

## Point

Le fine-tuning dépend directement de la qualité des textes. Il faut donc contrôler la langue, la longueur, la tokenisation, les anomalies de forme et la dépendance éventuelle à un contexte externe.

## Décision

Séparer plusieurs dimensions de qualité textuelle plutôt que d’utiliser une règle unique de longueur.

Une observation longue, courte ou atypique est **signalée**, mais n’est pas automatiquement supprimée.

## Mise en place

### Langue

Un échantillon de chaque source est analysé avec un détecteur de langue. La langue détectée est comparée à la langue attendue.

### Longueur

Les longueurs en caractères et en mots sont calculées pour les champs principaux.

Des statistiques médianes, percentiles et maxima sont produites.

### Tokens

Le tokenizer du modèle cible est utilisé pour mesurer la longueur réelle des séquences.

Les seuils 512, 1024 et 2048 tokens permettent d’anticiper les besoins de troncature et de choisir la longueur maximale d’entraînement.

### Anomalies textuelles

Plusieurs signaux sont recherchés :

- texte vide ;
- texte anormalement court ;
- espaces excessifs ;
- répétitions de caractères ;
- répétitions de mots ;
- proportion anormalement élevée de symboles.

### Autonomie

Pour MediQAl, l’audit recherche les formulations faisant référence à un contexte externe. Une question peut ainsi être classée comme dépendante avec contexte, dépendante sans contexte ou autonome/non détectée.

## Résultats

Les langues attendues sont très majoritairement respectées.

Les distributions de tokens montrent que la majorité des observations est compatible avec des longueurs raisonnables, mais quelques valeurs extrêmes existent et doivent être prises en compte lors de la préparation finale.

Les anomalies textuelles constituent principalement des candidats à la revue plutôt que des motifs automatiques d’exclusion.

## Projection pour le nettoyage

Le nettoyage :

- conservera la langue comme métadonnée ;
- exclura ou corrigera les textes réellement inutilisables ;
- préservera le contexte lorsqu’il est nécessaire ;
- évitera les troncatures silencieuses ;
- sélectionnera les exemples compatibles avec la longueur maximale retenue pour le SFT/DPO.

---

# 7. Répétitions, quasi-doublons et contradictions

## Point

Deux exemples proches ne sont pas nécessairement des doublons. Une même question peut avoir plusieurs formulations ou plusieurs réponses légitimes.

À l’inverse, conserver des copies presque identiques peut surpondérer certains contenus et provoquer des fuites entre splits.

## Décision

Distinguer quatre phénomènes :

1. répétition d’entrée ;
2. répétition complète entrée + cible ;
3. quasi-doublon ;
4. contradiction potentielle.

## Mise en place

Les textes sont normalisés puis plusieurs empreintes sont calculées :

- `input_hash` pour l’entrée ;
- `target_hash` pour la cible ;
- `example_hash` pour l’ensemble entrée + cible ;
- `group_hash` pour les groupes devant rester indivisibles.

Les quasi-doublons sont recherchés par similarité textuelle avec RapidFuzz.

Pour les QCM, une même question normalisée associée à plusieurs cibles est signalée comme contradiction potentielle.

Pour MedQuad-KV, plusieurs réponses à une même question sont signalées séparément, car plusieurs réponses ouvertes peuvent être légitimes.

## Résultats

Les corpus présentent des niveaux de répétition différents.

MedQuad-KV contient notamment davantage de questions répétées après normalisation, tandis qu’UltraMedical-Preference contient naturellement de nombreux prompts réutilisés dans différentes comparaisons.

L’audit évite volontairement de considérer « même question » comme synonyme de « doublon à supprimer ».

## Projection pour le nettoyage

Les triplets ou exemples réellement identiques seront dédupliqués avant la création des splits.

Les quasi-doublons et contradictions seront examinés selon leur nature. Les variantes apportant une information clinique ou pédagogique différente pourront être conservées.

---

# 8. Audit de diversité

## Point

Les volumes bruts sont fortement déséquilibrés. Une sélection proportionnelle au volume ferait dominer les plus grands corpus sans garantir une meilleure couverture de la tâche.

## Décision

Mesurer la diversité avant la sélection afin de construire volontairement le dataset final.

Les dimensions étudiées comprennent :

- source ;
- type de tâche ;
- présence de contexte ;
- spécialité médicale lorsqu’elle est disponible ;
- langue.

## Mise en place

Les répartitions sont calculées à partir de la table canonique.

Les métadonnées de spécialité sont utilisées comme indicateurs lorsqu’elles existent, mais elles ne sont pas interprétées comme une description exhaustive du contenu médical.

## Résultats

UltraMedical-Preference domine le volume brut, ce qui est cohérent avec son rôle DPO mais ne doit pas influencer artificiellement la composition du SFT.

Les sources SFT apportent des formats complémentaires : QCM uniques, QCM multiples, réponses ouvertes et cas contextualisés.

## Projection pour le nettoyage

La constitution des 5 000 exemples SFT sera pilotée par des objectifs explicites :

- équilibre FR/EN ;
- diversité des sources ;
- diversité des formats ;
- couverture médicale ;
- représentation suffisante des priorités P1/P2/P3 ;
- limitation des doublons et des groupes surreprésentés.

---

# 9. Audit médical, sécurité clinique et triage

## Point

Un dataset peut être médical sans être pertinent pour le triage.

Il faut donc distinguer la présence de vocabulaire médical, les situations potentiellement aiguës et les contenus à fort enjeu clinique.

Ces signaux ne constituent **pas des labels de priorité clinique**.

## Décision

Utiliser les marqueurs médicaux et de triage uniquement comme outils de caractérisation et de sélection.

La priorité P1/P2/P3 ne doit jamais être déduite directement de la présence d’un mot-clé.

## Mise en place

Plusieurs familles de marqueurs sont utilisées :

- termes médicaux ;
- symptômes aigus ;
- signes vitaux ;
- contexte aigu ;
- risque suicidaire ou auto-agressif ;
- intoxication ;
- hémorragie sévère ;
- urgence cardiaque ;
- urgence neurologique ;
- détresse respiratoire ;
- obstétrique ;
- pédiatrie ;
- catégories de santé sensibles.

Chaque observation peut recevoir plusieurs catégories d’audit.

Les exemples présentant plusieurs marqueurs de triage peuvent être identifiés comme `TRIAGE_MARKERS_STRONG`.

## Résultats

Les corpus présentent des niveaux variables de pertinence pour le triage.

MediQAl et UltraMedical contiennent de nombreux exemples potentiellement intéressants pour des situations aiguës. Les contenus à risque sont également présents, ce qui est nécessaire à la couverture d’un système de triage.

`TRIAGE_MARKERS_STRONG` signifie **forte présence de marqueurs associés au triage**, et non « urgence critique validée ».

## Projection pour le nettoyage

Les marqueurs serviront à constituer un vivier de candidats.

Les exemples potentiellement aigus pourront être davantage représentés parmi les candidats à valider, mais l’attribution finale P1/P2/P3 reposera sur la pipeline clinique : protocole FRENCH, règles déterministes, convergence des connaissances et revue lorsque nécessaire.

---

# 10. Audit RGPD et confidentialité

## Point

Les corpus médicaux peuvent contenir des informations permettant potentiellement d’identifier une personne.

Il faut cependant distinguer :

- une **PII potentielle** ;
- une **information médicale sensible** ;
- une donnée réellement identifiante nécessitant une action.

## Décision

Mettre en place une détection bilingue combinant Presidio, spaCy et des règles personnalisées.

Une détection automatique reste un **candidat à vérifier**. Elle ne constitue pas à elle seule la preuve qu’une donnée personnelle réelle est présente.

## Mise en place

Presidio est configuré pour le français et l’anglais.

Les catégories recherchées comprennent notamment :

- personne ;
- email ;
- téléphone ;
- adresse IP ;
- URL ;
- identifiant ;
- numéro professionnel médical ;
- date de naissance ;
- introduction explicite d’identité ;
- dates et localisations comme signaux nécessitant éventuellement une revue.

Des expressions régulières personnalisées complètent les reconnaisseurs standards.

Pour réduire le coût de l’analyse, chaque texte unique est analysé une seule fois grâce à une empreinte stable.

Les résultats techniques conservent les catégories, positions et scores sans recopier la valeur brute détectée.

Des aperçus masqués peuvent être générés pour la revue.

## Résultats

La majorité des corpus présente très peu de candidats PII.

MedQuad-KV concentre davantage de signalements que les autres sources et nécessite une attention particulière. UltraMedical-Preference comporte également quelques cas à examiner.

Les catégories de santé sensibles sont volontairement conservées séparément. Leur présence est normale dans un corpus médical et ne constitue pas un motif de suppression.

## Projection pour le nettoyage

Chaque candidat pertinent sera traité selon une décision explicite :

- `approve` si le signal est un faux positif ou non identifiant ;
- `redact` si l’information peut être anonymisée sans dégrader le contenu clinique ;
- `exclude` si la confidentialité ne peut pas être garantie ;
- `review` lorsqu’une décision humaine reste nécessaire.

Les transformations devront rester traçables et les données brutes séparées des données nettoyées.

---

# 11. Audit spécifique DPO

## Point

Le DPO n’apprend pas une réponse absolue : il apprend une **préférence** entre une réponse `chosen` et une réponse `rejected`.

Une paire mal construite peut donc enseigner un comportement incorrect même si sa structure JSON est valide.

## Décision

Auditer séparément la structure, les répétitions, les inversions et les conflits de rôle des paires UltraMedical-Preference.

La réponse `chosen` n’est jamais considérée comme médicalement correcte par définition.

## Mise en place

Les prompts, réponses choisies et rejetées sont normalisés.

Plusieurs contrôles sont réalisés :

- prompt manquant ;
- `chosen` manquant ;
- `rejected` manquant ;
- `chosen == rejected` ;
- déséquilibre important de longueur ;
- prompts répétés ;
- triplets répétés ;
- inversion `chosen/rejected` ;
- même réponse utilisée dans les deux rôles pour un même prompt ;
- distribution des types de labels ;
- répétition des identifiants de prompts ;
- disponibilité des métadonnées et de la réponse de référence lorsqu’elle existe.

Deux empreintes sont particulièrement utiles :

- `triplet_hash` conserve la direction `chosen → rejected` ;
- `unordered_pair_hash` ignore l’ordre des deux réponses et permet de repérer les inversions.

## Résultats

La structure générale d’UltraMedical-Preference est exploitable.

Les prompts répétés sont fréquents mais ne constituent pas automatiquement un défaut : un même prompt peut servir à plusieurs comparaisons distinctes.

Les triplets strictement répétés, inversions et conflits de rôle sont en revanche des candidats directs au nettoyage.

Le déséquilibre de longueur est traité comme un **signal susceptible de contribuer à un biais de préférence**, et non comme la preuve d’un biais à lui seul.

## Projection pour le nettoyage

Le dataset DPO final devra :

- conserver uniquement des paires complètes ;
- supprimer les triplets réellement dupliqués ;
- résoudre les inversions ;
- éliminer ou revoir les conflits de rôle ;
- vérifier la cohérence médicale de la préférence ;
- préserver la traçabilité vers l’exemple source.

---

# 12. Audit des fuites entre splits

## Point

Un modèle ne doit pas être évalué sur des exemples qu’il a déjà vus, directement ou sous une forme équivalente, pendant l’entraînement.

La fuite peut concerner une ligne exacte, une question normalisée ou un groupe logique lié au même cas.

## Décision

Mesurer les chevauchements avant de reconstruire les splits finaux.

Les comparaisons sont effectuées entre :

- train / validation ;
- train / test ;
- validation / test.

## Mise en place

Les datasets appartenant à une même famille sont regroupés dans un `corpus_group`.

Deux niveaux de comparaison sont utilisés :

- `input_hash` pour le contenu normalisé ;
- `group_hash` pour les groupes qui doivent rester indivisibles.

Un contrôle supplémentaire recherche les contenus identiques présents dans plusieurs sources.

## Résultats

Les niveaux de chevauchement diffèrent selon les corpus.

FrenchMedMCQA présente une séparation globalement propre, tandis que MediQAl et UltraMedical-Preference nécessitent davantage d’attention avant la création des jeux finaux.

L’objectif de cet audit n’est pas de supprimer immédiatement les lignes concernées, mais d’identifier les groupes qui devront être affectés à un seul split.

## Projection pour le nettoyage

La déduplication sera réalisée **avant** la création des splits définitifs.

Les observations appartenant au même groupe clinique ou à une paire bilingue resteront dans le même split afin d’éviter qu’une version d’un cas soit utilisée pour l’entraînement et une autre pour l’évaluation.

---

# 13. Traçabilité, intégrité et export des rapports

## Point

Un audit n’est réellement exploitable que si ses résultats peuvent être reproduits et reliés aux fichiers sources utilisés.

## Décision

Centraliser les rapports et produire un manifeste final plutôt que sauvegarder des fichiers indépendamment dans chaque cellule.

## Mise en place

Une fonction calcule le SHA-256 des fichiers bruts par blocs afin de vérifier leur identité sans les charger entièrement en mémoire.

Les rapports produits par les différentes sections sont centralisés dans `REPORTS`.

Le manifeste final conserve notamment :

- date de génération ;
- seed ;
- tokenizer utilisé pour l’audit ;
- volumes des fichiers bruts ;
- empreintes SHA-256 ;
- nombre de lignes canoniques ;
- liste des rapports générés ;
- décisions encore ouvertes.

Les volumes relus sont comparés aux volumes initiaux afin de vérifier que l’audit n’a pas altéré les données.

## Résultats

Le notebook fournit une trace technique de l’état des corpus au moment de l’audit.

Les données sources restent intactes et les résultats de chaque contrôle peuvent être reliés à une version précise des fichiers.

## Projection pour le nettoyage

Le notebook de nettoyage devra conserver le même principe :

**source → décision → transformation → dataset final**

Les fichiers nettoyés et les futurs datasets SFT/DPO devront disposer de leurs propres empreintes et manifestes.

---

# 14. Décisions issues de l’audit

| Point audité | Décision |
|---|---|
| Schémas différents | Conserver des traitements spécifiques SFT/DPO derrière une représentation d’audit commune |
| Contexte MediQAl | Le conserver lorsqu’il est nécessaire, ne jamais l’inventer |
| QCM | Normaliser les positions A-E en préservant réponse unique/multiple |
| Textes atypiques | Signaler puis revoir, pas de suppression sur la seule longueur |
| Langue | Conserver la langue attendue et contrôler les divergences |
| Tokens | Tenir compte des valeurs extrêmes avant tokenisation finale |
| Doublons | Dédupliquer les exemples réellement équivalents avant les splits |
| Quasi-doublons | Examiner leur valeur ajoutée avant suppression |
| Contradictions QCM | Revue avant utilisation |
| Réponses ouvertes multiples | Ne pas les considérer automatiquement comme contradictoires |
| Diversité | Ne pas reproduire les proportions brutes des sources |
| Marqueurs de triage | Utiliser pour caractériser/sélectionner, jamais comme label clinique direct |
| Contenus à risque | Les conserver lorsqu’ils sont médicalement légitimes |
| PII | Revue, anonymisation ou exclusion selon le cas |
| Santé sensible | Ne pas confondre avec une PII |
| DPO | Vérifier la préférence médicale, pas seulement la structure |
| Fuites | Résoudre avant la construction des splits |
| MedQuad-KV | Ne pas intégrer définitivement à l’entraînement tant que provenance/licence non confirmées |
| Traçabilité | Conserver hashes, rapports, décisions et manifestes |

---

# 15. Projection vers le nettoyage et la construction du SFT/DPO

L’audit est une étape de diagnostic. La phase suivante applique les décisions identifiées.

## Étape 1 — Nettoyage déterministe

Appliquer les règles ne nécessitant pas de jugement clinique :

- normalisation des formats ;
- suppression des doublons certains ;
- contrôle des champs obligatoires ;
- résolution des problèmes structurels ;
- traitement des PII validées ;
- exclusion des observations techniquement inutilisables.

## Étape 2 — Revue des cas ambigus

Examiner :

- contradictions ;
- quasi-doublons ;
- PII candidates ;
- préférences DPO ambiguës ;
- contenus dépendants d’un contexte insuffisant ;
- situations médicales dont la priorité ne peut pas être déterminée automatiquement.

## Étape 3 — Attribution clinique

Pour les candidats SFT pertinents pour le triage, déterminer la priorité cible à l’aide de la pipeline clinique :

1. informations patient ;
2. connaissances médicales extraites ;
3. règles déterministes ;
4. protocole FRENCH ;
5. convergence des sources de décision ;
6. revue humaine lorsqu’une décision automatique n’est pas suffisamment fiable.

Les niveaux finaux visés sont P1, P2 et P3.

## Étape 4 — Sélection des 5 000 exemples SFT

Construire un dataset de haute qualité avec :

- 2 500 exemples français ;
- 2 500 exemples anglais ;
- équilibre contrôlé P1/P2/P3 ;
- diversité des situations cliniques ;
- limitation des répétitions ;
- texte source complet lorsque nécessaire ;
- priorité validée ;
- instructions cohérentes avec le niveau d’urgence.

Le dataset de sélection pourra conserver davantage de métadonnées que le dataset réellement envoyé au modèle.

## Étape 5 — Construction des splits

Les splits seront reconstruits après déduplication.

Les groupes cliniques et paires bilingues resteront indivisibles.

La séparation train / validation / test devra être stratifiée autant que possible sur les priorités sans créer de fuite.

## Étape 6 — Dataset SFT compact

Seuls les champs utiles au modèle seront conservés dans le dataset d’entraînement.

Le format cible sera un échange conversationnel structuré, par exemple :

```text
system    → rôle et règles du système de triage
user      → questionnaire / situation clinique
assistant → niveau d’urgence + instructions
```

## Étape 7 — Dataset DPO

Après le SFT, les paires de préférences validées seront utilisées pour améliorer le comportement du modèle.

Le DPO devra favoriser les réponses médicalement plus appropriées et plus sûres, sans apprendre des préférences dues uniquement au style ou à la longueur.

## Étape 8 — Évaluation

L’évaluation finale devra être séparée de l’audit des sources.

Elle portera notamment sur :

- accuracy ;
- précision, rappel et F1 par classe ;
- matrice de confusion ;
- rappel P1 ;
- erreurs P1 → P2/P3 ;
- cohérence des instructions ;
- cas de sécurité clinique ;
- hallucinations ou recommandations inappropriées.

Pour un système de triage, le rappel de la classe la plus urgente devra être surveillé particulièrement afin d’identifier les urgences sous-estimées.

---

# 16. Limites de l’audit

L’audit fournit des **signaux de qualité**, pas une certification médicale ou réglementaire.

Plusieurs limites doivent être conservées dans l’interprétation :

- la détection de langue peut se tromper sur les textes courts ou très médicaux ;
- les regex et Presidio peuvent produire des faux positifs ou faux négatifs ;
- un marqueur de triage ne détermine pas une priorité clinique ;
- une similarité textuelle élevée ne prouve pas qu’un exemple est inutile ;
- plusieurs réponses ouvertes peuvent être médicalement compatibles ;
- `chosen` n’est pas nécessairement médicalement supérieure à `rejected` ;
- un déséquilibre de longueur DPO n’est pas, à lui seul, la preuve d’un biais ;
- l’absence de PII détectée ne démontre pas l’absence absolue de donnée personnelle ;
- l’audit des données ne remplace pas l’évaluation du modèle après fine-tuning.

Ces limites justifient le maintien d’une revue humaine pour les cas sensibles ou ambigus.

---

# 17. Conclusion

L’audit constitue la **porte d’entrée qualité** de la pipeline de fine-tuning.

Il permet de passer d’un ensemble de corpus médicaux hétérogènes à une connaissance structurée de leurs forces, limites et risques, sans modifier les données brutes.

La démarche suivie est :

```text
Corpus bruts
    ↓
Documentation et provenance
    ↓
Chargement contrôlé
    ↓
Canonicalisation pour l’audit
    ↓
Structure / QCM / texte
    ↓
Doublons / contradictions / diversité
    ↓
Pertinence médicale et triage
    ↓
RGPD / confidentialité
    ↓
Audit spécifique DPO
    ↓
Fuites entre splits
    ↓
Rapports + SHA-256 + manifeste
    ↓
Décisions de nettoyage
    ↓
Sélection et validation clinique
    ↓
SFT 5 000 exemples
    ↓
DPO
    ↓
Évaluation clinique du modèle
```

À l’issue de cette étape, l’audit est suffisamment complet pour un **POC de fine-tuning médical**. Il ne signifie pas que toutes les données sont immédiatement prêtes à entraîner le modèle : il fournit précisément les éléments nécessaires pour construire, de manière traçable, le nettoyage et la sélection qui suivent.

La prochaine étape consiste donc à transformer les constats de l’audit en **règles de nettoyage reproductibles**, puis à constituer les datasets SFT et DPO finaux sans fuite, avec une validation clinique explicite des cibles utilisées pour l’apprentissage.
