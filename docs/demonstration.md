# Démonstration de l'API de triage médical

> **Cadre de démonstration**\
> Ces scénarios sont fictifs et servent uniquement à démontrer le
> fonctionnement du POC.\
> L'API classe les situations en **P1 = urgence élevée**, **P2 = urgence
> modérée**, **P3 = situation différable**.\
> Les priorités attendues ci-dessous sont des attentes de démonstration,
> et non une validation médicale du modèle.

## Objectif

Tester l'endpoint `POST /triage` avec des situations variées : détresse
respiratoire, traumatisme, douleur, infection, symptômes modérés et
situations différables.

Pour chaque scénario, saisir le JSON dans Swagger puis comparer la
priorité retournée par le modèle à la priorité attendue pour la
démonstration.

## Cas 1 --- Douleur thoracique avec signes de gravité

**Priorité attendue pour la démonstration : P1**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient de 68 ans présentant depuis 30 minutes une douleur thoracique brutale et persistante.",
  "answer": "Douleur intense avec dyspnée, sueurs et malaise."
}
```

**Intérêt du cas :** vérifier la réaction du modèle devant plusieurs
signes potentiellement graves associés à une douleur thoracique.

## Cas 2 --- Détresse respiratoire importante

**Priorité attendue pour la démonstration : P1**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patiente de 72 ans amenée aux urgences pour aggravation rapide d'une difficulté respiratoire.",
  "answer": "Dyspnée majeure au repos, difficulté à parler normalement et coloration bleutée des lèvres."
}
```

**Intérêt du cas :** présenter une urgence respiratoire avec des
éléments explicites de gravité.

## Cas 3 --- Traumatisme avec déficit neurologique

**Priorité attendue pour la démonstration : P1**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient victime d'un accident de la route avec traumatisme cervical.",
  "answer": "Douleur cervicale importante et diminution récente de la force des quatre membres."
}
```

**Intérêt du cas :** tester un scénario traumatique associé à un déficit
neurologique.

## Cas 4 --- Douleur abdominale sans signe de choc

**Priorité attendue pour la démonstration : P2**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patiente de 35 ans consultant pour une douleur abdominale apparue depuis plusieurs heures.",
  "answer": "Douleur persistante avec nausées, sans perte de connaissance ni difficulté respiratoire."
}
```

**Intérêt du cas :** tester une situation nécessitant une évaluation
médicale sans signe majeur de détresse décrit.

## Cas 5 --- Fièvre et symptômes infectieux

**Priorité attendue pour la démonstration : P2**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient de 42 ans présentant de la fièvre depuis deux jours.",
  "answer": "Température à 39,2 °C, frissons, toux productive et fatigue importante. Patient conscient et sans détresse respiratoire."
}
```

**Intérêt du cas :** montrer une situation infectieuse symptomatique
mais sans signe de détresse explicitement fourni.

## Cas 6 --- Entorse probable de cheville

**Priorité attendue pour la démonstration : P3**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient de 27 ans ayant tourné la cheville en marchant il y a deux heures.",
  "answer": "Douleur modérée et léger gonflement. Le patient peut marcher avec difficulté. Pas de plaie ni de déformation visible."
}
```

**Intérêt du cas :** vérifier que le modèle peut différencier une
atteinte traumatique mineure d'un traumatisme grave.

## Cas 7 --- Symptômes ORL légers

**Priorité attendue pour la démonstration : P3**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient de 31 ans présentant un mal de gorge et un écoulement nasal depuis deux jours.",
  "answer": "Douleur légère, température à 37,6 °C, respiration normale et état général conservé."
}
```

**Intérêt du cas :** disposer d'un exemple clairement différable avec
peu de signes de gravité décrits.

## Cas 8 --- Version anglaise, urgence élevée

**Priorité attendue pour la démonstration : P1**

``` json
{
  "language": "en",
  "question": "What is the triage priority?",
  "context": "A 64-year-old patient presents with sudden onset of severe chest pain.",
  "answer": "Persistent chest pain associated with shortness of breath, sweating and faintness."
}
```

**Intérêt du cas :** démontrer que le modèle accepte également les
entrées en anglais.

## Cas 9 --- Version anglaise, situation différable

**Priorité attendue pour la démonstration : P3**

``` json
{
  "language": "en",
  "question": "What is the triage priority?",
  "context": "A 25-year-old patient presents with mild wrist pain after a minor fall yesterday.",
  "answer": "Mild swelling, normal finger movement, no visible deformity and no open wound."
}
```

**Intérêt du cas :** compléter la démonstration bilingue avec une
situation de faible urgence.

## Cas 10 --- Cas volontairement ambigu

**Priorité attendue : à observer**

``` json
{
  "language": "fr",
  "question": "Quelle est la priorité de triage ?",
  "context": "Patient de 55 ans consultant pour des vertiges apparus dans la matinée.",
  "answer": "Vertiges persistants et nausées, sans autre information clinique disponible."
}
```

**Intérêt du cas :** montrer une limite importante du POC. Lorsque les
informations sont insuffisantes ou ambiguës, le modèle doit malgré tout
choisir P1, P2 ou P3. Ce scénario permet donc de discuter de
l'incertitude et de la nécessité d'une supervision clinique.

## Déroulement conseillé de la démonstration

1.  Ouvrir `http://127.0.0.1:8000/`, qui redirige vers Swagger.
2.  Sélectionner `POST /triage`.
3.  Cliquer sur **Try it out**.
4.  Copier un scénario JSON.
5.  Cliquer sur **Execute**.
6.  Montrer la priorité, la version du modèle et la latence retournées.
7.  Terminer par le cas ambigu pour présenter les limites du POC.

## Points à rappeler pendant la démonstration

Le modèle final est un **POC d'assistance au triage**, pas un dispositif
médical validé. Il produit obligatoirement une classe P1, P2 ou P3 à
partir des informations fournies. Une sortie techniquement valide ne
garantit donc pas qu'une décision soit médicalement correcte.

Les scénarios de cette démonstration sont fictifs. Les priorités
indiquées comme « attendues » servent à vérifier le comportement attendu
du POC et à rendre la démonstration lisible. Elles ne constituent pas
des annotations validées par des médecins.
