| Paramètre              |                        Try 1 |
| ---------------------- | ---------------------------: |
| Modèle                 |       `Qwen/Qwen3-1.7B-Base` |
| Méthode                |                  QLoRA 4-bit |
| Epochs                 |                        **3** |
| Learning rate          |                     **2e-4** |
| Batch GPU              |                        **1** |
| Gradient accumulation  |                        **8** |
| Batch effectif         |                        **8** |
| Max length             |                     **1024** |
| Optimiseur             |           `paged_adamw_8bit` |
| Scheduler              |                     `cosine` |
| Max grad norm          |                      **1.0** |
| Gradient checkpointing |                       `True` |
| BF16                   |            si GPU compatible |
| FP16                   |                     fallback |
| LoRA `r`               |                       **16** |
| LoRA alpha             |                       **32** |
| LoRA dropout           |                     **0.05** |
| LoRA modules           |       q/k/v/o + gate/up/down |
| Seed                   |                       **42** |
| Eval                   |       tous les **100 steps** |
| Save                   |       tous les **100 steps** |
| Logging                |        tous les **10 steps** |
| Loss                   | CE assistant token par token |
| Prompt                 |                masqué `-100` |
| Priorité               |                       **×1** |
| Explication            |                       **×1** |


| Paramètre              |                         Try 2 |
| ---------------------- | ----------------------------: |
| Modèle                 |        `Qwen/Qwen3-1.7B-Base` |
| Méthode                |                   QLoRA 4-bit |
| Epochs                 |                         **3** |
| Learning rate          |                      **2e-4** |
| Batch GPU              |                         **1** |
| Gradient accumulation  |                         **8** |
| Batch effectif         |                         **8** |
| Max length             |                      **1024** |
| Optimiseur             |            `paged_adamw_8bit` |
| Scheduler              |                      `cosine` |
| Max grad norm          |                       **1.0** |
| Gradient checkpointing |                        `True` |
| LoRA `r`               |                        **16** |
| LoRA alpha             |                        **32** |
| LoRA dropout           |                      **0.05** |
| Seed                   |                        **42** |
| Eval / save            |                 **100 steps** |
| Logging                |                  **10 steps** |
| Prompt                 |                 masqué `-100` |
| Explication            |                        **×1** |
| P1                     |                       **×10** |
| P2                     |                       **×10** |
| P3                     |                       **×10** |
| Sélection checkpoint   |              **Triage Score** |
| Poids Triage Score     | **P1 50 %, P2 25 %, P3 25 %** |



## Différences entre Try 1 et Try 2

| Paramètre | Try 1 | Try 2 |
|---|---:|---:|
| Poids priorité P1 | ×1 | ×10 |
| Poids priorité P2 | ×1 | ×10 |
| Poids priorité P3 | ×1 | ×10 |
| Poids explication | ×1 | ×1 |
| Sélection du meilleur checkpoint | Recall macro | Triage Score |
| Pondération du Triage Score | — | P1 50 % / P2 25 % / P3 25 % |

### Paramètres inchangés

- Modèle : `Qwen/Qwen3-1.7B-Base`
- Méthode : QLoRA 4-bit
- Dataset et splits identiques
- Seed : `42`
- Epochs : `3`
- Learning rate : `2e-4`
- Batch GPU : `1`
- Gradient accumulation : `8`
- Batch effectif : `8`
- Max length : `1024`
- Optimiseur : `paged_adamw_8bit`
- Scheduler : `cosine`
- LoRA `r` : `16`
- LoRA alpha : `32`
- LoRA dropout : `0.05`

### Synthèse

**Try 1** utilise une Cross-Entropy standard : tous les tokens de la réponse assistant ont le même poids.

**Try 2** renforce ×10 les tokens correspondant à la priorité `P1`, `P2` ou `P3`, tandis que les tokens de l'explication restent à ×1.

Le Try 2 modifie également le critère de sélection du meilleur checkpoint :

`Triage Score = 0.50 × Recall P1 + 0.25 × Recall P2 + 0.25 × Recall P3`

Il y a donc **deux différences expérimentales** entre Try 1 et Try 2 : la pondération de la loss et le critère de sélection du checkpoint.

## Try 3 — Renforcement spécifique de P1

### Modification par rapport au Try 2

| Paramètre | Try 2 | Try 3 |
|---|---:|---:|
| Poids priorité P1 | ×10 | **×15** |
| Poids priorité P2 | ×10 | ×10 |
| Poids priorité P3 | ×10 | ×10 |
| Poids explication | ×1 | ×1 |
| Sélection du meilleur checkpoint | Triage Score | Triage Score |
| Pondération du Triage Score | P1 50 % / P2 25 % / P3 25 % | P1 50 % / P2 25 % / P3 25 % |

### Paramètres inchangés

- Modèle : `Qwen/Qwen3-1.7B-Base`
- Méthode : QLoRA 4-bit
- Dataset et splits identiques
- Seed : `42`
- Epochs : `3`
- Learning rate : `2e-4`
- Batch GPU : `1`
- Gradient accumulation : `8`
- Batch effectif : `8`
- Max length : `1024`
- Optimiseur : `paged_adamw_8bit`
- Scheduler : `cosine`
- LoRA `r` : `16`
- LoRA alpha : `32`
- LoRA dropout : `0.05`
- Évaluation : tous les `100` steps
- Sauvegarde : tous les `100` steps

### Loss

Try 3 conserve le renforcement des tokens de priorité introduit dans Try 2, mais donne davantage de poids aux exemples dont la priorité correcte est `P1`.

- Exemple P1 : token de priorité ×15, explication ×1
- Exemple P2 : token de priorité ×10, explication ×1
- Exemple P3 : token de priorité ×10, explication ×1

### Sélection du checkpoint

Le meilleur checkpoint reste sélectionné avec :

`Triage Score = 0.50 × Recall P1 + 0.25 × Recall P2 + 0.25 × Recall P3`

### Objectif

L'objectif du Try 3 est d'augmenter spécifiquement le **Recall P1**, prioritaire pour le triage, sans dégrader excessivement les performances sur P2 et P3.

Contrairement au passage Try 1 → Try 2, le passage **Try 2 → Try 3 ne modifie qu'une variable expérimentale : le poids de P1 passe de ×10 à ×15**.