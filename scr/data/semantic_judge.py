import json
from pathlib import Path

import torch

from scr.config import CONFIG
from scr.data.cleaning import (
    assemble_input,
    clean_text
)
from scr.data.selection import SFTSourceConfig
from scr.data.semantic import (
    SemanticConfig,
    semantic_verdict
)
from scr.utils.hashing import digest
from scr.utils.serialization import read_jsonl

class JudgeConfig:
    """Configuration du juge sémantique local."""

    MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    REVISION = "main"
    VERSION = "20260905.v1"

    BATCH_SIZE = 1
    MAX_NEW_TOKENS = 96
    RETRY_MAX_NEW_TOKENS = 128

    CONFIDENCE_VALUES = {
        'HIGH',
        'MEDIUM',
        'LOW'
    }

    SCORE_FIELDS = (
        'completeness',
        'relevance',
        'alignment',
        'safety'
    )

    FIELDS = {
        *SCORE_FIELDS,
        'confidence',
        'reason'
    }
    
    # Prompt

class SemanticSourcePromptBuilder:
    """Construit le prompt du juge sémantique source."""

    # Initialise le constructeur avec le record source à examiner.
    # Aucun payload final ni information de TRIAGE n'est requis ici.
    def __init__(
        self,
        record: dict
    ) -> None:
        """Initialise le constructeur."""
        self.record = record

    # Demande uniquement un second avis sur la qualité sémantique source.
    # Le modèle ne détermine ni urgence, ni priorité, ni décision finale.
    def build(
        self
    ) -> str:
        """Construit le prompt."""
        source_input = assemble_input(self.record)
        target = self._target()

        return f"""
Tu contrôles la qualité d'un exemple médical source.

Règles :
- N'invente aucune information.
- Ne réécris pas la réponse.
- Ne fournis aucun raisonnement détaillé.
- Une réponse courte peut être correcte.
- Évalue uniquement la qualité de l'exemple.
- Ne détermine aucune urgence médicale.
- Ne détermine aucune priorité de triage.
- Ne modifie pas le verdict déterministe.

Valeurs autorisées :
PASS, REVIEW ou FAIL.

Retourne uniquement ce JSON :

{{
  "completeness": "PASS|REVIEW|FAIL",
  "relevance": "PASS|REVIEW|FAIL",
  "alignment": "PASS|REVIEW|FAIL",
  "safety": "PASS|REVIEW|FAIL",
  "confidence": "HIGH|MEDIUM|LOW",
  "reason": "une phrase courte"
}}

ENTRÉE :
{source_input}

CIBLE :
{target}
""".strip()

    # Construit une représentation lisible de la cible source attendue.
    # Les QCM utilisent les réponses correctes et les réponses ouvertes leur texte.
    def _target(
        self
    ) -> str:
        """Construit la cible source."""
        answer = clean_text(
            self.record.get('answer')
        )

        if answer:
            return answer

        choices = self.record.get('choices') or {}
        labels = self.record.get('labels') or []

        selected = [
            clean_text(choices.get(label))
            for label in labels
            if clean_text(choices.get(label))
        ]

        if selected:
            return ' | '.join(selected)

        return 'Aucune cible exploitable.'


# Juge

class SemanticJudge:
    """Exécute le second avis sémantique local."""

    # Initialise le juge avec les objets Transformers déjà chargés.
    # Le modèle et le tokenizer sont réutilisés pour toutes les évaluations.
    def __init__(
        self,
        tokenizer,
        model
    ) -> None:
        """Initialise le juge."""
        self.tokenizer = tokenizer
        self.model = model

    # Génère déterministement une sortie courte pour chaque record fourni.
    # Seuls les tokens générés après le prompt sont conservés.
    def generate(
        self,
        records: list[dict],
        max_new_tokens: int
    ) -> list[str]:
        """Génère les décisions."""
        if not records:
            return []

        conversations = [
            [{
                'role': 'user',
                'content': SemanticSourcePromptBuilder(
                    record
                ).build()
            }]
            for record in records
        ]

        inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors='pt'
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        prompt_length = inputs['input_ids'].shape[1]

        return [
            self.tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            ).strip()
            for output in generated
        ]

    # Évalue chaque record puis valide strictement le JSON retourné.
    # Une sortie invalide bénéficie d'une seule nouvelle tentative.
    def evaluate_batch(
        self,
        records: list[dict]
    ) -> list[dict]:
        """Évalue un lot."""
        outputs = self.generate(
            records,
            JudgeConfig.MAX_NEW_TOKENS
        )

        return [
            self._evaluate(record, output)
            for record, output in zip(
                records,
                outputs,
                strict=True
            )
        ]

    # Valide la première sortie puis effectue une seconde tentative si nécessaire.
    # Deux échecs produisent uniquement un REVIEW technique explicite.
    def _evaluate(
        self,
        record: dict,
        output: str
    ) -> dict:
        """Sécurise une évaluation."""
        try:
            return self._parse(output)

        except (ValueError, KeyError, TypeError) as first_error:
            retry = self.generate(
                [record],
                JudgeConfig.RETRY_MAX_NEW_TOKENS
            )[0]

            try:
                return self._parse(retry)

            except (ValueError, KeyError, TypeError) as retry_error:
                return self._fallback(
                    f'{type(first_error).__name__} / '
                    f'{type(retry_error).__name__}'
                )

    # Extrait et valide strictement le JSON produit par le modèle.
    # Le verdict global est toujours recalculé localement.
    def _parse(
        self,
        output: str
    ) -> dict:
        """Parse une décision."""
        result = self._extract_json(output)

        if set(result) != JudgeConfig.FIELDS:
            raise ValueError(
                'Schéma du juge invalide.'
            )

        scores = {
            field: clean_text(
                str(result[field])
            ).upper()
            for field in JudgeConfig.SCORE_FIELDS
        }

        if any(
            value not in SemanticConfig.SCORE_VALUES
            for value in scores.values()
        ):
            raise ValueError(
                'Verdict sémantique invalide.'
            )

        confidence = clean_text(
            str(result['confidence'])
        ).upper()

        reason = (
            clean_text(result['reason'])
            if isinstance(result['reason'], str)
            else ''
        )

        if confidence not in JudgeConfig.CONFIDENCE_VALUES:
            raise ValueError(
                f'Confiance invalide : {confidence!r}.'
            )

        if not reason:
            raise ValueError(
                'Justification absente.'
            )

        return {
            **scores,
            'confidence': confidence,
            'verdict': semantic_verdict(scores),
            'reason': reason
        }

    # Recherche le premier objet JSON valide dans la sortie du modèle.
    # Le texte parasite éventuel est ignoré sans assouplir le schéma attendu.
    def _extract_json(
        self,
        value: str
    ) -> dict:
        """Extrait un objet JSON."""
        decoder = json.JSONDecoder()

        for index, char in enumerate(value.strip()):
            if char != '{':
                continue

            try:
                result, _ = decoder.raw_decode(
                    value.strip()[index:]
                )
            except json.JSONDecodeError:
                continue

            if isinstance(result, dict):
                return result

        raise ValueError(
            'Aucun JSON valide retourné.'
        )

    # Produit un REVIEW technique lorsque les deux générations sont invalides.
    # Aucun jugement sémantique n'est inventé dans ce cas.
    def _fallback(
        self,
        reason: str
    ) -> dict:
        """Construit le fallback."""
        return {
            **{
                field: SemanticConfig.REVIEW
                for field in JudgeConfig.SCORE_FIELDS
            },
            'confidence': 'LOW',
            'verdict': SemanticConfig.REVIEW,
            'reason': (
                'Sortie invalide après deux tentatives : '
                f'{reason}.'
            )
        }


# Cache

class SemanticJudgeCache:
    """Gère le cache versionné du juge."""

    # Initialise le cache persistant avec son chemin JSONL.
    # La version du juge invalide automatiquement les anciennes décisions.
    def __init__(
        self,
        path: Path
    ) -> None:
        """Initialise le cache."""
        self.path = path

    # Construit une clé liée au record et à son contenu nettoyé.
    # Toute modification du contenu ou du juge provoque une nouvelle évaluation.
    def key(
        self,
        record_id: str,
        content_hash: str
    ) -> tuple:
        """Construit une clé de cache."""
        return (
            record_id,
            content_hash,
            JudgeConfig.VERSION
        )

    # Charge uniquement les décisions compatibles avec le juge courant.
    # Une clé dupliquée est considérée comme une corruption du cache.
    def load(
        self
    ) -> dict:
        """Charge le cache."""
        if not self.path.is_file():
            return {}

        cache = {}

        for decision in read_jsonl(self.path):
            if (
                decision.get('judge_version') != JudgeConfig.VERSION
                or decision.get('judge_model') != JudgeConfig.MODEL
                or decision.get('judge_revision') != JudgeConfig.REVISION
            ):
                continue

            key = self.key(
                decision.get('id'),
                decision.get('clean_content_hash')
            )

            if not all(key[:2]) or key in cache:
                raise ValueError(
                    'Cache du juge invalide.'
                )

            cache[key] = decision

        return cache

    # Ajoute immédiatement une décision validée au cache JSONL.
    # La persistance après chaque inférence limite les pertes en cas d'arrêt.
    def append(
        self,
        decision: dict
    ) -> None:
        """Ajoute une décision."""
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.path.open(
            'a',
            encoding='utf-8'
        ) as file:
            file.write(
                json.dumps(
                    decision,
                    ensure_ascii=False,
                    sort_keys=True
                ) + '\n'
            )


# Traitement

class SemanticJudgeProcessor:
    """Exécute et rattache le second avis LLM."""

    # Initialise le processeur avec les records à revoir et le cache.
    # Les décisions déterministes restent inchangées pendant tout le traitement.
    def __init__(
        self,
        records: list[dict],
        judge: SemanticJudge,
        cache: SemanticJudgeCache
    ) -> None:
        """Initialise le processeur."""
        self.records = records
        self.judge = judge
        self.cache = cache

    # Évalue uniquement les records absents du cache puis rattache les avis.
    # Chaque décision est enregistrée immédiatement après validation.
    def process(
        self
    ) -> dict:
        """Exécute le juge."""
        decisions = self.cache.load()

        pending = [
            record
            for record in self.records
            if self.cache.key(
                record['id'],
                record['clean_content_hash']
            ) not in decisions
        ]

        print(
            f'{len(self.records):,} REVIEW | '
            f'{len(self.records) - len(pending):,} cache | '
            f'{len(pending):,} à évaluer'
        )

        for start in range(
            0,
            len(pending),
            JudgeConfig.BATCH_SIZE
        ):
            batch = pending[
                start:start + JudgeConfig.BATCH_SIZE
            ]

            judgements = self.judge.evaluate_batch(batch)

            for record, judgement in zip(
                batch,
                judgements,
                strict=True
            ):
                decision = {
                    'id': record['id'],
                    'clean_content_hash': record['clean_content_hash'],
                    'source': 'LOCAL_LLM_JUDGE',
                    'judge_model': JudgeConfig.MODEL,
                    'judge_revision': JudgeConfig.REVISION,
                    'judge_version': JudgeConfig.VERSION,
                    **judgement
                }

                self.cache.append(decision)

                decisions[
                    self.cache.key(
                        record['id'],
                        record['clean_content_hash']
                    )
                ] = decision

            print(
                f'Évaluation : '
                f'{min(start + len(batch), len(pending)):,} / '
                f'{len(pending):,}'
            )

        self._apply(decisions)

        return decisions

    # Rattache le second avis sans modifier le verdict déterministe officiel.
    # Tout désaccord est uniquement conservé comme signal d'audit.
    def _apply(
        self,
        decisions: dict
    ) -> None:
        """Rattache les avis."""
        for record in self.records:
            key = self.cache.key(
                record['id'],
                record['clean_content_hash']
            )
            decision = decisions.get(key)

            if decision is None:
                continue

            record['semantic_judge'] = {
                key: value
                for key, value in decision.items()
                if key not in {
                    'id',
                    'clean_content_hash'
                }
            }

            deterministic = (
                record.get('semantic_quality')
                or {}
            ).get('verdict')

            flags = record.setdefault('flags', set())
            flags.discard(
                'SEMANTIC_JUDGE_DISAGREEMENT'
            )

            if decision['verdict'] != deterministic:
                flags.add(
                    'SEMANTIC_JUDGE_DISAGREEMENT'
                )

# Ordre déterministe

# Produit un ordre stable pour les records envoyés au juge.
# La sélection reste reproductible sans dépendre du StratifiedSelector final.
def semantic_judge_order(
    record: dict
) -> str:
    """Produit un ordre stable."""
    return digest([
        CONFIG.seed,
        record["id"]
    ])
def semantic_judge_order(
    record: dict
) -> str:
    """Produit un ordre stable."""
    return digest([
        CONFIG.seed,
        record['id']
    ])


# Sélectionne uniquement les sources SFT encore en REVIEW déterministe.
# Les PASS restent dans le flux automatique et les FAIL restent exclus.
def is_sft_semantic_review(
    record: dict
) -> bool:
    """Identifie une source SFT à revoir."""
    if (
        record.get('semantic_quality', {}).get('verdict')
        != SemanticConfig.REVIEW
    ):
        return False

    language = record.get('language')
    family = record.get('family')

    if family not in SFTSourceConfig.ALLOWED_FAMILIES.get(
        language,
        set()
    ):
        return False

    return (
        language != 'en'
        or record.get('source') == SFTSourceConfig.EN_SOURCE
    )

def is_dpo_semantic_review(
    record: dict
) -> bool:
    """Identifie une source DPO à revoir."""
    return (
        record.get("family") == "dpo"
        and (
            record.get("semantic_quality")
            or {}
        ).get("verdict") == SemanticConfig.REVIEW
    )