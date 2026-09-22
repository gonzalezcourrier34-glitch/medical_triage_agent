import re

from scr.data.cleaning import (
    clean_text,
    key_text
)

class SemanticConfig:
    """Configuration du contrôle sémantique."""

    PASS = 'PASS'
    REVIEW = 'REVIEW'
    FAIL = 'FAIL'

    SCORE_VALUES = {
        PASS: 1.0,
        REVIEW: 0.5,
        FAIL: 0.0
    }

    BLOCKING_OPEN_FLAGS = {
        'UNINFORMATIVE_ANSWER',
        'RESOURCE_ONLY_ANSWER',
        'NAVIGATION_ONLY_ANSWER'
    }

    REVIEW_OPEN_FLAGS = {
        'QUESTION_AS_ANSWER',
        'VERY_SHORT_OPEN_ANSWER'
    }

    AUTONOMY_FLAGS = {
        'NEEDS_CONTEXT',
        'EXTERNAL_REFERENCE'
    }

    TEXT_REVIEW_FLAGS = {
        'TEXT_ANOMALY'
    }


class OpenAnswerPatterns:
    """Motifs déterministes des réponses ouvertes impropres."""

    UNINFORMATIVE = {
        'topic', 'topics',
        'resource', 'resources',
        'reference', 'references',
        'more information', 'additional information',
        'sujet', 'sujets',
        'ressource', 'ressources',
        'référence', 'références',
        'plus d informations',
        'informations complémentaires'
    }

    RESOURCE_PREFIXES = (
        'these resources address',
        'the following resources',
        'for more information',
        'additional resources',
        'resources for health professionals',
        'resources for patients',
        'you can find more information',
        'ces ressources concernent',
        'les ressources suivantes',
        'pour plus d informations',
        'ressources complémentaires',
        'ressources pour les professionnels de santé',
        'ressources pour les patients',
        'vous trouverez plus d informations'
    )

    NAVIGATION_PREFIXES = (
        'click here',
        'read more',
        'learn more',
        'see also',
        'visit ',
        'cliquez ici',
        'lire la suite',
        'en savoir plus',
        'voir aussi',
        'consultez ',
        'visitez '
    )
    



# Détermine le verdict global avec priorité aux erreurs puis aux revues.
# L'absence de signal bloquant ou incertain produit automatiquement PASS.
def semantic_verdict(scores: dict) -> str:
    """Détermine le verdict sémantique."""
    verdicts = set(scores.values())

    if SemanticConfig.FAIL in verdicts:
        return SemanticConfig.FAIL

    if SemanticConfig.REVIEW in verdicts:
        return SemanticConfig.REVIEW

    return SemanticConfig.PASS


# Convertit les dimensions sémantiques en score numérique comparable.
# Ce score mesure la qualité source et non la confiance de triage clinique.
def semantic_quality_score(scores: dict) -> float:
    """Calcule le score de qualité."""
    values = [
        SemanticConfig.SCORE_VALUES[value]
        for value in scores.values()
        if value in SemanticConfig.SCORE_VALUES
    ]

    return round(sum(values) / len(values), 4) if values else 0.0


# Validation

class SemanticQualityValidator:
    """Évalue déterministement la qualité sémantique d'un record."""

    # Initialise le validateur avec le record et ses flags existants.
    # Les contrôles complètent les audits précédents sans les réinitialiser.
    def __init__(self, record: dict) -> None:
        """Initialise le validateur."""
        self.record = record
        self.flags = record.setdefault('flags', set())

    # Recalcule les signaux déterministes puis évalue SFT source ou DPO source.
    # La famille reste utilisée ici car les records TRIAGE finaux n'existent pas encore.
    def evaluate(self) -> dict:
        """Évalue la qualité sémantique."""
        self.flags.discard('SEMANTIC_REVIEW')
        self._refresh_open_flags()

        if self.record['family'] == 'dpo':
            return self._evaluate_dpo()

        return self._evaluate_sft()

    # Enregistre l'évaluation et adapte le statut selon le verdict obtenu.
    # PASS et REVIEW restent exploitables tandis que FAIL bloque le record.
    def apply(self) -> None:
        """Applique le verdict."""
        quality = self.evaluate()
        verdict = quality['verdict']

        self.record['semantic_quality'] = quality
        self.record['status'] = (
            'SEMANTIC_QUALITY_FAILED'
            if verdict == SemanticConfig.FAIL
            else 'ELIGIBLE'
        )

        if verdict == SemanticConfig.REVIEW:
            self.flags.add('SEMANTIC_REVIEW')

    # Évalue complétude, alignement, langue, autonomie, texte et fuite de cible.
    # Le contrôle porte encore sur la donnée médicale source et non sur le triage final.
    def _evaluate_sft(self) -> dict:
        """Évalue un exemple SFT source."""
        reasons = []
        input_text = clean_text(self.record.get('input'))
        answer_text = clean_text(self.record.get('answer'))

        completeness = SemanticConfig.PASS
        alignment = SemanticConfig.PASS

        if not input_text:
            completeness = SemanticConfig.FAIL
            reasons.append('Entrée SFT vide.')

        if self.record.get('choices'):
            alignment, reason = self._qcm_alignment()

            if reason:
                reasons.append(reason)

        elif not answer_text:
            completeness = SemanticConfig.FAIL
            alignment = SemanticConfig.FAIL
            reasons.append('Réponse SFT vide.')

        elif self.record['family'] == 'open':
            blocking = (
                self.flags
                & SemanticConfig.BLOCKING_OPEN_FLAGS
            )
            review = (
                self.flags
                & SemanticConfig.REVIEW_OPEN_FLAGS
            )

            if blocking:
                alignment = SemanticConfig.FAIL
                reasons.append(
                    'Réponse ouverte impropre au SFT : '
                    f'{", ".join(sorted(blocking))}.'
                )

            elif review:
                alignment = SemanticConfig.REVIEW
                reasons.append(
                    'Réponse ouverte à vérifier : '
                    f'{", ".join(sorted(review))}.'
                )

        language, reason = self._language_quality()

        if reason:
            reasons.append(reason)

        autonomy, reason = self._autonomy_quality()

        if reason:
            reasons.append(reason)

        leakage, reason = self._target_leakage()

        if reason:
            reasons.append(reason)

        text_quality = (
            SemanticConfig.REVIEW
            if self.flags & SemanticConfig.TEXT_REVIEW_FLAGS
            else SemanticConfig.PASS
        )

        if text_quality == SemanticConfig.REVIEW:
            reasons.append('Anomalie textuelle à vérifier.')

        scores = {
            'completeness': completeness,
            'alignment': alignment,
            'language': language,
            'autonomy': autonomy,
            'text_quality': text_quality,
            'target_leakage': leakage
        }

        return self._result(scores, reasons)

    # Évalue la complétude et la cohérence structurelle des préférences DPO.
    # Cette étape ne décide pas encore quelle réponse est médicalement préférable.
    def _evaluate_dpo(self) -> dict:
        """Évalue une paire DPO source."""
        reasons = []

        prompt = clean_text(
            self.record.get('input')
            or self.record.get('question')
        )
        chosen = clean_text(
            self.record.get('preference_chosen')
            or self.record.get('chosen')
        )
        rejected = clean_text(
            self.record.get('preference_rejected')
            or self.record.get('rejected')
        )

        completeness = SemanticConfig.PASS

        for value, reason in (
            (prompt, 'Prompt DPO vide.'),
            (chosen, 'Réponse chosen vide.'),
            (rejected, 'Réponse rejected vide.')
        ):
            if not value:
                completeness = SemanticConfig.FAIL
                reasons.append(reason)

        preference = SemanticConfig.PASS

        if (
            chosen
            and rejected
            and key_text(chosen) == key_text(rejected)
        ):
            preference = SemanticConfig.FAIL
            reasons.append(
                'Les réponses chosen et rejected sont identiques.'
            )

        if self.flags & {
            'DPO_INVERSION',
            'DPO_PREFERENCE_CYCLE'
        }:
            preference = SemanticConfig.FAIL
            reasons.append(
                'Contradiction de préférence DPO.'
            )

        language, reason = self._language_quality()

        if reason:
            reasons.append(reason)

        autonomy, reason = self._autonomy_quality()

        if reason:
            reasons.append(reason)

        leakage, reason = self._target_leakage()

        if reason:
            reasons.append(reason)

        text_quality = (
            SemanticConfig.REVIEW
            if self.flags & SemanticConfig.TEXT_REVIEW_FLAGS
            else SemanticConfig.PASS
        )

        if 'DPO_LENGTH_BIAS' in self.flags:
            reasons.append(
                'Biais de longueur potentiel.'
            )

        if 'HIGH_RISK' in self.flags:
            reasons.append(
                'Contenu médical à haut risque.'
            )

        scores = {
            'completeness': completeness,
            'language': language,
            'autonomy': autonomy,
            'text_quality': text_quality,
            'target_leakage': leakage,
            'preference_consistency': preference
        }

        return self._result(scores, reasons)

    # Compare la langue détectée à celle attendue pour le record.
    # Une langue non détectée n'ajoute pas artificiellement un défaut.
    def _language_quality(
        self
    ) -> tuple[str, str | None]:
        """Vérifie la cohérence de langue."""
        expected = clean_text(
            self.record.get('language')
        ).lower()
        detected = clean_text(
            self.record.get('detected_language')
        ).lower()

        if not detected:
            return SemanticConfig.PASS, None

        if detected != expected:
            return (
                SemanticConfig.REVIEW,
                f'Langue détectée {detected!r} '
                f'différente de {expected!r}.'
            )

        return SemanticConfig.PASS, None

    # Vérifie que l'exemple reste compréhensible sans contexte externe manquant.
    # Les signaux produits par les audits précédents sont directement réutilisés.
    def _autonomy_quality(
        self
    ) -> tuple[str, str | None]:
        """Vérifie l'autonomie de l'exemple."""
        flags = (
            self.flags
            & SemanticConfig.AUTONOMY_FLAGS
        )

        if flags:
            return (
                SemanticConfig.REVIEW,
                'Autonomie à vérifier : '
                f'{", ".join(sorted(flags))}.'
            )

        return SemanticConfig.PASS, None

    # Recherche si une réponse cible est déjà révélée dans l'entrée source.
    # Les contrôles restent adaptés séparément aux formats DPO, QCM et ouverts.
    def _target_leakage(
        self
    ) -> tuple[str, str | None]:
        """Recherche une fuite de cible."""
        if self.record['family'] == 'dpo':
            return self._dpo_leakage()

        if self.record.get('choices'):
            return self._qcm_leakage()

        answer = key_text(
            self.record.get('answer')
        )
        source = self._source_text()

        if len(answer) >= 30 and answer in source:
            return (
                SemanticConfig.REVIEW,
                'Réponse ouverte déjà présente dans l’entrée.'
            )

        return SemanticConfig.PASS, None

    # Recherche une indication explicite de la bonne option dans le texte source.
    # La liste normale des propositions du QCM reste volontairement exclue.
    def _qcm_leakage(
        self
    ) -> tuple[str, str | None]:
        """Recherche une fuite de cible QCM."""
        source = self._source_text()

        for label in self.record.get('labels', []):
            pattern = (
                r'\b(?:correct answer|right answer|answer|'
                r'bonne reponse|reponse correcte)'
                rf'\s*(?:is|est|:)?\s*'
                rf'{re.escape(label.lower())}\b'
            )

            if re.search(pattern, source):
                return (
                    SemanticConfig.REVIEW,
                    f'Label correct {label!r} explicitement révélé.'
                )

        return SemanticConfig.PASS, None

    # Vérifie que le prompt DPO ne contient pas déjà une réponse candidate complète.
    # Seules les reprises suffisamment longues déclenchent une revue.
    def _dpo_leakage(
        self
    ) -> tuple[str, str | None]:
        """Recherche une fuite de cible DPO."""
        prompt = key_text(
            self.record.get('input')
            or self.record.get('question')
        )

        for field in ('chosen', 'rejected'):
            value = key_text(
                self.record.get(f'preference_{field}')
                or self.record.get(field)
            )

            if len(value) >= 30 and value in prompt:
                return (
                    SemanticConfig.REVIEW,
                    f'Réponse DPO {field!r} '
                    'déjà présente dans le prompt.'
                )

        return SemanticConfig.PASS, None

    # Regroupe le contexte et la question sans inclure les choix du QCM.
    # Cette représentation normalisée sert aux contrôles de fuite de cible.
    def _source_text(self) -> str:
        """Retourne le texte source."""
        parts = (
            clean_text(self.record.get('context')),
            clean_text(self.record.get('question'))
        )

        return key_text(
            ' '.join(filter(None, parts))
        )

    # Vérifie que chaque label QCM désigne un choix présent et non vide.
    # Toute cible absente ou invalide entraîne directement un FAIL.
    def _qcm_alignment(
        self
    ) -> tuple[str, str | None]:
        """Vérifie l'alignement QCM."""
        labels = self.record.get('labels', [])
        choices = self.record.get('choices', {})

        if not labels:
            return (
                SemanticConfig.FAIL,
                'QCM sans réponse correcte.'
            )

        missing = [
            label
            for label in labels
            if label not in choices
        ]

        if missing:
            return (
                SemanticConfig.FAIL,
                'Labels absents des choix : '
                f'{", ".join(missing)}.'
            )

        empty = [
            label
            for label in labels
            if not clean_text(choices[label])
        ]

        if empty:
            return (
                SemanticConfig.FAIL,
                'Choix corrects sans contenu : '
                f'{", ".join(empty)}.'
            )

        return SemanticConfig.PASS, None

    # Assemble verdict, score numérique et justification dans un format uniforme.
    # Le quality_score reste distinct de toute future confiance TRIAGE.
    def _result(
        self,
        scores: dict,
        reasons: list[str]
    ) -> dict:
        """Construit le résultat sémantique."""
        return {
            **scores,
            'confidence': 'DETERMINISTIC',
            'verdict': semantic_verdict(scores),
            'quality_score': semantic_quality_score(scores),
            'reason': ' '.join(reasons) or None
        }

    # Recalcule les défauts déterministes propres aux réponses ouvertes.
    # Les anciens flags concernés sont retirés avant chaque nouvelle évaluation.
    def _refresh_open_flags(self) -> None:
        """Recalcule les flags ouverts."""
        if self.record['family'] != 'open':
            return

        self.flags.difference_update(
            SemanticConfig.BLOCKING_OPEN_FLAGS
            | SemanticConfig.REVIEW_OPEN_FLAGS
        )

        answer = clean_text(
            self.record.get('answer')
        )

        if not answer:
            return

        normalized = key_text(answer)

        if normalized in OpenAnswerPatterns.UNINFORMATIVE:
            self.flags.add(
                'UNINFORMATIVE_ANSWER'
            )

        if any(
            normalized.startswith(key_text(prefix))
            for prefix in OpenAnswerPatterns.RESOURCE_PREFIXES
        ):
            self.flags.add(
                'RESOURCE_ONLY_ANSWER'
            )

        if any(
            normalized.startswith(key_text(prefix))
            for prefix in OpenAnswerPatterns.NAVIGATION_PREFIXES
        ):
            self.flags.add(
                'NAVIGATION_ONLY_ANSWER'
            )

        if len(answer.split()) <= 3:
            self.flags.add(
                'VERY_SHORT_OPEN_ANSWER'
            )

        if answer.endswith('?'):
            self.flags.add(
                'QUESTION_AS_ANSWER'
            )
