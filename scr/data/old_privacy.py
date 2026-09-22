import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

from scr.config import (
    ClinicalPrivacyPatterns,
    PrivacyConfig,
    PrivacyPatterns,
    TextConfig
)

from scr.data.cleaning import (
    assemble_input,
    clean_text,
    content_hash
)

from scr.utils.hashing import digest
from scr.utils.serialization import read_jsonl


class PrivacyProcessor:
    """Détecte et anonymise les informations personnelles."""

    # Initialise le processeur sur le record canonique courant.
    # Les opérations de confidentialité modifient directement ce record.
    def __init__(
        self,
        record: dict
    ) -> None:
        """Initialise le processeur."""
        self.record = record

    # Parcourt tous les champs textuels soumis aux contrôles de confidentialité.
    # Les choix QCM sont inclus car ils peuvent également contenir une PII.
    def iter_text_content(
        self
    ):
        """Parcourt les textes contrôlés."""
        for field in TextConfig.TEXT_FIELDS:
            value = self.record.get(field)

            if isinstance(value, str) and value:
                yield field, value

        for label, value in self.record.get(
            'choices',
            {}
        ).items():
            if isinstance(value, str) and value:
                yield f'choice_{label}', value

    # Agrège les textes soumis aux contrôles de confidentialité.
    # Cette vue sert aux validations nécessitant le contenu complet.
    def all_text(
        self
    ) -> str:
        """Agrège le contenu textuel."""
        return '\n'.join(
            value
            for _, value in self.iter_text_content()
        )

    # Recherche toutes les PII définies dans la configuration.
    # Chaque occurrence conserve son champ, sa catégorie et sa valeur.
    def pii_matches(
        self
    ) -> list[dict]:
        """Retourne les PII détectées."""
        return [
            {
                'field': field,
                'category': category,
                'value': match.group(0)
            }
            for field, value in self.iter_text_content()
            for category, pattern
            in PrivacyPatterns.PII_PATTERNS.items()
            for match in pattern.finditer(value)
        ]

    # Recherche uniquement les noms explicitement introduits dans les textes.
    # Les variantes détectées sont nettoyées puis dédupliquées.
    def explicit_names(
        self
    ) -> set[str]:
        """Retourne les noms explicites."""
        names = set()

        for _, value in self.iter_text_content():
            for match in PrivacyPatterns.EXPLICIT_NAME.finditer(value):
                name = clean_text(
                    match.group('name')
                ).strip(' ,.;:!?')

                if name:
                    names.add(name)

        return names

    # Regroupe les catégories de PII détectées dans le record.
    # Les noms explicitement introduits reçoivent leur catégorie dédiée.
    def pii_categories(
        self
    ) -> list[str]:
        """Retourne les catégories PII."""
        categories = {
            match['category']
            for match in self.pii_matches()
        }

        if self.explicit_names():
            categories.add('EXPLICIT_NAME')

        return sorted(categories)

    # Transforme une date de naissance en âge avant son anonymisation.
    # Sans date clinique de référence fiable, aucune conversion n'est effectuée.
    def replace_birth_date_with_age(
        self,
        reference_date: date | None = None
    ) -> list[dict]:
        """Remplace une naissance par un âge."""
        if reference_date is None:
            return []

        replacements = []
        patterns = (
            (
                ClinicalPrivacyPatterns.BIRTH_DATE_FR,
                'fr'
            ),
            (
                ClinicalPrivacyPatterns.BIRTH_DATE_EN,
                'en'
            )
        )

        for field, value in list(
            self.iter_text_content()
        ):
            updated_value = value

            for pattern, language in patterns:
                matches = list(
                    pattern.finditer(updated_value)
                )

                for match in reversed(matches):
                    birth_date = match.group('date')

                    try:
                        age = self._age_from_birth_date(
                            birth_date,
                            reference_date
                        )
                    except ValueError:
                        continue

                    if field in TextConfig.CLINICAL_INPUT_FIELDS:
                        self.record['patient_age_years'] = age

                    replacement = (
                        f'âgé de {age} ans'
                        if language == 'fr'
                        else f'{age} years old'
                    )

                    start, end = match.span()

                    updated_value = (
                        updated_value[:start]
                        + replacement
                        + updated_value[end:]
                    )

                    replacements.append({
                        'field': field,
                        'category': 'BIRTH_DATE',
                        'value_hash': digest(birth_date),
                        'derived_age': age,
                        'reference_date': reference_date.isoformat(),
                        'replacement': replacement
                    })

            if updated_value == value:
                continue

            target = self.text_container(field)

            if target is not None:
                container, key = target
                container[key] = updated_value

        self.record.setdefault(
            'automatic_redactions',
            []
        ).extend(replacements)

        return replacements

    # Remplace uniquement les catégories configurées pour l'anonymisation automatique.
    # Chaque remplacement conserve une empreinte de la valeur supprimée.
    def apply_automatic_redactions(
        self
    ) -> list[dict]:
        """Anonymise les PII automatiques."""
        redactions = []

        for match in list(self.pii_matches()):
            replacement = (
                PrivacyConfig
                .AUTOMATIC_REDACTION_REPLACEMENTS
                .get(match['category'])
            )

            if replacement is None:
                continue

            target = self.text_container(
                match['field']
            )

            if target is None:
                continue

            container, key = target
            current_text = container.get(key)
            value = match['value']

            if (
                not isinstance(current_text, str)
                or value not in current_text
            ):
                continue

            container[key] = current_text.replace(
                value,
                replacement
            )

            redactions.append({
                'field': match['field'],
                'category': match['category'],
                'value_hash': digest(value),
                'replacement': replacement
            })

        self.record.setdefault(
            'automatic_redactions',
            []
        ).extend(redactions)

        return redactions

    # Anonymise les noms explicitement introduits dans les champs textuels.
    # Les noms les plus longs sont remplacés en premier pour éviter les collisions.
    def redact_explicit_names(
        self
    ) -> list[dict]:
        """Anonymise les noms explicites."""
        names = self.explicit_names()

        if not names:
            return []

        replacement = (
            PrivacyConfig
            .AUTOMATIC_REDACTION_REPLACEMENTS[
                'EXPLICIT_NAME'
            ]
        )
        redactions = []

        for name in sorted(
            names,
            key=len,
            reverse=True
        ):
            pattern = re.compile(
                rf'(?<!\w){re.escape(name)}(?!\w)',
                re.IGNORECASE
            )

            for field, _ in list(
                self.iter_text_content()
            ):
                target = self.text_container(field)

                if target is None:
                    continue

                container, key = target
                current_text = container.get(key)

                if (
                    not isinstance(current_text, str)
                    or not pattern.search(current_text)
                ):
                    continue

                container[key] = pattern.sub(
                    replacement,
                    current_text
                )

                redactions.append({
                    'field': field,
                    'category': 'EXPLICIT_NAME',
                    'value_hash': digest(name),
                    'replacement': replacement
                })

        self.record.setdefault(
            'automatic_redactions',
            []
        ).extend(redactions)

        return redactions

    # Recherche les PII automatiques encore présentes après anonymisation.
    # Les noms explicites résiduels sont ajoutés au même contrôle.
    def remaining_automatic_pii(
        self
    ) -> list[dict]:
        """Retourne les PII restantes."""
        remaining = [
            match
            for match in self.pii_matches()
            if match['category']
            in PrivacyConfig.AUTOMATIC_PII_CATEGORIES
        ]

        remaining.extend(
            {
                'field': None,
                'category': 'EXPLICIT_NAME',
                'value': name
            }
            for name in self.explicit_names()
        )

        return remaining

    # Retourne uniquement les noms explicites encore présents dans le record.
    # Cette vue est utilisée par les validations finales de confidentialité.
    def remaining_explicit_names(
        self
    ) -> list[str]:
        """Retourne les noms restants."""
        return sorted(
            self.explicit_names()
        )

    # Retrouve le conteneur modifiable correspondant à un champ textuel.
    # Les champs standards et choix QCM sont manipulés uniformément.
    def text_container(
        self,
        field: str
    ) -> tuple[dict, str] | None:
        """Retourne le conteneur du champ."""
        if field.startswith('choice_'):
            label = field.removeprefix(
                'choice_'
            )
            choices = self.record.get(
                'choices',
                {}
            )

            if label in choices:
                return choices, label

            return None

        if field in TextConfig.TEXT_FIELDS:
            return self.record, field

        return None

    # Vérifie qu'un âge dérivé appartient aux limites techniques configurées.
    # Cette validation sert uniquement à écarter des valeurs impossibles.
    def _valid_age(
        self,
        value: int
    ) -> bool:
        """Vérifie la plausibilité d'un âge."""
        limits = (
            ClinicalPrivacyPatterns
            .PATIENT_AGE_LIMITS
        )

        return (
            limits['min']
            <= value
            <= limits['max']
        )

    # Calcule l'âge à partir d'une naissance et d'une référence clinique connue.
    # La date actuelle n'est jamais utilisée implicitement dans ce calcul.
    def _age_from_birth_date(
        self,
        value: str,
        reference_date: date
    ) -> int:
        """Calcule l'âge à une date donnée."""
        formats = (
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%d.%m.%Y',
            '%Y/%m/%d',
            '%Y-%m-%d',
            '%Y.%m.%d'
        )

        birth_date = next(
            (
                parsed
                for date_format in formats
                if (
                    parsed := self._parse_date(
                        value,
                        date_format
                    )
                ) is not None
            ),
            None
        )

        if birth_date is None:
            raise ValueError(
                f'Date de naissance non interprétable : '
                f'{value}'
            )

        if birth_date > reference_date:
            raise ValueError(
                f'Date de naissance postérieure '
                f'à la référence : {value}'
            )

        age = (
            reference_date.year
            - birth_date.year
        )

        if (
            reference_date.month,
            reference_date.day
        ) < (
            birth_date.month,
            birth_date.day
        ):
            age -= 1

        if not self._valid_age(age):
            raise ValueError(
                f'Âge dérivé hors limites : {age}'
            )

        return age

    # Essaie d'interpréter une chaîne selon un format de date déterminé.
    # Un format incompatible renvoie None sans interrompre le pipeline.
    def _parse_date(
        self,
        value: str,
        date_format: str
    ) -> date | None:
        """Tente d'interpréter une date."""
        try:
            return datetime.strptime(
                value,
                date_format
            ).date()
        except ValueError:
            return None


class PrivacyReviewConfig:
    """Configuration des décisions humaines RGPD."""

    DECISIONS = {'approve', 'redact', 'exclude'}
    REQUIRED_FIELDS = {
        'id',
        'decision',
        'reason',
        'reviewer',
        'reviewed_at',
        'source_content_hash',
        'approved_content_hash'
    }


class PrivacyReviewManager:
    """Charge et valide les décisions humaines RGPD."""

    # Initialise le gestionnaire avec la file et le fichier de décisions.
    # Les décisions validées sont ensuite indexées par identifiant.
    def __init__(
        self,
        review_records: list[dict],
        decisions_file: Path
    ) -> None:
        """Initialise le gestionnaire."""
        self.review_records = review_records
        self.decisions_file = decisions_file
        self.decisions: list[dict] = []
        self.decision_by_id: dict[str, dict] = {}

    # Charge les décisions puis vérifie leur cohérence et leur traçabilité.
    # Une liste vide reste valide lorsqu'aucune revue n'a encore été saisie.
    def load(self) -> None:
        """Charge et valide les décisions."""
        self.decisions = read_jsonl(
            self.decisions_file
        )
        self._validate_decisions()

        self.decision_by_id = {
            decision['id']: decision
            for decision in self.decisions
        }

    # Retourne le nombre de records encore sans décision humaine.
    # Le calcul compare les identifiants attendus aux décisions disponibles.
    def remaining_count(self) -> int:
        """Retourne les décisions restantes."""
        review_ids = {
            record['id']
            for record in self.review_records
        }

        return len(
            review_ids - self.decision_by_id.keys()
        )

    # Vérifie unicité, schéma, action, portée et métadonnées des décisions.
    # Toute incohérence bloque l'application de la revue humaine.
    def _validate_decisions(self) -> None:
        """Valide les décisions RGPD."""
        ids = [
            decision.get('id')
            for decision in self.decisions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                'Plusieurs décisions RGPD existent pour un même ID.'
            )

        review_ids = {
            record['id']
            for record in self.review_records
        }

        for decision in self.decisions:
            record_id = decision.get(
                'id',
                '<unknown>'
            )
            missing = (
                PrivacyReviewConfig.REQUIRED_FIELDS
                - decision.keys()
            )

            if missing:
                raise ValueError(
                    f'Décision RGPD incomplète pour {record_id!r} : '
                    f'{sorted(missing)}.'
                )

            if record_id not in review_ids:
                raise ValueError(
                    f'Décision RGPD hors file de revue : {record_id!r}.'
                )

            action = decision['decision']

            if action not in PrivacyReviewConfig.DECISIONS:
                raise ValueError(
                    f'Action RGPD invalide pour {record_id!r} : '
                    f'{action!r}.'
                )

            required_values = (
                'reviewer',
                'reviewed_at',
                'reason',
                'source_content_hash'
            )

            if any(
                not str(
                    decision.get(field, '')
                ).strip()
                for field in required_values
            ):
                raise ValueError(
                    f'Métadonnées RGPD incomplètes pour {record_id!r}.'
                )

            if (
                action in {'approve', 'redact'}
                and not str(
                    decision.get(
                        'approved_content_hash',
                        ''
                    )
                ).strip()
            ):
                raise ValueError(
                    f'Hash approuvé manquant pour {record_id!r}.'
                )



class PrivacyDecisionProcessor:
    """Applique une décision RGPD validée à un record."""

    # Initialise le processeur avec le record canonique courant.
    # Les modifications restent strictement limitées au périmètre RGPD.
    def __init__(
        self,
        record: dict
    ) -> None:
        """Initialise le processeur."""
        self.record = record

    # Applique la décision disponible ou place le record en attente.
    # Toute décision humaine est validée avant modification du contenu.
    def apply(
        self,
        decision: dict | None
    ) -> None:
        """Applique la décision RGPD."""
        if decision is None:
            self.record['privacy_status'] = 'PENDING'
            return

        self._validate_source_hash(
            decision
        )

        action = decision['decision']

        if action == 'exclude':
            self._exclude(
                decision
            )
        elif action == 'approve':
            self._approve(
                decision
            )
        elif action == 'redact':
            self._redact(
                decision
            )
        else:
            raise ValueError(
                f'Décision RGPD inconnue pour '
                f'{self.record["id"]!r} : {action!r}.'
            )

    # Vérifie que la décision cible exactement le contenu nettoyé courant.
    # Une décision produite sur une ancienne version du record est refusée.
    def _validate_source_hash(
        self,
        decision: dict
    ) -> None:
        """Valide le hash source."""
        if (
            decision.get('source_content_hash')
            != self.record['clean_content_hash']
        ):
            raise ValueError(
                f'Décision RGPD périmée pour '
                f'{self.record["id"]!r}.'
            )

    # Exclut définitivement le record pour raison de confidentialité.
    # Les autres contrôles et flags du pipeline restent inchangés.
    def _exclude(
        self,
        decision: dict
    ) -> None:
        """Exclut le record."""
        self.record['flags'].add(
            'PRIVACY_EXCLUDED'
        )
        self.record['privacy_status'] = 'EXCLUDED'
        self.record['privacy_action'] = 'exclude'

        self._store_review_metadata(
            decision
        )

    # Approuve le contenu courant après vérification de son empreinte.
    # Seuls les signaux de confidentialité résolus sont retirés.
    def _approve(
        self,
        decision: dict
    ) -> None:
        """Approuve le contenu courant."""
        if (
            decision.get('approved_content_hash')
            != content_hash(self.record)
        ):
            raise ValueError(
                f'Hash RGPD approuvé incorrect pour '
                f'{self.record["id"]!r}.'
            )

        self._clear_privacy_flags(
            self.record
        )

        self.record['privacy_status'] = 'APPROVED'
        self.record['privacy_action'] = 'approve'

        self._store_review_metadata(
            decision
        )

    # Applique les remplacements humains sur une copie indépendante du record.
    # Le record courant n'est remplacé qu'après validation complète du résultat.
    def _redact(
        self,
        decision: dict
    ) -> None:
        """Applique les remplacements RGPD."""
        replacements = decision.get(
            'replacements'
        )

        if (
            not isinstance(replacements, list)
            or not replacements
        ):
            raise ValueError(
                f'Replacements RGPD manquants pour '
                f'{self.record["id"]!r}.'
            )

        working_record = deepcopy(
            self.record
        )

        privacy = PrivacyProcessor(
            working_record
        )

        applied = [
            self._apply_replacement(
                privacy,
                replacement
            )
            for replacement in replacements
        ]

        self._rebuild_content(
            working_record
        )

        self._revalidate_redacted_record(
            working_record,
            privacy,
            decision
        )

        self._clear_privacy_flags(
            working_record
        )

        working_record.update({
            'privacy_status': 'APPROVED',
            'privacy_action': 'redact',
            'privacy_redactions': applied
        })

        self._store_review_metadata(
            decision,
            working_record
        )

        working_record.pop(
            'payload',
            None
        )
        working_record.pop(
            'tokens',
            None
        )

        self.record.clear()
        self.record.update(
            working_record
        )

    # Remplace un texte validé dans tous les champs textuels autorisés.
    # L'absence du texte source bloque l'opération pour éviter un faux succès.
    def _apply_replacement(
        self,
        privacy: PrivacyProcessor,
        replacement: dict
    ) -> dict:
        """Applique un remplacement RGPD."""
        if not isinstance(
            replacement,
            dict
        ):
            raise ValueError(
                f'Replacement RGPD invalide pour '
                f'{self.record["id"]!r}.'
            )

        source = clean_text(
            replacement.get('source')
        )

        target = (
            clean_text(
                replacement.get('target')
            )
            or '[REDACTED]'
        )

        if not source:
            raise ValueError(
                f'Source de redaction vide pour '
                f'{self.record["id"]!r}.'
            )

        applied_fields = []

        for field, value in privacy.iter_text_content():
            if source not in value:
                continue

            destination = privacy.text_container(
                field
            )

            if destination is None:
                continue

            container, key = destination

            container[key] = value.replace(
                source,
                target
            )

            applied_fields.append(
                field
            )

        if not applied_fields:
            raise ValueError(
                f'Texte à masquer introuvable pour '
                f'{self.record["id"]!r}.'
            )

        return {
            'source_hash': digest(source),
            'replacement': target,
            'fields': sorted(
                set(applied_fields)
            )
        }

    # Reconstruit uniquement le texte canonique après les remplacements RGPD.
    # L'extraction clinique détaillée sera entièrement rejouée plus tard.
    def _rebuild_content(
        self,
        working_record: dict
    ) -> None:
        """Reconstruit le contenu nettoyé."""
        working_record['input'] = assemble_input(
            working_record
        )

        self._clear_clinical_enrichment(
            working_record
        )

    # Supprime les enrichissements cliniques devenus potentiellement obsolètes.
    # Ils seront recalculés plus tard par l'extracteur clinique unique.
    def _clear_clinical_enrichment(
        self,
        record: dict
    ) -> None:
        """Supprime les données cliniques dérivées."""
        for field in (
            'patient_age_days',
            'patient_age_months',
            'patient_age_years',
            'patient_sex',
            'duration',
            'evolution',
            'critical_sign_states',
            'triage_clinical_enrichment'
        ):
            record[field] = None

        record['vital_signs'] = {}
        record['symptoms'] = []
        record['medical_history'] = []
        record['critical_signs'] = []

    # Recalcule les PII puis vérifie l'empreinte du contenu humainement approuvé.
    # Une PII automatique restante ou un hash différent bloque la redaction.
    def _revalidate_redacted_record(
        self,
        working_record: dict,
        privacy: PrivacyProcessor,
        decision: dict
    ) -> None:
        """Revalide le contenu anonymisé."""
        working_record['pii_matches'] = (
            privacy.pii_matches()
        )

        working_record['pii_categories'] = (
            privacy.pii_categories()
        )

        if privacy.remaining_automatic_pii():
            raise ValueError(
                f'PII automatique restante après redaction pour '
                f'{self.record["id"]!r}.'
            )

        from scr.data.cleaning import QualityValidator

        QualityValidator(
            working_record
        ).validate()

        final_hash = content_hash(
            working_record
        )

        if (
            decision.get('approved_content_hash')
            != final_hash
        ):
            raise ValueError(
                f'Hash du contenu anonymisé incorrect pour '
                f'{self.record["id"]!r}.'
            )

        working_record['clean_content_hash'] = (
            final_hash
        )

    # Retire uniquement les signaux de confidentialité résolus par la revue.
    # Tous les autres flags qualité et médicaux restent strictement inchangés.
    def _clear_privacy_flags(
        self,
        record: dict
    ) -> None:
        """Retire les flags RGPD résolus."""
        record['flags'].difference_update({
            'PII_CANDIDATE',
            'IDENTITY_INTRODUCTION'
        })

    # Enregistre les informations nécessaires à la traçabilité humaine.
    # Les hashes relient la décision, sa source et le contenu approuvé.
    def _store_review_metadata(
        self,
        decision: dict,
        target: dict | None = None
    ) -> None:
        """Enregistre les métadonnées RGPD."""
        target = (
            self.record
            if target is None
            else target
        )

        target.update({
            'privacy_reason': decision['reason'],
            'privacy_reviewer': decision['reviewer'],
            'privacy_reviewed_at': decision['reviewed_at'],
            'privacy_source_content_hash': (
                decision['source_content_hash']
            ),
            'privacy_approved_content_hash': (
                decision.get(
                    'approved_content_hash'
                )
            )
        })

def privacy_review_categories(record: dict) -> set[str]:
    """Retourne les catégories PII à revoir."""
    return (
        set(record.get("pii_categories", []))
        & PrivacyConfig.REVIEW_PII_CATEGORIES
    )

class PrivacyReviewStatus:
    """Valide l'état global de la revue RGPD."""

    VALID_STATUSES = {"PENDING", "EXCLUDED", "APPROVED"}

    def __init__(
        self,
        review_records: list[dict],
        records: list[dict]
    ) -> None:
        """Initialise le contrôleur."""
        self.review_records = review_records
        self.records = records
        self.pending: list[str] = []
        self.excluded: list[str] = []
        self.approved: list[str] = []
        self.automatic_pii_remaining: list[str] = []

    def validate(self) -> None:
        """Valide l'état global RGPD."""
        self._collect_statuses()
        self._validate_exclusions()
        self._check_automatic_pii()

    def passed(self) -> bool:
        """Indique si la revue RGPD est terminée."""
        return (
            not self.pending
            and not self.automatic_pii_remaining
        )

    def _collect_statuses(self) -> None:
        """Classe les records selon leur statut."""
        self.pending = []
        self.excluded = []
        self.approved = []

        buckets = {
            "PENDING": self.pending,
            "EXCLUDED": self.excluded,
            "APPROVED": self.approved
        }

        for record in self.review_records:
            status = record.get("privacy_status")

            if status not in self.VALID_STATUSES:
                raise ValueError(
                    f"Statut RGPD invalide pour {record['id']!r} : "
                    f"{status!r}."
                )

            buckets[status].append(record["id"])

    def _validate_exclusions(self) -> None:
        """Valide les exclusions RGPD."""
        invalid = [
            record["id"]
            for record in self.review_records
            if (
                record.get("privacy_status") == "EXCLUDED"
                and "PRIVACY_EXCLUDED"
                not in record.get("flags", set())
            )
        ]

        if invalid:
            raise ValueError(
                f"{len(invalid):,} exclusion(s) RGPD sans "
                "flag PRIVACY_EXCLUDED."
            )

    def _check_automatic_pii(self) -> None:
        """Recherche les PII automatiques restantes."""
        self.automatic_pii_remaining = [
            record["id"]
            for record in self.records
            if PrivacyProcessor(
                record
            ).remaining_automatic_pii()
        ]