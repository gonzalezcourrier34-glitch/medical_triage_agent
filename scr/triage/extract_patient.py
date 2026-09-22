# scr/triage/extract_patient.py

import re

from scr.data.cleaning import clean_text, key_text
from scr.triage.knowledge_matching import match_french_motif_candidates
from scr.triage.patient_patterns import (
    AGE_RULES,
    BLOOD_GLUCOSE_PATTERNS,
    BLOOD_PRESSURE_CMHG_PATTERNS,
    BLOOD_PRESSURE_MMHG_PATTERNS,
    CLINICAL_ITEM_SPLIT,
    CMHG_TO_MMHG,
    CONDITION_MENTION_EXCLUSION_PATTERNS,
    CONDITION_MENTION_PATTERNS,
    CRITICAL_SIGN_NEGATION_PATTERNS,
    CRITICAL_SIGN_PATTERNS,
    DIASTOLIC_BP_PATTERNS,
    DURATION_PATTERNS,
    EVOLUTION_PATTERNS,
    FRENCH_CONCEPT_NEGATION_PATTERNS,
    FRENCH_CONCEPT_PATTERNS,
    FRENCH_NUMBER_TENS,
    FRENCH_NUMBER_UNITS,
    FRENCH_WORD_AGE_PATTERNS,
    KETONE_PATTERNS,
    MEDICAL_HISTORY_NEGATION_PATTERNS,
    MEDICAL_HISTORY_NARRATIVE_PATTERNS,
    MEDICAL_HISTORY_PATTERNS,
    PAIN_SCORE_PATTERNS,
    PATIENT_SEX_PATTERNS,
    PATIENT_TEXT_FIELDS,
    PLAUSIBILITY_LIMITS,
    SYMPTOM_EXCLUSION_PATTERNS,
    SYMPTOM_NARRATIVE_PATTERNS,
    SYMPTOM_SECTION_PATTERNS,
    SYSTOLIC_BP_PATTERNS,
    TEMPERATURE_PATTERNS,
    VITAL_RULES
)


# ## Extracteur patient

class PatientExtractor:
    """Extrait les informations décrivant le patient."""

    def __init__(self, motif_lexicon: dict) -> None:
        self.motif_lexicon = motif_lexicon

    # ## Extraction

    def extract(self, record: dict) -> dict:
        """Extrait les informations patient."""
        patient = {}
        extracted = []
        text = self._source_text(record)

        self._extract_age(patient, text, extracted)
        self._extract_sex(patient, text, extracted)
        self._extract_vitals(patient, text, extracted)
        self._extract_textual_fields(patient, text, extracted)

        critical_conflicts = self._extract_critical_signs(
            patient, text, extracted
        )
        concept_conflicts = self._extract_french_concepts(
            patient, text, extracted
        )

        conditions = self._extract_condition_mentions(text)

        if conditions:
            patient["condition_mentions"] = conditions
            extracted.append("condition_mentions")

        patient["french_motif_candidates"] = match_french_motif_candidates(
            text,
            record.get("language"),
            self.motif_lexicon
        )

        patient["extraction"] = {
            "method": "deterministic_patient_extraction_v3",
            "extracted_fields": sorted(set(extracted)),
            "critical_sign_conflicts": critical_conflicts,
            "french_concept_conflicts": concept_conflicts,
            "source_clean_content_hash": record.get("clean_content_hash")
        }

        return patient

    # ## Texte source

    def _source_text(self, record: dict) -> str:
        """Construit le texte patient."""
        return " ".join(
            value
            for field in PATIENT_TEXT_FIELDS
            if (value := clean_text(record.get(field)))
        )

    # ## Âge

    def _extract_age(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> None:
        """Extrait l'âge."""
        for field, patterns, limit_key in AGE_RULES:
            value = self._extract_number(text, patterns)

            if value is None or not self._is_plausible(limit_key, value):
                continue

            patient[field] = int(value)
            extracted.append(field)
            return

        value = self._extract_french_word_age(text)

        if value is not None:
            patient["age_years"] = value
            extracted.append("age_years")

    def _extract_french_word_age(self, text: str) -> int | None:
        """Extrait un âge écrit en lettres."""
        for pattern in FRENCH_WORD_AGE_PATTERNS:
            match = pattern.search(text)

            if not match:
                continue

            value = self._parse_french_number(match.group(1))

            if value is not None and self._is_plausible("age_years", value):
                return value

        return None

    def _parse_french_number(self, value: str) -> int | None:
        """Convertit un nombre français."""
        value = clean_text(value).casefold()

        if value in FRENCH_NUMBER_UNITS:
            return FRENCH_NUMBER_UNITS[value]

        if value in FRENCH_NUMBER_TENS:
            return FRENCH_NUMBER_TENS[value]

        tokens = [
            token
            for token in value.replace("-", " ").split()
            if token != "et"
        ]

        if tokens == ["cent"]:
            return 100

        if tokens[:2] in (["quatre", "vingt"], ["quatre", "vingts"]):
            base, rest = 80, tokens[2:]
        elif tokens[:2] == ["soixante", "dix"]:
            base, rest = 70, tokens[2:]
        elif tokens and tokens[0] in FRENCH_NUMBER_TENS:
            base = FRENCH_NUMBER_TENS[tokens[0]]
            rest = tokens[1:]
        else:
            return None

        if not rest:
            return base

        if len(rest) != 1:
            return None

        unit = FRENCH_NUMBER_UNITS.get(rest[0])

        if unit is None:
            return None

        result = base + unit
        return result if result <= 120 else None

    # ## Sexe

    def _extract_sex(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> None:
        """Extrait le sexe."""
        detected = {
            sex
            for sex, patterns in PATIENT_SEX_PATTERNS.items()
            if self._has_pattern(text, patterns)
        }

        if len(detected) == 1:
            patient["sex"] = detected.pop()
            extracted.append("sex")

    # ## Constantes

    def _extract_vitals(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> None:
        """Extrait les constantes vitales."""
        vitals = {}

        self._add_vital(
            vitals,
            "temperature",
            self._extract_temperature(text),
            "temperature",
            extracted
        )

        pressure = self._extract_blood_pressure(text)

        if pressure:
            systolic, diastolic = pressure

            self._add_vital(
                vitals, "systolic_bp", systolic, "systolic_bp", extracted
            )
            self._add_vital(
                vitals, "diastolic_bp", diastolic, "diastolic_bp", extracted
            )
        else:
            self._add_vital(
                vitals,
                "systolic_bp",
                self._extract_number(text, SYSTOLIC_BP_PATTERNS),
                "systolic_bp",
                extracted
            )
            self._add_vital(
                vitals,
                "diastolic_bp",
                self._extract_number(text, DIASTOLIC_BP_PATTERNS),
                "diastolic_bp",
                extracted
            )

        for field, (patterns, limit_key) in VITAL_RULES.items():
            self._add_vital(
                vitals,
                field,
                self._extract_number(text, patterns),
                limit_key,
                extracted
            )

        for field, patterns in (
            ("blood_glucose_mmol_l", BLOOD_GLUCOSE_PATTERNS),
            ("ketones_mmol_l", KETONE_PATTERNS),
            ("pain_score", PAIN_SCORE_PATTERNS)
        ):
            self._add_vital(
                vitals,
                field,
                self._extract_number(text, patterns),
                field,
                extracted
            )

        if vitals:
            patient["vital_signs"] = vitals

    def _add_vital(
        self,
        vitals: dict,
        field: str,
        value: float | None,
        limit_key: str,
        extracted: list[str]
    ) -> None:
        """Ajoute une constante plausible."""
        if value is None or not self._is_plausible(limit_key, value):
            return

        vitals[field] = value
        extracted.append(f"vital_signs.{field}")

    # ## Texte clinique

    def _extract_textual_fields(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> None:
        """Extrait symptômes, antécédents, durée et évolution."""
        symptoms = self._extract_sections(text, SYMPTOM_SECTION_PATTERNS)
        symptoms += self._extract_narrative_values(
            text,
            SYMPTOM_NARRATIVE_PATTERNS,
            SYMPTOM_EXCLUSION_PATTERNS
        )
        symptoms = self._unique_texts(symptoms)

        if symptoms:
            patient["symptoms"] = symptoms
            extracted.append("symptoms")

        history = self._extract_history(text)

        if history:
            patient["medical_history"] = history
            extracted.append("medical_history")

        duration = self._extract_unique_match(text, DURATION_PATTERNS)

        if duration:
            patient["duration"] = duration
            extracted.append("duration")

            minutes = self._duration_to_minutes(duration)

            if minutes is not None:
                patient["duration_minutes"] = minutes
                extracted.append("duration_minutes")

        evolution = self._extract_category(text, EVOLUTION_PATTERNS)

        if evolution:
            patient["evolution"] = evolution
            extracted.append("evolution")

    # ## Antécédents

    def _extract_history(self, text: str) -> list[str]:
        """Extrait les antécédents."""
        values = []

        for pattern in MEDICAL_HISTORY_PATTERNS:
            for match in pattern.finditer(text):
                values += self._split_clinical_items(match.group(1))

        for pattern in MEDICAL_HISTORY_NARRATIVE_PATTERNS:
            for match in pattern.finditer(text):
                local = text[max(0, match.start() - 50):match.end()]

                if self._has_pattern(
                    local,
                    MEDICAL_HISTORY_NEGATION_PATTERNS
                ):
                    continue

                value = clean_text(match.group(1))

                if self._valid_text_value(value):
                    values.append(value)

        return self._unique_texts(values)

    # ## Pathologies

    def _extract_condition_mentions(self, text: str) -> list[str]:
        """Extrait les pathologies patient."""
        return self._extract_narrative_values(
            text,
            CONDITION_MENTION_PATTERNS,
            CONDITION_MENTION_EXCLUSION_PATTERNS
        )

    # ## Signes critiques

    def _extract_critical_signs(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> list[str]:
        """Extrait les signes critiques."""
        states, conflicts = self._extract_tri_states(
            text,
            CRITICAL_SIGN_PATTERNS,
            CRITICAL_SIGN_NEGATION_PATTERNS
        )

        patient["critical_sign_states"] = states
        patient["critical_signs"] = sorted(
            sign
            for sign, state in states.items()
            if state == "PRESENT"
        )

        if self._has_known_state(states):
            extracted.append("critical_sign_states")

        if patient["critical_signs"]:
            extracted.append("critical_signs")

        return conflicts

    # ## Concepts FRENCH

    def _extract_french_concepts(
        self,
        patient: dict,
        text: str,
        extracted: list[str]
    ) -> list[str]:
        """Extrait les concepts FRENCH."""
        states, conflicts = self._extract_tri_states(
            text,
            FRENCH_CONCEPT_PATTERNS,
            FRENCH_CONCEPT_NEGATION_PATTERNS
        )

        patient["french_concept_states"] = states

        if self._has_known_state(states):
            extracted.append("french_concept_states")

        return conflicts

    # ## Tri-state

    def _extract_tri_states(
        self,
        text: str,
        positive: dict,
        negative: dict
    ) -> tuple[dict[str, str], list[str]]:
        """Extrait PRESENT, ABSENT ou UNKNOWN."""
        states = {}
        conflicts = []

        for concept, patterns in positive.items():
            state, conflict = self._tri_state(
                text,
                patterns,
                negative.get(concept, ())
            )

            states[concept] = state

            if conflict:
                conflicts.append(concept)

        return states, sorted(conflicts)

    def _tri_state(
        self,
        text: str,
        positive: tuple[re.Pattern, ...],
        negative: tuple[re.Pattern, ...]
    ) -> tuple[str, bool]:
        """Détermine l'état d'un concept."""
        positives = [
            match
            for pattern in positive
            for match in pattern.finditer(text)
        ]

        negatives = [
            match
            for pattern in negative
            for match in pattern.finditer(text)
        ]

        positives = [
            match
            for match in positives
            if not any(
                neg.start() <= match.start()
                and match.end() <= neg.end()
                for neg in negatives
            )
        ]

        if positives and negatives:
            return "UNKNOWN", True

        if positives:
            return "PRESENT", False

        if negatives:
            return "ABSENT", False

        return "UNKNOWN", False

    def _has_known_state(self, states: dict[str, str]) -> bool:
        """Teste la présence d'un état connu."""
        return any(
            state != "UNKNOWN"
            for state in states.values()
        )

    # ## Valeurs numériques

    def _extract_number(
        self,
        text: str,
        patterns: re.Pattern | tuple[re.Pattern, ...]
    ) -> float | None:
        """Extrait une valeur numérique unique."""
        patterns = (
            (patterns,)
            if isinstance(patterns, re.Pattern)
            else patterns
        )

        values = {
            float(
                match.group(1)
                .replace(" ", "")
                .replace(",", ".")
            )
            for pattern in patterns
            for match in pattern.finditer(text)
        }

        return values.pop() if len(values) == 1 else None

    def _extract_temperature(self, text: str) -> float | None:
        """Extrait une température Celsius."""
        values = set()

        for pattern in TEMPERATURE_PATTERNS:
            for match in pattern.finditer(text):
                value = float(
                    match.group(1)
                    .replace(" ", "")
                    .replace(",", ".")
                )

                unit = (
                    match.group(2).casefold()
                    if match.lastindex
                    and match.lastindex >= 2
                    and match.group(2)
                    else "c"
                )

                if unit in {"f", "fahrenheit"}:
                    value = (value - 32) * 5 / 9

                values.add(round(value, 1))

        return values.pop() if len(values) == 1 else None

    def _extract_blood_pressure(
        self,
        text: str
    ) -> tuple[float, float] | None:
        """Extrait une pression artérielle."""
        sources = (
            (BLOOD_PRESSURE_CMHG_PATTERNS, CMHG_TO_MMHG),
            (BLOOD_PRESSURE_MMHG_PATTERNS, 1)
        )

        for patterns, factor in sources:
            for pattern in patterns:
                match = pattern.search(text)

                if not match:
                    continue

                systolic = float(
                    match.group(1).replace(",", ".")
                ) * factor
                diastolic = float(
                    match.group(2).replace(",", ".")
                ) * factor

                if (
                    self._is_plausible("systolic_bp", systolic)
                    and self._is_plausible("diastolic_bp", diastolic)
                    and systolic > diastolic
                ):
                    return systolic, diastolic

        return None

    def _is_plausible(self, field: str, value: float) -> bool:
        """Vérifie la plausibilité technique."""
        minimum, maximum = PLAUSIBILITY_LIMITS[field]
        return minimum <= value <= maximum

    # ## Durée

    def _duration_to_minutes(self, value: str) -> float | None:
        """Convertit une durée en minutes."""
        match = re.fullmatch(
            r"\s*(\d+(?:[.,]\d+)?)\s*"
            r"(minutes?|mins?|heures?|hours?|hrs?|jours?|days?|"
            r"semaines?|weeks?|mois|months?|ans?|years?)\s*",
            clean_text(value),
            re.IGNORECASE
        )

        if not match:
            return None

        amount = float(
            match.group(1).replace(",", ".")
        )
        unit = match.group(2).casefold()

        factors = (
            (("min",), 1),
            (("heur", "hour", "hr"), 60),
            (("jour", "day"), 1440),
            (("sem", "week"), 10080),
            (("mois", "month"), 43830),
            (("an", "year"), 525960)
        )

        factor = next(
            (
                factor
                for prefixes, factor in factors
                if unit.startswith(prefixes)
            ),
            None
        )

        return (
            round(amount * factor, 2)
            if factor is not None
            else None
        )

    # ## Utilitaires

    def _extract_sections(
        self,
        text: str,
        patterns: tuple[re.Pattern, ...]
    ) -> list[str]:
        """Extrait une section clinique."""
        values = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                values += self._split_clinical_items(
                    match.group(1)
                )

        return self._unique_texts(values)

    def _extract_narrative_values(
        self,
        text: str,
        patterns: tuple[re.Pattern, ...],
        exclusions: tuple[re.Pattern, ...] = ()
    ) -> list[str]:
        """Extrait des fragments narratifs."""
        values = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                value = clean_text(match.group(1))

                if (
                    self._valid_text_value(value)
                    and not self._has_pattern(value, exclusions)
                ):
                    values.append(value)

        return self._unique_texts(values)

    def _extract_unique_match(
        self,
        text: str,
        patterns: tuple[re.Pattern, ...]
    ) -> str | None:
        """Extrait une valeur non ambiguë."""
        values = self._unique_texts([
            clean_text(match.group(1))
            for pattern in patterns
            for match in pattern.finditer(text)
        ])

        return values[0] if len(values) == 1 else None

    def _extract_category(
        self,
        text: str,
        patterns_by_category: dict
    ) -> str | None:
        """Extrait une catégorie unique."""
        detected = {
            category
            for category, patterns in patterns_by_category.items()
            if self._has_pattern(text, patterns)
        }

        return detected.pop() if len(detected) == 1 else None

    def _split_clinical_items(self, value: str) -> list[str]:
        """Découpe une liste clinique."""
        return [
            item
            for item in map(
                clean_text,
                CLINICAL_ITEM_SPLIT.split(
                    clean_text(value)
                )
            )
            if self._valid_text_value(item)
        ]

    def _valid_text_value(self, value: str) -> bool:
        """Valide un fragment clinique."""
        return bool(
            value
            and 3 <= len(value) <= 240
            and len(value.split()) <= 30
        )

    def _has_pattern(
        self,
        text: str,
        patterns: re.Pattern | tuple[re.Pattern, ...]
    ) -> bool:
        """Teste un ou plusieurs patterns."""
        patterns = (
            (patterns,)
            if isinstance(patterns, re.Pattern)
            else patterns
        )

        return any(
            pattern.search(text)
            for pattern in patterns
        )

    def _unique_texts(self, values: list[str]) -> list[str]:
        """Déduplique les textes."""
        unique = {}

        for value in values:
            value = clean_text(value)

            if value:
                unique.setdefault(
                    key_text(value).casefold(),
                    value
                )

        return list(unique.values())