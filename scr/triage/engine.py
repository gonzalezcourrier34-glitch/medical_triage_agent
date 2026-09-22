import re
import unicodedata

from scr.data.cleaning import clean_text
from scr.triage.confidence import triage_confidence_label
from scr.triage.protocol import AutomaticTriageEngineConfig


class AutomaticTriageDecisionEngine:
    """Moteur déterministe FRENCH."""

    MIN_PARTIAL_SCORE = 0.5
    PATIENT_FIELDS = {
        "age_days", "age_months", "age_years", "sex", "symptoms",
        "medical_history", "duration", "evolution", "critical_signs",
        "critical_sign_states"
    }

    def __init__(self, protocol: dict) -> None:
        self.protocol = protocol
        self.mapping = (
            protocol.get("mapping")
            or protocol.get("priority_mapping", {}).get("french_to_target")
        )
        if not isinstance(self.mapping, dict):
            raise ValueError("Mapping FRENCH absent.")

        self.motifs = {
            clean_text(m["rule_id"]): m
            for m in protocol.get("motifs", [])
            if clean_text(m.get("rule_id"))
        }
        self.rules = self._build_rules()

    # Protocole
    # Construit uniquement les règles automatisables.

    def _build_rules(self) -> list[dict]:
        rules = []

        for motif in self.protocol.get("motifs", []):
            rule_id = clean_text(motif.get("rule_id"))

            for level, criterion in (motif.get("triage_levels") or {}).items():
                automation = criterion.get("automation") if isinstance(criterion, dict) else None
                if not automation:
                    continue

                level = clean_text(level).upper()
                if level not in self.mapping:
                    raise ValueError(f"Niveau FRENCH inconnu : {level}")

                rules.append({
                    "rule_id": rule_id,
                    "french_level": level,
                    "priority": self.mapping[level],
                    "reason": clean_text(criterion.get("fr")),
                    "criteria": automation
                })

        return sorted(rules, key=self._rule_sort_key)

    def _rule_sort_key(self, rule: dict) -> tuple:
        return (
            AutomaticTriageEngineConfig.LEVEL_ORDER[rule["french_level"]],
            rule["rule_id"]
        )

    # Données
    # Centralise patient, knowledge et motifs candidats.

    @staticmethod
    def _patient(record: dict) -> dict:
        """Retourne les données patient extraites."""
        return record.get("triage_clinical_enrichment") or {}

    @staticmethod
    def _medical_knowledge(record: dict) -> dict:
        return record.get("medical_knowledge") or {}

    def _candidates(self, record: dict, source: str) -> list[dict]:
        data = (
            self._patient(record)
            if source == "patient"
            else self._medical_knowledge(record)
        )
        return [
            c for c in data.get("french_motif_candidates") or []
            if isinstance(c, dict)
        ]

    @staticmethod
    def _rule_ids(candidates: list[dict]) -> list[str]:
        return sorted({
            clean_text(c.get("rule_id"))
            for c in candidates
            if clean_text(c.get("rule_id"))
        })

    def _has_patient_evidence(self, record: dict) -> bool:
        patient = self._patient(record)
        fields = set((patient.get("extraction") or {}).get("extracted_fields") or [])

        return bool(
            fields & self.PATIENT_FIELDS
            or any(field.startswith("vital_signs.") for field in fields)
            or patient.get("vital_signs")
        )

    # Décision
    # Sépare strictement triage patient et connaissance médicale.

    def build_decisions(self, records: list[dict]) -> list[dict]:
        return [self._evaluate(record) for record in records]

    def _evaluate(self, record: dict) -> dict:
        return (
            self._evaluate_patient(record)
            if self._has_patient_evidence(record)
            else self._evaluate_knowledge(record)
        )

    def _evaluate_patient(self, record: dict) -> dict:
        candidates = self._candidates(record, "patient")

        if not candidates:
            return self._undetermined(
                record, "no_patient_motif",
                "Aucun motif FRENCH patient.", "patient_triage"
            )

        rule_ids = self._rule_ids(candidates)
        rules = [r for r in self.rules if r["rule_id"] in rule_ids]

        if not rules:
            return self._undetermined(
                record, "unsupported_protocol_rule",
                "Motif identifié sans règle automatique.",
                "patient_triage", rule_ids
            )

        matched = [r for r in rules if self._matches(record, r["criteria"])]

        if matched:
            return self._resolve_exact(record, matched)

        return self._fallback(record, rules)

    # Règles exactes
    # Une priorité unique est acceptée, sinon le conflit reste explicite.

    def _resolve_exact(self, record: dict, rules: list[dict]) -> dict:
        rules = sorted(rules, key=self._rule_sort_key)
        priorities = {r["priority"] for r in rules}
        rule_ids = sorted({r["rule_id"] for r in rules})
        levels = sorted({r["french_level"] for r in rules})

        if len(priorities) > 1:
            return self._conflict(
                record, "exact_rule_conflict",
                "Règles exactes de priorités différentes.",
                "patient_triage", rule_ids, priorities, levels
            )

        method = "exact_rule" if len(rules) == 1 else "exact_rule_convergence"
        return self._decision(
            record, rules[0], 1.0, method,
            "patient_triage", rule_ids, levels
        )

    # Knowledge
    # Seules les références resolved et utilisables peuvent attribuer une priorité.

    def _evaluate_knowledge(self, record: dict) -> dict:
        candidates = self._candidates(record, "knowledge")

        if not candidates:
            return self._undetermined(
                record, "no_knowledge_motif",
                "Aucun motif FRENCH knowledge.", "medical_knowledge"
            )

        references = self._knowledge_references(candidates)

        if not references:
            return self._undetermined(
                record, "knowledge_reference_missing",
                "Aucune référence de gravité.", "medical_knowledge"
            )

        if all(self._resolved(r) for r in references):
            return self._resolve_knowledge(record, references)

        return self._resolve_knowledge_review(record, references)

    @staticmethod
    def _resolved(reference: dict) -> bool:
        return (
            reference["status"] == "resolved"
            and reference["usable"]
            and not reference["requires_review"]
            and reference["priority"] is not None
        )

    def _resolve_knowledge(self, record: dict, references: list[dict]) -> dict:
        priorities = {r["priority"] for r in references}
        rule_ids = self._reference_rule_ids(references)
        levels = self._reference_levels(references)

        if len(priorities) > 1:
            return self._conflict(
                record, "knowledge_priority_conflict",
                "Références knowledge resolved incompatibles.",
                "medical_knowledge", rule_ids, priorities, levels
            )

        return self._knowledge_decision(
            record, next(iter(priorities)),
            rule_ids, levels, "knowledge_reference"
        )

    # Les références non résolues restent en revue même si elles convergent.
    # La priorité commune est conservée comme candidate, jamais comme label automatique.

    def _resolve_knowledge_review(self, record: dict, references: list[dict]) -> dict:
        sets = [p for r in references if (p := self._reference_priorities(r))]

        if not sets:
            return self._knowledge_review(
                record, references,
                "knowledge_review_required",
                "Références knowledge à valider."
            )

        union = set().union(*sets)
        intersection = set.intersection(*sets)

        if len(union) == 1:
            return self._knowledge_review(
                record, references,
                "knowledge_full_convergence_review",
                "Convergence de références knowledge non résolues.",
                union
            )

        if len(intersection) == 1:
            return self._knowledge_review(
                record, references,
                "knowledge_common_priority_review",
                "Priorité commune entre références knowledge non résolues.",
                intersection
            )

        resolved = {r["priority"] for r in references if self._resolved(r)}

        if len(resolved) == 1:
            priority = next(iter(resolved))
            if all(priority in priorities for priorities in sets):
                return self._knowledge_review(
                    record, references,
                    "knowledge_resolved_dominance_review",
                    "Référence resolved compatible avec les références à valider.",
                    {priority}
                )

        return self._conflict(
            record, "knowledge_priority_conflict",
            "Priorités knowledge incompatibles.",
            "medical_knowledge",
            self._reference_rule_ids(references), union
        )

    def _knowledge_references(self, candidates: list[dict]) -> list[dict]:
        references, seen = [], set()

        for candidate in candidates:
            rule_id = clean_text(candidate.get("rule_id"))
            motif = self.motifs.get(rule_id)

            if not rule_id or rule_id in seen or not motif:
                continue

            seen.add(rule_id)
            ref = motif.get("knowledge_reference")

            if not isinstance(ref, dict):
                continue

            references.append({
                "rule_id": rule_id,
                "status": clean_text(ref.get("status")).lower(),
                "priority": ref.get("priority"),
                "priority_label": ref.get("priority_label"),
                "french_levels": list(ref.get("french_levels") or []),
                "candidate_priorities": list(ref.get("candidate_priorities") or []),
                "usable": bool(ref.get("usable_for_medical_knowledge")),
                "requires_review": bool(ref.get("requires_review")),
                "basis": ref.get("basis")
            })

        return references

    def _reference_priorities(self, reference: dict) -> set[int]:
        if self._resolved(reference):
            return {reference["priority"]}

        priorities = {
            p for p in reference["candidate_priorities"]
            if p in {1, 2, 3}
        }

        if reference["priority"] in {1, 2, 3}:
            priorities.add(reference["priority"])

        return priorities

    @staticmethod
    def _reference_rule_ids(references: list[dict]) -> list[str]:
        return sorted({r["rule_id"] for r in references})

    @staticmethod
    def _reference_levels(references: list[dict]) -> list[str]:
        return sorted({x for r in references for x in r["french_levels"]})

    # Fallback
    # Compare les règles partielles puis utilise la qualité des preuves en cas d'égalité.

    def _fallback(self, record: dict, rules: list[dict]) -> dict:
        scored = [
            (self._score(record, r["criteria"]), r)
            for r in rules
        ]
        scored = [(s, r) for s, r in scored if s >= self.MIN_PARTIAL_SCORE]

        if not scored:
            return self._fallback_failure(record, rules)

        score = max(s for s, _ in scored)
        best = [r for s, r in scored if s == score]

        return self._resolve_partial(record, best, score)

    def _fallback_failure(self, record: dict, rules: list[dict]) -> dict:
        rule_ids = sorted({r["rule_id"] for r in rules})
        missing = sorted({
            field
            for rule in rules
            for field in self._missing_fields(record, rule["criteria"])
        })

        if missing:
            decision = self._undetermined(
                record, "missing_required_field",
                "Données requises absentes.",
                "patient_triage", rule_ids
            )
            decision["missing_fields"] = missing
            return decision

        return self._undetermined(
            record, "no_matching_criterion",
            "Aucune règle partielle n'atteint le seuil.",
            "patient_triage", rule_ids
        )

    def _resolve_partial(
        self,
        record: dict,
        rules: list[dict],
        score: float
    ) -> dict:
        rules = sorted(rules, key=self._rule_sort_key)
        rule_ids = {r["rule_id"] for r in rules}
        priorities = {r["priority"] for r in rules}

        if len(priorities) > 1:
            quality = [(self._evidence_quality(record, r), r) for r in rules]
            best_quality = max(q for q, _ in quality)
            best = [r for q, r in quality if q == best_quality]
            best_priorities = {r["priority"] for r in best}

            if len(best_priorities) == 1:
                method = (
                    "deterministic_evidence_dominance"
                    if len(rule_ids) == 1
                    else "deterministic_multi_evidence_dominance"
                )
                return self._decision(
                    record, best[0], score, method,
                    "patient_triage_candidate",
                    sorted({r["rule_id"] for r in best}),
                    sorted({r["french_level"] for r in best})
                )

        rule_ids = sorted(rule_ids)
        levels = sorted({r["french_level"] for r in rules})

        if len(priorities) > 1:
            decision = self._conflict(
                record, "deterministic_priority_conflict",
                "Règles partielles de même score avec priorités différentes.",
                "patient_triage_candidate", rule_ids, priorities, levels
            )
            decision["best_score"] = round(score, 4)
            return decision

        method = "deterministic_unique" if len(rules) == 1 else "deterministic_same_priority"
        return self._decision(
            record, rules[0], score, method,
            "patient_triage_candidate", rule_ids, levels
        )

    # Résultats
    # Uniformise décisions validées, knowledge, conflits et indéterminés.

    def _decision(
        self,
        record: dict,
        rule: dict,
        score: float,
        method: str,
        scope: str,
        rule_ids: list[str] | None = None,
        levels: list[str] | None = None
    ) -> dict:
        return self._result(
            record, rule["priority"], score, method, scope,
            rule_ids or [rule["rule_id"]],
            levels or [rule["french_level"]],
            rule["reason"]
        )

    def _knowledge_decision(
        self,
        record: dict,
        priority: int,
        rule_ids: list[str],
        levels: list[str],
        method: str
    ) -> dict:
        reasons = {
            "knowledge_reference": "Priorité knowledge_reference FRENCH.",
            "knowledge_full_convergence": "Toutes les références FRENCH convergent.",
            "knowledge_common_priority": "Toutes les références partagent une priorité.",
            "knowledge_resolved_dominance": "Les références resolved convergent."
        }
        return self._result(
            record, priority, 1.0, method,
            "medical_knowledge_reference", rule_ids, levels,
            reasons[method]
        )

    def _result(
        self,
        record: dict,
        priority: int,
        score: float,
        method: str,
        scope: str,
        rule_ids: list[str],
        levels: list[str],
        reason: str
    ) -> dict:
        rule_ids = sorted(set(rule_ids))
        levels = sorted(set(levels))
        score = round(score, 4)

        return {
            "id": record["id"],
            "clean_content_hash": record["clean_content_hash"],
            "priority": priority,
            "french_level": levels[0] if len(levels) == 1 else None,
            "reference_french_levels": levels,
            "confidence_score": score,
            "confidence": triage_confidence_label(score),
            "decision_method": method,
            "decision_scope": scope,
            "patient_triage": scope.startswith("patient_triage"),
            "rule_id": rule_ids[0] if len(rule_ids) == 1 else None,
            "rule_ids": rule_ids,
            "reason": reason,
            "protocol": self.protocol["protocol_name"],
            "protocol_version": self.protocol["protocol_version"]
        }

    def _conflict(
        self,
        record: dict,
        method: str,
        reason: str,
        scope: str,
        rule_ids: list[str],
        priorities: set,
        levels: list[str] | None = None
    ) -> dict:
        decision = self._undetermined(record, method, reason, scope, rule_ids)
        decision["candidate_priorities"] = sorted(priorities)
        if levels is not None:
            decision["candidate_french_levels"] = sorted(levels)
        return decision

    # Conserve toutes les informations utiles pour la revue humaine.
    # Aucune priorité candidate n'est transformée en priorité validée.

    def _knowledge_review(
        self,
        record: dict,
        references: list[dict],
        method: str,
        reason: str,
        priorities: set | None = None
    ) -> dict:
        decision = self._undetermined(
            record, method, reason,
            "medical_knowledge",
            self._reference_rule_ids(references)
        )

        decision["candidate_priorities"] = sorted(
            priorities
            if priorities is not None
            else {p for r in references for p in self._reference_priorities(r)}
        )
        decision["candidate_french_levels"] = self._reference_levels(references)
        decision["knowledge_reference_statuses"] = sorted({
            r["status"] for r in references
        })
        decision["requires_review"] = True

        return decision

    def _undetermined(
        self,
        record: dict,
        method: str,
        reason: str,
        scope: str,
        rule_ids: list[str] | None = None
    ) -> dict:
        rule_ids = sorted(set(rule_ids or []))

        return {
            "id": record["id"],
            "clean_content_hash": record["clean_content_hash"],
            "priority": None,
            "french_level": None,
            "reference_french_levels": [],
            "confidence_score": 0.0,
            "confidence": "low",
            "decision_method": method,
            "decision_scope": scope,
            "patient_triage": scope.startswith("patient_triage"),
            "rule_id": rule_ids[0] if len(rule_ids) == 1 else None,
            "rule_ids": rule_ids,
            "reason": reason,
            "protocol": self.protocol["protocol_name"],
            "protocol_version": self.protocol["protocol_version"]
        }

    # Conditions
    # Distingue matched, missing et not_matched sans masquer les données absentes.

    def _condition_status(self, record: dict, condition: dict) -> str:
        if not isinstance(condition, dict):
            return "invalid"

        for group in ("all", "any"):
            if group not in condition:
                continue

            statuses = [self._condition_status(record, c) for c in condition[group]]

            if group == "all":
                if any(s == "not_matched" for s in statuses):
                    return "not_matched"
                if all(s == "matched" for s in statuses):
                    return "matched"
            else:
                if any(s == "matched" for s in statuses):
                    return "matched"
                if all(s == "not_matched" for s in statuses):
                    return "not_matched"

            if any(s == "missing" for s in statuses):
                return "missing"

            return "invalid"

        field = clean_text(condition.get("field"))
        operator = clean_text(condition.get("operator")).lower()

        if not field or operator not in AutomaticTriageEngineConfig.OPERATORS:
            return "invalid"

        actual = self._field_value(record, field)

        if actual is None:
            return "missing"

        return "matched" if self._compare(actual, operator, condition.get("value")) else "not_matched"

    def _matches(self, record: dict, group: dict) -> bool:
        return self._condition_status(record, group) == "matched"

    def _missing_fields(self, record: dict, group: dict) -> list[str]:
        if self._condition_status(record, group) != "missing":
            return []

        missing = []

        def visit(condition: dict) -> None:
            if not isinstance(condition, dict):
                return

            for key in ("all", "any"):
                if key in condition:
                    for child in condition[key]:
                        visit(child)
                    return

            field = clean_text(condition.get("field"))
            if field and self._field_value(record, field) is None:
                missing.append(field)

        visit(group)
        return sorted(set(missing))

    # Score
    # Mesure la couverture d'une règle et départage les égalités par qualité de preuve.

    def _score(self, record: dict, group: dict) -> float:
        if not isinstance(group, dict):
            return 0.0

        if "all" in group:
            values = [self._score(record, c) for c in group["all"]]
            return sum(values) / len(values) if values else 0.0

        if "any" in group:
            values = [self._score(record, c) for c in group["any"]]
            return max(values, default=0.0)

        return float(self._condition_status(record, group) == "matched")

    def _evidence_quality(self, record: dict, rule: dict) -> tuple:
        statuses = []

        def visit(condition: dict) -> None:
            if not isinstance(condition, dict):
                return
            for key in ("all", "any"):
                if key in condition:
                    for child in condition[key]:
                        visit(child)
                    return
            statuses.append(self._condition_status(record, condition))

        visit(rule["criteria"])
        return (-statuses.count("missing"), statuses.count("matched"))

    # Champs
    # Lit les valeurs cliniques, le texte normalisé et le shock index dérivé.

    def _field_value(self, record: dict, field: str):
        if field == "text":
            return self._record_text(record)
        if field == "derived.shock_index":
            return self._shock_index(record)

        field = field.removeprefix("patient.")
        current = self._patient(record)

        for part in field.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]

        return current

    def _compare(self, actual, operator: str, expected) -> bool:
        if actual is None:
            return False

        if operator == "contains_any":
            if not isinstance(expected, list):
                return False
            text = self._normalize_text(actual)
            return any(
                value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text)
                for item in expected
                if (value := self._normalize_text(item))
            )

        if operator == "==":
            if isinstance(actual, str) or isinstance(expected, str):
                return self._normalize_text(actual) == self._normalize_text(expected)
            try:
                return float(actual) == float(expected)
            except (TypeError, ValueError):
                return actual == expected

        try:
            actual, expected = float(actual), float(expected)
        except (TypeError, ValueError):
            return False

        return {
            "<": actual < expected,
            "<=": actual <= expected,
            ">": actual > expected,
            ">=": actual >= expected
        }.get(operator, False)

    def _record_text(self, record: dict) -> str:
        patient = self._patient(record)
        parts = [clean_text(record.get("context")), clean_text(record.get("question"))]

        for field in ("symptoms", "medical_history", "critical_signs", "condition_mentions"):
            value = patient.get(field)
            if isinstance(value, list):
                parts.extend(clean_text(v) for v in value if clean_text(v))
            elif clean_text(value):
                parts.append(clean_text(value))

        return self._normalize_text(" ".join(filter(None, parts)))

    @staticmethod
    def _normalize_text(value) -> str:
        text = unicodedata.normalize("NFKD", clean_text(value).casefold())
        return "".join(c for c in text if not unicodedata.combining(c))

    def _shock_index(self, record: dict) -> float | None:
        vitals = self._patient(record).get("vital_signs") or {}

        try:
            heart_rate = float(vitals["heart_rate"])
            systolic_bp = float(vitals["systolic_bp"])
        except (KeyError, TypeError, ValueError):
            return None

        return heart_rate / systolic_bp if heart_rate > 0 and systolic_bp > 0 else None