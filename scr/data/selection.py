import math
from collections import Counter, defaultdict

from scr.config import (
    CONFIG,
    SOURCE_REGISTRY,
    SourceSchema
)
from scr.data.cleaning import unresolved
from scr.data.semantic import SemanticConfig
from scr.utils.hashing import digest

class AdmissionSelector:
    """Sélectionne un seul record admissible par groupe de doublons."""

    def __init__(self, records: list[dict]) -> None:
        """Initialise le sélecteur."""
        self.records = records

    def select(self) -> list[dict]:
        """Sélectionne les records admissibles."""
        return self._deduplicate(self._filter_admissible())


    # Admissibilité
    # Attribue un statut initial selon les contrôles humains, RGPD et qualité.

    def _filter_admissible(self) -> list[dict]:
        """Filtre les records admissibles."""
        admitted = []

        for record in self.records:
            record["duplicate_of"] = None
            record["status"] = self._status(record)

            if record["status"] == "ELIGIBLE":
                admitted.append(record)

        return admitted

    def _status(self, record: dict) -> str:
        """Détermine le statut d'admissibilité."""
        flags = set(record.get("flags") or [])

        if record.get("reviewer_excluded", False):
            return "HUMAN_EXCLUDED"

        if "PRIVACY_EXCLUDED" in flags:
            return "PRIVACY_EXCLUDED"

        if "PII_REDACTION_FAILED" in flags:
            return "PRIVACY_FAILED"

        if not SOURCE_REGISTRY[record["source"]].allowed:
            return "SOURCE_BLOCKED"

        if unresolved(record):
            return "QUARANTINE"

        return "ELIGIBLE"


    # Dédoublonnage
    # Conserve un seul représentant déterministe par group_id.

    def _deduplicate(self, admitted: list[dict]) -> list[dict]:
        """Élimine les doublons du même groupe."""
        seen = {}
        retained = []

        for record in sorted(admitted, key=lambda item: item["id"]):
            group_id = record.get("group_id")

            if group_id is None:
                raise RuntimeError(
                    f"Record admissible sans group_id : {record['id']}"
                )

            duplicate_of = seen.get(group_id)

            if duplicate_of is not None:
                record["status"] = "DUPLICATE"
                record["duplicate_of"] = duplicate_of
                continue

            seen[group_id] = record["id"]
            retained.append(record)

        return retained

class StratifiedSelector:
    """Sélectionne les meilleurs exemples TRIAGE avec diversité."""

    def __init__(
        self,
        records: list[dict]
    ) -> None:
        """Initialise le sélecteur."""
        self.records = records

    def cap_groups(
        self,
        cap: int | None
    ) -> list[dict]:
        """Limite les exemples par groupe."""
        if cap is not None and cap < 1:
            raise ValueError(
                "Le plafond doit être positif ou None."
            )

        counts = Counter()
        selected = []

        for record in sorted(
            self.records,
            key=self._stable_order
        ):
            group_id = (
                record.get("group_id")
                or record["id"]
            )

            if (
                cap is not None
                and counts[group_id] >= cap
            ):
                continue

            counts[group_id] += 1
            selected.append(record)

        return selected

    def pick(
        self,
        target: int
    ) -> list[dict]:
        """Sélectionne un volume stratifié."""
        if target < 0:
            raise ValueError(
                "Le volume cible doit être positif ou nul."
            )

        if not self.records or target == 0:
            return []

        strata = self._build_strata()

        capacities = {
            key: len(records)
            for key, records in strata.items()
        }

        weights = {
            key: math.sqrt(capacity)
            for key, capacity in capacities.items()
        }

        quotas = self._allocate(
            capacities,
            weights,
            target
        )

        selected = []

        for key in sorted(strata):
            selected.extend(
                sorted(
                    strata[key],
                    key=self._quality_order
                )[:quotas[key]]
            )

        return selected

    def _build_strata(
        self
    ) -> dict[tuple, list[dict]]:
        """Construit les strates TRIAGE."""
        strata = defaultdict(list)

        for record in self.records:
            training_type = record.get(
                "training_type"
            )
            language = record.get(
                "language"
            )
            priority = record.get(
                "priority"
            )

            if training_type not in {"sft", "dpo"}:
                raise ValueError(
                    "Type d’entraînement invalide : "
                    f"{training_type!r}."
                )

            if language not in {"fr", "en"}:
                raise ValueError(
                    f"Langue invalide : {language!r}."
                )

            if priority not in {1, 2, 3}:
                raise ValueError(
                    "Priorité TRIAGE invalide : "
                    f"{priority!r}."
                )

            strata[
                training_type,
                language,
                priority
            ].append(record)

        return strata

    def _stable_order(
        self,
        record: dict
    ) -> str:
        """Produit un ordre déterministe."""
        return digest([
            CONFIG.seed,
            record["id"]
        ])

    def _quality_order(
        self,
        record: dict
    ) -> tuple:
        """Classe les exemples par qualité."""
        quality = (
            record.get("semantic_quality")
            or {}
        )

        quality_score = float(
            quality.get(
                "quality_score",
                0.0
            )
        )

        confidence_score = float(
            record.get(
                "confidence_score",
                0.0
            )
        )

        verdict_rank = {
            SemanticConfig.PASS: 0,
            SemanticConfig.REVIEW: 1,
            SemanticConfig.FAIL: 2
        }.get(
            quality.get("verdict"),
            3
        )

        return (
            -quality_score,
            -confidence_score,
            verdict_rank,
            len(record.get("notes") or []),
            self._stable_order(record)
        )

    def _allocate(
        self,
        capacities: dict,
        weights: dict,
        target: int
    ) -> dict:
        """Calcule les quotas."""
        if target < 0:
            raise ValueError(
                "Le volume cible doit être positif ou nul."
            )

        if any(
            weight <= 0
            for weight in weights.values()
        ):
            raise ValueError(
                "Toutes les pondérations doivent être positives."
            )

        quotas = dict.fromkeys(
            capacities,
            0
        )

        remaining = min(
            target,
            sum(capacities.values())
        )

        while remaining:
            active = [
                key
                for key, capacity in capacities.items()
                if quotas[key] < capacity
            ]

            total_weight = sum(
                weights[key]
                for key in active
            )

            ideal = {
                key: (
                    remaining
                    * weights[key]
                    / total_weight
                )
                for key in active
            }

            additions = {
                key: min(
                    capacities[key] - quotas[key],
                    math.floor(ideal[key])
                )
                for key in active
            }

            if not sum(additions.values()):
                ranked = sorted(
                    active,
                    key=lambda key: (
                        -(ideal[key] % 1),
                        str(key)
                    )
                )

                for key in ranked[:remaining]:
                    additions[key] = 1

            for key, count in additions.items():
                quotas[key] += count
                remaining -= count

        return quotas
class SFTSourceConfig:
    """Configuration des sources destinées au SFT."""

    ALLOWED_FAMILIES = {
        "fr": SourceSchema.QCM_FAMILIES,
        "en": {"open"}
    }

    EN_SOURCE = "MedQuad-KV"


class SFTSourcePoolBuilder:
    """Construit le pool des sources destinées au SFT TRIAGE."""

    def __init__(
        self,
        records: list[dict]
    ) -> None:
        """Initialise le constructeur."""
        self.records = records

    def build(
        self
    ) -> tuple[list[dict], dict]:
        """Construit le pool source SFT."""
        candidates = sorted(
            (
                record
                for record in self.records
                if self._is_candidate(record)
            ),
            key=self._stable_order
        )

        for record in candidates:
            record["training_type"] = "sft"

        return (
            candidates,
            self._build_profile(candidates)
        )

    def _is_candidate(
        self,
        record: dict
    ) -> bool:
        """Vérifie l'admissibilité source."""
        language = record.get("language")
        family = record.get("family")

        if family not in SFTSourceConfig.ALLOWED_FAMILIES.get(
            language,
            set()
        ):
            return False

        if (
            record.get(
                "semantic_quality",
                {}
            ).get("verdict")
            != SemanticConfig.PASS
        ):
            return False

        return (
            language != "en"
            or record.get("source") == SFTSourceConfig.EN_SOURCE
        )

    def _stable_order(
        self,
        record: dict
    ) -> str:
        """Produit un ordre déterministe."""
        return digest([
            CONFIG.seed,
            record["id"]
        ])

    def _build_profile(
        self,
        candidates: list[dict]
    ) -> dict:
        """Construit le profil du pool."""
        scores = [
            record["semantic_quality"]["quality_score"]
            for record in candidates
        ]

        return {
            "total": len(candidates),
            "languages": dict(Counter(
                record.get("language", "UNKNOWN")
                for record in candidates
            )),
            "sources": dict(Counter(
                record.get("source", "UNKNOWN")
                for record in candidates
            )),
            "families": dict(Counter(
                record.get("family", "UNKNOWN")
                for record in candidates
            )),
            "verdicts": dict(Counter(
                record["semantic_quality"]["verdict"]
                for record in candidates
            )),
            "mean_quality_score": (
                round(
                    sum(scores) / len(scores),
                    4
                )
                if scores
                else 0.0
            )
        }