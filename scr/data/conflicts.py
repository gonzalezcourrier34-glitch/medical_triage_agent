from collections import Counter, defaultdict, deque

from scr.data.cleaning import key_text
from scr.utils.hashing import digest


class FingerprintBuilder:
    """Construit les empreintes de comparaison d'un record."""

    def __init__(self, record: dict) -> None:
        """Initialise le constructeur."""
        self.record = record

    def build(self) -> None:
        """Construit les empreintes."""
        context = self.record.get("context", "")
        target = self._target_signature()

        self.record["input_key"] = digest(
            key_text(self.record["input"])
        )

        self.record["context_key"] = (
            digest(key_text(context))
            if context
            else None
        )

        self.record["question_key"] = digest(
            self._question_signature()
        )

        self.record["target_key"] = digest(target)

        self.record["example_key"] = digest([
            self.record["family"],
            self.record["input_key"],
            target
        ])

    def _question_signature(self) -> list:
        """Construit la signature de la question."""
        return [
            key_text(self.record.get("context", "")),
            key_text(self.record.get("question", "")),
            sorted(
                key_text(choice)
                for choice in self.record.get(
                    "choices",
                    {}
                ).values()
            )
        ]

    def _target_signature(self):
        """Construit la signature de la cible."""
        if self.record["family"] == "dpo":
            return [
                key_text(self.record.get("chosen", "")),
                key_text(self.record.get("rejected", ""))
            ]

        if self.record.get("choices"):
            return sorted(
                self.record.get("labels", [])
            )

        return key_text(
            self.record.get("answer", "")
        )


def has_cycle(edges: list[tuple[str, str]]) -> bool:
    """Détecte un cycle de préférences."""
    graph = defaultdict(set)
    incoming = Counter()
    nodes = set()

    for source, target in edges:
        nodes.update((source, target))

        if target in graph[source]:
            continue

        graph[source].add(target)
        incoming[target] += 1

    queue = deque(
        node
        for node in nodes
        if incoming[node] == 0
    )

    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1

        for target in graph[node]:
            incoming[target] -= 1

            if incoming[target] == 0:
                queue.append(target)

    return visited != len(nodes)


class ConflictAnalyzer:
    """Détecte les contradictions QCM et incohérences DPO."""

    def __init__(self, records: list[dict]) -> None:
        """Initialise l'analyseur."""
        self.records = records
        self.qcm_groups = defaultdict(list)
        self.dpo_groups = defaultdict(list)

    def analyze(self) -> None:
        """Analyse les conflits."""
        self._build_groups()
        self._flag_qcm_conflicts()
        self._flag_dpo_conflicts()

    def _build_groups(self) -> None:
        """Construit les groupes comparables."""
        for record in self.records:
            FingerprintBuilder(record).build()

            if self._is_excluded(record):
                continue

            input_key = record["input_key"]

            if record.get("choices"):
                self.qcm_groups[input_key].append(record)

            if record["family"] == "dpo":
                self.dpo_groups[input_key].append(record)

    # Exclusions
    # Écarte tout record non exploitable avant son ajout aux groupes de conflits.

    def _is_excluded(self, record: dict) -> bool:
        """Indique si le record est exclu."""
        flags = set(record.get("flags") or [])

        return (
            record.get("reviewer_excluded", False)
            or bool(
                flags & {
                    "SCHEMA_UNSUPPORTED",
                    "PRIVACY_EXCLUDED",
                    "PII_REDACTION_FAILED"
                }
            )
        )

    def _flag_qcm_conflicts(self) -> None:
        """Détecte les conflits QCM."""
        for group in self.qcm_groups.values():
            targets = {
                record["target_key"]
                for record in group
            }

            if len(targets) <= 1:
                continue

            for record in group:
                record["flags"].add(
                    "QCM_TARGET_CONFLICT"
                )

    def _flag_dpo_conflicts(self) -> None:
        """Détecte les conflits DPO."""
        for group in self.dpo_groups.values():
            edges = self._preference_edges(group)
            edge_set = set(edges)

            inverse_edges = {
                edge
                for edge in edge_set
                if edge[::-1] in edge_set
            }

            for record, edge in zip(group, edges):
                if edge in inverse_edges:
                    record["flags"].add(
                        "DPO_INVERSION"
                    )

            if has_cycle(edges):
                for record in group:
                    record["flags"].add(
                        "DPO_PREFERENCE_CYCLE"
                    )

            chosen = {
                source
                for source, _ in edges
            }

            rejected = {
                target
                for _, target in edges
            }

            if chosen & rejected:
                for record in group:
                    record["notes"].add(
                        "RELATIVE_ROLE_CHANGE"
                    )

    def _preference_edges(
        self,
        group: list[dict]
    ) -> list[tuple[str, str]]:
        """Construit les arêtes DPO."""
        return [
            (
                digest(
                    key_text(record.get("chosen", ""))
                ),
                digest(
                    key_text(record.get("rejected", ""))
                )
            )
            for record in group
        ]