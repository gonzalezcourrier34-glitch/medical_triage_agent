from scr.config import CONFIG
from scr.data.cleaning import key_text


class UnionFind:
    """Fusionne efficacement des groupes d'identifiants liés."""

    def __init__(self, ids: list[str]) -> None:
        """Initialise les groupes."""
        self.parent = {
            record_id: record_id
            for record_id in ids
        }

    def find(self, record_id: str) -> str:
        """Retourne la racine du groupe."""
        while self.parent[record_id] != record_id:
            self.parent[record_id] = self.parent[
                self.parent[record_id]
            ]
            record_id = self.parent[record_id]

        return record_id

    def union(self, first_id: str, second_id: str) -> None:
        """Fusionne deux groupes."""
        first_root = self.find(first_id)
        second_root = self.find(second_id)

        if first_root == second_root:
            return

        lower_root, upper_root = sorted((
            first_root,
            second_root
        ))

        self.parent[upper_root] = lower_root


class DuplicateAnalyzer:
    """Construit les groupes de doublons exacts et approchés."""

    def __init__(self, records: list[dict]) -> None:
        """Initialise l'analyseur."""
        self.candidates = [
            record
            for record in records
            if self._is_candidate(record)
        ]

        self.records_by_id = {
            record["id"]: record
            for record in self.candidates
        }

        self.union_find = UnionFind(
            list(self.records_by_id)
        )

    def build_exact_groups(self) -> None:
        """Construit les groupes exacts."""
        key_owner = {}

        for record in self.candidates:
            for key in self._comparison_keys(record):
                owner_id = key_owner.setdefault(
                    key,
                    record["id"]
                )

                if owner_id != record["id"]:
                    self.union_find.union(
                        record["id"],
                        owner_id
                    )

    def find_near_edges(self) -> list[dict]:
        """Recherche les quasi-doublons."""
        from datasketch import MinHash, MinHashLSH
        from rapidfuzz import fuzz

        unique_texts = self._unique_comparison_texts()

        lsh = MinHashLSH(
            threshold=CONFIG.lsh_threshold,
            num_perm=CONFIG.lsh_num_perm
        )

        indexed_texts = {}
        edges = []
        comparisons = 0

        for index, (normalized_text, record_id) in enumerate(
            sorted(unique_texts.items())
        ):
            search_text = normalized_text.casefold()

            grams = {
                search_text[position:position + 4].encode("utf-8")
                for position in range(len(search_text) - 3)
            }

            signature = MinHash(
                num_perm=CONFIG.lsh_num_perm,
                seed=CONFIG.seed
            )

            signature.update_batch(
                sorted(grams)
            )

            for candidate_key in sorted(
                lsh.query(signature)
            ):
                candidate_text, candidate_id = (
                    indexed_texts[candidate_key]
                )

                comparisons += 1

                if comparisons > CONFIG.max_near_candidates:
                    raise RuntimeError(
                        "Trop de comparaisons proches : "
                        "analyser les grands groupes."
                    )

                if record_id == candidate_id:
                    continue

                score = fuzz.ratio(
                    search_text,
                    candidate_text.casefold()
                )

                if score >= CONFIG.near_threshold:
                    edges.append({
                        "first_id": candidate_id,
                        "second_id": record_id,
                        "score": round(score, 2),
                        "origin": "GLOBAL_LSH"
                    })

            index_key = str(index)

            indexed_texts[index_key] = (
                normalized_text,
                record_id
            )

            lsh.insert(
                index_key,
                signature
            )

            if index and index % 10_000 == 0:
                print(
                    f"Recherche approchée : "
                    f"{index:,} textes indexés"
                )

        return edges

    def apply_near_edges(self, edges: list[dict]) -> None:
        """Applique les quasi-doublons."""
        if not edges:
            return

        self._validate_edge_ids(edges)

        touched_roots = {
            self.union_find.find(record_id)
            for edge in edges
            for record_id in (
                edge["first_id"],
                edge["second_id"]
            )
        }

        for record in self.candidates:
            if (
                self.union_find.find(record["id"])
                in touched_roots
            ):
                record["flags"].add(
                    "NEAR_REVIEW"
                )

        for edge in edges:
            self.union_find.union(
                edge["first_id"],
                edge["second_id"]
            )

    def _is_candidate(self, record: dict) -> bool:
        """Indique si le record est candidat."""
        flags = record.get("flags", set())

        return (
            not record.get("reviewer_excluded", False)
            and "SCHEMA_UNSUPPORTED" not in flags
            and "PRIVACY_EXCLUDED" not in flags
        )

    def _comparison_keys(
        self,
        record: dict
    ) -> set[tuple]:
        """Construit les clés exactes."""
        keys = {
            ("text", record["input_key"]),
            ("text", record["original_input_key"])
        }

        for key in (
            record.get("context_key"),
            record.get("original_context_key")
        ):
            if key:
                keys.add(
                    ("text", key)
                )

        if record.get("choices"):
            keys.add((
                "qcm_unordered",
                record["question_key"]
            ))

        if (
            record["family"] == "dpo"
            and record.get("source_id")
        ):
            keys.add((
                "prompt_id",
                record["source"],
                record["source_id"]
            ))

        return keys

    def _unique_comparison_texts(self) -> dict[str, str]:
        """Construit les textes uniques comparables."""
        unique_texts = {}

        for record in sorted(
            self.candidates,
            key=lambda item: item["id"]
        ):
            for field in ("input", "context"):
                normalized = key_text(
                    record.get(field, "")
                )

                if len(normalized) >= 30:
                    unique_texts.setdefault(
                        normalized,
                        record["id"]
                    )

        return unique_texts

    def _validate_edge_ids(
        self,
        edges: list[dict]
    ) -> None:
        """Valide les IDs des arêtes."""
        known_ids = self.records_by_id.keys()

        if any(
            edge["first_id"] not in known_ids
            or edge["second_id"] not in known_ids
            for edge in edges
        ):
            raise ValueError(
                "ID inconnu dans les quasi-doublons : audit périmé."
            )