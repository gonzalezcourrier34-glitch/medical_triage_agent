import re


# ## Utilitaire

def rx(pattern: str) -> re.Pattern:
    """Compile une regex clinique."""
    return re.compile(pattern, re.IGNORECASE)


# ## Pathologies

CONDITION_PATTERNS = (
    rx(
        r"\b(?:diagnostic\s+de|diagnostiqué(?:e)?\s+(?:avec|pour)|"
        r"atteint(?:e)?\s+de|souffre\s+de|maladie\s+de|"
        r"antécédents?\s+de|connu(?:e)?\s+pour)\s+([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:diagnosed\s+with|diagnosis\s+of|suffers?\s+from|"
        r"history\s+of|known\s+for|known\s+with|medical\s+history\s+of)\s+"
        r"([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:has|had)\s+(?:a|an)?\s*"
        r"((?:acute|chronic|metastatic|recurrent|severe|advanced|"
        r"stage\s+[ivx0-9]+|type\s+[12])\s+[^.;:?!\n]{2,100})"
    )
)


# ## Symptômes

SYMPTOM_PATTERNS = (
    rx(
        r"\b(?:sympt[oô]mes?|signes?\s+cliniques?|"
        r"manifestations?\s+cliniques?)\s*[:\-]\s*([^.;?!\n]{3,180})"
    ),
    rx(
        r"\b(?:symptoms?|clinical\s+signs?|clinical\s+manifestations?)"
        r"\s*[:\-]\s*([^.;?!\n]{3,180})"
    ),
    rx(
        r"\b(?:présente|présentant|se\s+plaint\s+de)\s+"
        r"([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:presents?|presented)\s+with\s+([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:complains?\s+of|experiencing|experienced)\s+"
        r"([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:caractérisé(?:e)?\s+par|associé(?:e)?\s+à)\s+"
        r"([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:characterized\s+by|associated\s+with)\s+"
        r"([^.;?!\n]{3,160})"
    )
)


# ## Médicaments

MEDICATION_PATTERNS = (
    rx(
        r"\b(?:traité(?:e)?\s+par|traitement\s+par|reçoit|recevant|"
        r"administration\s+de|prescription\s+de|prend|sous)\s+"
        r"([^.;:?!\n]{2,120})"
    ),
    rx(
        r"\b(?:treated\s+with|receives?|receiving|administered|"
        r"prescribed|taking|started\s+on|maintained\s+on|currently\s+on)\s+"
        r"([^.;:?!\n]{2,120})"
    ),
    rx(
        r"\bon\s+([^.;:?!\n]{2,80})\s+"
        r"(?:therapy|treatment|medication|medications)\b"
    )
)


# ## Traitements

TREATMENT_PATTERNS = (
    rx(
        r"\b(?:traitement|prise\s+en\s+charge|thérapie|therapie)\s*"
        r"(?:recommandé(?:e)?|indiqué(?:e)?|de\s+choix|consiste\s+en|"
        r"comprend|inclut|:)?\s*([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:treatment|management|therapy)\s*"
        r"(?:recommended|indicated|of\s+choice|consists?\s+of|includes?|:)?"
        r"\s*([^.;?!\n]{3,160})"
    ),
    rx(
        r"\b(?:managed\s+with|management\s+with)\s+([^.;?!\n]{3,140})"
    )
)


# ## Procédures

PROCEDURE_PATTERNS = (
    rx(
        r"\b(?:intervention|procédure|procedure|chirurgie|opération|operation)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),
    rx(
        r"\b(?:surgery|procedure|operation|intervention)\s*[:\-]?\s*"
        r"([^.;?!\n]{3,140})"
    ),
    rx(
        r"\b(?:underwent|undergoing|scheduled\s+for)\s+([^.;?!\n]{3,140})"
    ),
    rx(
        r"\b(?:a\s+subi|doit\s+subir|prévu(?:e)?\s+pour)\s+"
        r"([^.;?!\n]{3,140})"
    )
)


# ## Biologie

LAB_PATTERNS = (
    rx(
        r"\b(?:biologie|bilan\s+biologique|analyse\s+de\s+sang|"
        r"laboratoire|résultats?\s+biologiques?)\s*[:\-]\s*"
        r"([^.;?!\n]{3,220})"
    ),
    rx(
        r"\b(?:laboratory|lab\s+results?|blood\s+test|"
        r"laboratory\s+findings?)\s*[:\-]\s*([^.;?!\n]{3,220})"
    ),
    rx(
        r"\b(?:shows?|showed|reveals?|revealed|demonstrates?)\s+"
        r"((?:elevated|decreased|low|high|increased|reduced)\s+"
        r"[^.;?!\n]{2,120})"
    ),
    rx(
        r"\b(?:montre|montrait|révèle|revele|objective)\s+"
        r"((?:une?\s+)?(?:augmentation|diminution|élévation|elevation|"
        r"baisse)\s+[^.;?!\n]{2,120})"
    )
)


# ## Imagerie

IMAGING_PATTERNS = (
    rx(
        r"\b(?:radiographie|scanner|TDM|IRM|échographie|echographie|imagerie)"
        r"\s*(?:montre|révèle|revele|objective|:|-)?\s*"
        r"([^.;?!\n]{3,180})"
    ),
    rx(
        r"\b(?:x[- ]?ray|CT(?:\s+scan)?|MRI|ultrasound|imaging)"
        r"\s*(?:shows?|showed|reveals?|revealed|demonstrates?|:|-)?\s*"
        r"([^.;?!\n]{3,180})"
    )
)


# ## Tests diagnostiques

DIAGNOSTIC_TEST_PATTERNS = (
    rx(
        r"\b(?:test|examen|bilan)\s+"
        r"(?:diagnostique|diagnostic|de\s+confirmation|confirmatoire)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),
    rx(
        r"\b(?:diagnostic\s+test|confirmatory\s+test|workup)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),
    rx(r"\b(?:confirmed\s+by|diagnosed\s+by)\s+([^.;?!\n]{3,120})"),
    rx(
        r"\b(?:confirmé(?:e)?\s+par|diagnostiqué(?:e)?\s+par)\s+"
        r"([^.;?!\n]{3,120})"
    )
)


# ## Complications

COMPLICATION_PATTERNS = (
    rx(
        r"\b(?:complication(?:s)?|compliqué(?:e)?\s+de|évolue\s+vers|"
        r"ayant\s+entraîné|entraînant)\s+([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:complication(?:s)?|complicated\s+by|progresses?\s+to|"
        r"resulting\s+in|leading\s+to)\s+([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:developed|develops)\s+"
        r"((?:acute|severe|progressive|new(?:ly)?[- ]onset)\s+"
        r"[^.;:?!\n]{2,120})"
    )
)


# ## Facteurs de risque

RISK_FACTOR_PATTERNS = (
    rx(
        r"\b(?:facteurs?\s+de\s+risque|terrain\s+à\s+risque|"
        r"risque\s+accru\s+lié\s+à)\s*[:\-]?\s*([^.;?!\n]{3,180})"
    ),
    rx(
        r"\b(?:risk\s+factors?|high[- ]risk\s+features?|"
        r"increased\s+risk\s+due\s+to)\s*[:\-]?\s*([^.;?!\n]{3,180})"
    )
)


# ## Pathogènes

PATHOGEN_PATTERNS = (
    rx(
        r"\b(?:infection\s+(?:par|à)|causé(?:e)?\s+par|"
        r"germe\s+responsable|agent\s+responsable)\s+"
        r"([^.;:?!\n]{2,100})"
    ),
    rx(
        r"\b(?:infection\s+(?:with|by)|caused\s+by|"
        r"causative\s+(?:organism|pathogen|agent))\s+"
        r"([^.;:?!\n]{2,100})"
    )
)


# ## Causes

CAUSE_PATTERNS = (
    rx(
        r"\b(?:due\s+to|secondary\s+to|caused\s+by|attributed\s+to)\s+"
        r"([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:dû(?:e)?\s+à|du(?:e)?\s+à|secondaire\s+à|"
        r"causé(?:e)?\s+par|attribué(?:e)?\s+à)\s+"
        r"([^.;:?!\n]{3,140})"
    )
)


# ## Anatomie

ANATOMY_TERMS = {
    "fr": {
        "cœur", "poumon", "poumons", "cerveau", "foie", "rein", "reins",
        "estomac", "intestin", "colon", "pancréas", "rate", "thorax",
        "abdomen", "peau", "œil", "yeux", "oreille", "gorge", "larynx",
        "pharynx", "trachée", "bronches", "artère", "veine",
        "moelle épinière"
    },
    "en": {
        "heart", "lung", "lungs", "brain", "liver", "kidney", "kidneys",
        "stomach", "intestine", "colon", "pancreas", "spleen", "chest",
        "abdomen", "skin", "eye", "eyes", "ear", "throat", "larynx",
        "pharynx", "trachea", "bronchi", "artery", "vein", "spinal cord"
    }
}


# ## Nettoyage

# Coupe les fins de capture qui basculent vers la question QCM.
QUESTION_TAIL_PATTERN = rx(
    r"\s*,?\s*\b(?:which\s+of\s+the\s+following|which|what|who|why|how|"
    r"quel(?:le)?s?|lequel|laquelle|parmi\s+les|quelle\s+est)\b.*$"
)

VALUE_EXCLUSION_PATTERNS = (
    rx(
        r"^(?:which|what|who|why|how|quel(?:le)?s?|lequel|"
        r"laquelle|parmi)\b"
    ),
    rx(
        r"^(?:this|that|these|those|ceci|cela|cette\s+condition|"
        r"these\s+conditions)$"
    ),
    rx(
        r"^(?:treatment|management|diagnosis|traitement|"
        r"prise\s+en\s+charge|diagnostic)$"
    )
)


# ## Registre

MEDICAL_KNOWLEDGE_PATTERNS = {
    "conditions": CONDITION_PATTERNS,
    "symptoms": SYMPTOM_PATTERNS,
    "medications": MEDICATION_PATTERNS,
    "treatments": TREATMENT_PATTERNS,
    "procedures": PROCEDURE_PATTERNS,
    "laboratory_findings": LAB_PATTERNS,
    "imaging_findings": IMAGING_PATTERNS,
    "diagnostic_tests": DIAGNOSTIC_TEST_PATTERNS,
    "complications": COMPLICATION_PATTERNS,
    "risk_factors": RISK_FACTOR_PATTERNS,
    "pathogens": PATHOGEN_PATTERNS,
    "causes": CAUSE_PATTERNS
}