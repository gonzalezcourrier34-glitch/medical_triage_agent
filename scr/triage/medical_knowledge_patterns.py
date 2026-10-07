# scr/triage/medical_knowledge_patterns.py

import re


# ## Utilitaire

def rx(pattern: str) -> re.Pattern:
    """Compile une regex clinique."""
    return re.compile(pattern, re.IGNORECASE)


# ## Pathologies

CONDITION_PATTERNS = (
    # Cas patient FR
    rx(
        r"\b(?:diagnostic\s+de|diagnostiqué(?:e)?\s+(?:avec|pour)|"
        r"atteint(?:e)?\s+de|souffre\s+de|maladie\s+de|"
        r"antécédents?\s+de|connu(?:e)?\s+pour)\s+"
        r"([^.;:?!\n]{3,140})"
    ),

    # Cas patient EN
    rx(
        r"\b(?:diagnosed\s+with|diagnosis\s+of|suffers?\s+from|"
        r"history\s+of|known\s+for|known\s+with|medical\s+history\s+of)\s+"
        r"([^.;:?!\n]{3,140})"
    ),

    rx(
        r"\b(?:has|had)\s+(?:a|an)?\s*"
        r"((?:acute|chronic|metastatic|recurrent|severe|advanced|"
        r"stage\s+[ivx0-9]+|type\s+[12])\s+[^.;:?!\n]{2,100})"
    ),

    # Description documentaire EN
    rx(
        r"\b([A-Z][^.;:?!\n]{2,100}?)\s+"
        r"(?:is|are)\s+(?:a|an)\s+"
        r"(?:rare\s+)?(?:genetic\s+|inherited\s+|chronic\s+|acute\s+)?"
        r"(?:disorder|disease|condition|syndrome)\b"
    ),

    rx(
        r"\b(?:condition|disease|disorder|syndrome)\s+"
        r"(?:called|known\s+as)\s+([^.;:?!\n]{3,100})"
    )
)


# ## Symptômes

SYMPTOM_PATTERNS = (
    # Formes structurées
    rx(
        r"\b(?:sympt[oô]mes?|signes?\s+cliniques?|"
        r"manifestations?\s+cliniques?)\s*[:\-]\s*"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:symptoms?|clinical\s+signs?|clinical\s+manifestations?)"
        r"\s*[:\-]\s*([^.;?!\n]{3,180})"
    ),

    # Patient FR / EN
    rx(
        r"\b(?:présente|présentant|se\s+plaint\s+de)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:presents?|presented)\s+with\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:complains?\s+of|experiencing|experienced)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    # Description documentaire
    rx(
        r"\b(?:symptoms?|signs?|manifestations?)\s+"
        r"(?:include|includes|may\s+include|can\s+include|"
        r"often\s+include|typically\s+include|are)\s+"
        r"([^.;?!\n]{3,200})"
    ),

    rx(
        r"\b(?:common|typical|possible|early|initial|major)\s+"
        r"(?:symptoms?|signs?)\s+(?:include|are)\s+"
        r"([^.;?!\n]{3,200})"
    ),

    rx(
        r"\b(?:patients?|people|individuals|children|adults)\s+"
        r"(?:with\s+[^.;?!\n]{2,100}\s+)?"
        r"(?:may|can|often|commonly|typically)\s+"
        r"(?:have|develop|experience|show|present\s+with)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:affected\s+(?:people|individuals|patients|children))\s+"
        r"(?:may|can|often|commonly|typically)?\s*"
        r"(?:have|develop|experience|show)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    # Association
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
    # Patient FR
    rx(
        r"\b(?:traité(?:e)?\s+par|traitement\s+par|reçoit|recevant|"
        r"administration\s+de|prescription\s+de|prend|sous)\s+"
        r"([^.;:?!\n]{2,120})"
    ),

    # Patient EN
    rx(
        r"\b(?:treated\s+with|receives?|receiving|administered|"
        r"prescribed|taking|started\s+on|maintained\s+on|"
        r"currently\s+on)\s+"
        r"([^.;:?!\n]{2,120})"
    ),

    rx(
        r"\bon\s+([^.;:?!\n]{2,80})\s+"
        r"(?:therapy|treatment|medication|medications)\b"
    ),

    # Description documentaire
    rx(
        r"\b(?:medications?|drugs?)\s+(?:such\s+as|include|includes)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:may|can)\s+be\s+(?:treated|managed)\s+with\s+"
        r"([^.;?!\n]{3,160})"
    )
)


# ## Traitements

TREATMENT_PATTERNS = (
    # FR
    rx(
        r"\b(?:traitement|prise\s+en\s+charge|thérapie|therapie)\s*"
        r"(?:recommandé(?:e)?|indiqué(?:e)?|de\s+choix|consiste\s+en|"
        r"comprend|inclut|:)?\s*"
        r"([^.;?!\n]{3,160})"
    ),

    # EN structuré
    rx(
        r"\b(?:treatment|management|therapy)\s*"
        r"(?:recommended|indicated|of\s+choice|consists?\s+of|"
        r"includes?|may\s+include|can\s+include|:)\s*"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:managed\s+with|management\s+with)\s+"
        r"([^.;?!\n]{3,140})"
    ),

    # Texte documentaire
    rx(
        r"\b(?:treatments?|therapies)\s+"
        r"(?:include|includes|may\s+include|can\s+include)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:treatment|management)\s+(?:usually|typically|often)?\s*"
        r"(?:involves?|requires?)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:treated|managed)\s+(?:by|with|using)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:there\s+is\s+)?no\s+(?:specific\s+)?"
        r"(?:treatment|therapy|cure)\b"
        r"([^.;?!\n]{0,120})"
    )
)


# ## Procédures

PROCEDURE_PATTERNS = (
    # FR
    rx(
        r"\b(?:intervention|procédure|procedure|chirurgie|opération|operation)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),

    # EN
    rx(
        r"\b(?:surgery|procedure|operation|intervention)\s*[:\-]?\s*"
        r"([^.;?!\n]{3,140})"
    ),

    rx(
        r"\b(?:underwent|undergoing|scheduled\s+for)\s+"
        r"([^.;?!\n]{3,140})"
    ),

    rx(
        r"\b(?:a\s+subi|doit\s+subir|prévu(?:e)?\s+pour)\s+"
        r"([^.;?!\n]{3,140})"
    ),

    # Documentaire EN
    rx(
        r"\b(?:procedures?|surgical\s+procedures?)\s+"
        r"(?:include|includes|may\s+include)\s+"
        r"([^.;?!\n]{3,180})"
    )
)


# ## Biologie

LAB_PATTERNS = (
    # FR
    rx(
        r"\b(?:biologie|bilan\s+biologique|analyse\s+de\s+sang|"
        r"laboratoire|résultats?\s+biologiques?)\s*[:\-]\s*"
        r"([^.;?!\n]{3,220})"
    ),

    # EN
    rx(
        r"\b(?:laboratory|lab\s+results?|blood\s+tests?|"
        r"laboratory\s+findings?)\s*[:\-]\s*"
        r"([^.;?!\n]{3,220})"
    ),

    rx(
        r"\b(?:blood\s+tests?|laboratory\s+tests?)\s+"
        r"(?:show|shows|showed|reveal|reveals|may\s+show)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:shows?|showed|reveals?|revealed|demonstrates?)\s+"
        r"((?:elevated|decreased|low|high|increased|reduced|abnormal)\s+"
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
    # FR
    rx(
        r"\b(?:radiographie|scanner|TDM|IRM|échographie|echographie|imagerie)"
        r"\s*(?:montre|révèle|revele|objective|:|-)?\s*"
        r"([^.;?!\n]{3,180})"
    ),

    # EN
    rx(
        r"\b(?:x[- ]?ray|CT(?:\s+scan)?|MRI|ultrasound|imaging)"
        r"\s*(?:shows?|showed|reveals?|revealed|demonstrates?|"
        r"may\s+show|can\s+show|:|-)\s*"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:imaging|x[- ]?rays?|CT(?:\s+scans?)?|MRI|ultrasound)"
        r"\s+(?:may|can|often)?\s*"
        r"(?:show|reveal|demonstrate)\s+"
        r"([^.;?!\n]{3,180})"
    )
)


# ## Tests diagnostiques

DIAGNOSTIC_TEST_PATTERNS = (
    # FR
    rx(
        r"\b(?:test|examen|bilan)\s+"
        r"(?:diagnostique|diagnostic|de\s+confirmation|confirmatoire)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),

    rx(
        r"\b(?:confirmé(?:e)?\s+par|diagnostiqué(?:e)?\s+par)\s+"
        r"([^.;?!\n]{3,120})"
    ),

    # EN
    rx(
        r"\b(?:diagnostic\s+test|confirmatory\s+test|workup)"
        r"\s*[:\-]?\s*([^.;?!\n]{3,140})"
    ),

    rx(
        r"\b(?:confirmed\s+by|diagnosed\s+by|diagnosed\s+with)\s+"
        r"([^.;?!\n]{3,140})"
    ),

    # Documentaire EN
    rx(
        r"\b(?:diagnosis|diagnosing\s+[^.;?!\n]{2,80})\s+"
        r"(?:involves?|requires?|includes?|is\s+based\s+on)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:tests?|testing|examinations?)\s+"
        r"(?:used\s+to|can\s+help|may\s+help)\s+"
        r"(?:diagnose|confirm|detect)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:blood\s+tests?|genetic\s+testing|biopsy|"
        r"electrocardiogram|ECG|EKG|CT(?:\s+scan)?|MRI|ultrasound)\s+"
        r"(?:may|can|will|is|are)\s+"
        r"(?:be\s+)?(?:used|performed|done|ordered)\b"
        r"([^.;?!\n]{0,120})"
    )
)


# ## Complications

COMPLICATION_PATTERNS = (
    # FR
    rx(
        r"\b(?:complication(?:s)?|compliqué(?:e)?\s+de|évolue\s+vers|"
        r"ayant\s+entraîné|entraînant)\s+"
        r"([^.;:?!\n]{3,140})"
    ),

    # EN
    rx(
        r"\b(?:complication(?:s)?|complicated\s+by|progresses?\s+to|"
        r"resulting\s+in|leading\s+to)\s+"
        r"([^.;:?!\n]{3,160})"
    ),

    rx(
        r"\b(?:developed|develops)\s+"
        r"((?:acute|severe|progressive|new(?:ly)?[- ]onset)\s+"
        r"[^.;:?!\n]{2,120})"
    ),

    # Documentaire EN
    rx(
        r"\b(?:complications?|possible\s+complications?)\s+"
        r"(?:include|includes|may\s+include|can\s+include|are)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:may|can)\s+(?:lead\s+to|result\s+in|cause)\s+"
        r"([^.;?!\n]{3,160})"
    )
)


# ## Facteurs de risque

RISK_FACTOR_PATTERNS = (
    # FR
    rx(
        r"\b(?:facteurs?\s+de\s+risque|terrain\s+à\s+risque|"
        r"risque\s+accru\s+lié\s+à)\s*[:\-]?\s*"
        r"([^.;?!\n]{3,180})"
    ),

    # EN
    rx(
        r"\b(?:risk\s+factors?|high[- ]risk\s+features?|"
        r"increased\s+risk\s+due\s+to)\s*[:\-]?\s*"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:risk\s+factors?)\s+"
        r"(?:include|includes|may\s+include|are)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:people|patients|individuals)\s+(?:who|with)\s+"
        r"([^.;?!\n]{3,140})\s+"
        r"(?:are|have)\s+(?:at\s+)?(?:an?\s+)?"
        r"(?:increased|higher|greater)?\s*risk\b"
    ),

    rx(
        r"\b(?:increases?|raises?)\s+(?:the\s+)?risk\s+of\s+"
        r"([^.;?!\n]{3,140})"
    )
)


# ## Pathogènes

PATHOGEN_PATTERNS = (
    # FR
    rx(
        r"\b(?:infection\s+(?:par|à)|causé(?:e)?\s+par|"
        r"germe\s+responsable|agent\s+responsable)\s+"
        r"([^.;:?!\n]{2,100})"
    ),

    # EN
    rx(
        r"\b(?:infection\s+(?:with|by)|caused\s+by|"
        r"causative\s+(?:organism|pathogen|agent))\s+"
        r"([^.;:?!\n]{2,120})"
    ),

    rx(
        r"\b(?:bacterium|bacteria|virus|fungus|fungi|parasite)\s+"
        r"(?:called|known\s+as)\s+"
        r"([^.;:?!\n]{2,100})"
    )
)


# ## Causes

CAUSE_PATTERNS = (
    # EN
    rx(
        r"\b(?:due\s+to|secondary\s+to|caused\s+by|attributed\s+to)\s+"
        r"([^.;:?!\n]{3,160})"
    ),

    # FR
    rx(
        r"\b(?:dû(?:e)?\s+à|du(?:e)?\s+à|secondaire\s+à|"
        r"causé(?:e)?\s+par|attribué(?:e)?\s+à)\s+"
        r"([^.;:?!\n]{3,160})"
    ),

    # Documentaire EN
    rx(
        r"\b(?:cause|causes)\s+(?:include|includes|may\s+include|are)\s+"
        r"([^.;?!\n]{3,180})"
    ),

    rx(
        r"\b(?:results?\s+from|arises?\s+from)\s+"
        r"([^.;?!\n]{3,160})"
    ),

    rx(
        r"\b(?:mutation|mutations|changes?)\s+in\s+"
        r"([^.;?!\n]{3,120})"
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
        "pharynx", "trachea", "bronchi", "artery", "vein", "spinal cord",
        "bone", "bones", "cartilage", "blood", "nervous system",
        "white matter", "nerve", "nerves"
    }
}


# ## Nettoyage

# Coupe une capture lorsqu'elle bascule vers une nouvelle question.
QUESTION_TAIL_PATTERN = rx(
    r"\s*,?\s*\b(?:which\s+of\s+the\s+following|which|what|who|why|how|"
    r"quel(?:le)?s?|lequel|laquelle|parmi\s+les|quelle\s+est)\b.*$"
)


# Élimine les captures éditoriales sans valeur médicale exploitable.
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

# ## Intent question

QUESTION_INTENT_PATTERNS = (
    ("symptoms", rx(r"\b(?:signs and symptoms|symptoms?|signs?|manifestations?|sympt[oô]mes?|signes?|manifestations?)\b")),
    ("treatment", rx(r"\b(?:treatments?|what to do for|treated|therapy|therapies|traitements?|prise en charge|th[ée]rapies?)\b")),
    ("diagnosis", rx(r"\b(?:diagnos(?:e|ed|is|ing|tic)|tests?|testing|diagnosti(?:c|quer|qu[ée])|examens?|tests?)\b")),
    ("complications", rx(r"\b(?:complications?|complicated|compliqu[ée]e?s?)\b")),
    ("risk", rx(r"\b(?:risk factors?|at risk|facteurs? de risques?|[àa] risque)\b")),
    ("causes", rx(r"\b(?:causes?|caused by|etiology|causes?|caus[ée]e? par|[ée]tiologie)\b")),
    ("prognosis", rx(r"\b(?:prognosis|outlook|survival|pronostic|survie|[ée]volution)\b")),
    ("prevalence", rx(r"\b(?:how many people|how common|prevalence|frequency|combien de personnes|pr[ée]valence|fr[ée]quence)\b")),
    ("inheritance", rx(r"\b(?:genetic|inherited|inheritance|hereditary|g[ée]n[ée]tique|h[ée]r[ée]ditaire|h[ée]r[ée]dit[ée])\b")),
    ("research", rx(r"\b(?:research|clinical trials?|recherche|essais? cliniques?)\b")),
    ("prevention", rx(r"\b(?:prevent|prevents|prevented|preventing|prevention|preventable|pr[ée]venir|pr[ée]vention|[ée]viter)\b")),
    ("information", rx(r"\b(?:information about|informations? (?:sur|concernant|[àa] propos de))\b")),
    ("definition", rx(r"^(?:what (?:is|are)(?:\s+\(are\))?|qu['’]est[- ]ce que|quels? sont|quelles? sont)\b"))
)


# ## Sujet question

QUESTION_SUBJECT_PATTERNS = (
    rx(r"^what research \(or clinical trials\) is being done for (.+?)\s*\?$"),
    rx(r"^what are the (?:signs and )?symptoms (?:of|for) (.+?)\s*\?$"),
    rx(r"^what are the treatments for (.+?)\s*\?$"),
    rx(r"^what are the complications of (.+?)\s*\?$"),
    rx(r"^what are the genetic changes related to (.+?)\s*\?$"),
    rx(r"^what are the stages of (.+?)\s*\?$"),
    rx(r"^who is at risk for (.+?)\??\s*\?$"),
    rx(r"^how many people are affected by (.+?)\s*\?$"),
    rx(r"^what is the outlook for (.+?)\s*\?$"),
    rx(r"^how common is (.+?)\s*\?$"),
    rx(r"^how is hps diagnosed and treated for (.+?)\s*\?$"),
    rx(r"^how can the spread of (.+?) be prevented\s*\?$"),
    rx(r"^how can (.+?) be (?:treated|prevented)\s*\?$"),
    rx(r"^how to (?:diagnose|prevent) (.+?)\s*\?$"),
    rx(r"^what to do for (.+?)\s*\?$"),
    rx(r"^what causes (.+?)\s*\?$"),
    rx(r"^are there complications from (.+?)\s*\?$"),
    rx(r"^is (.+?) inherited\s*\?$"),
    rx(r"^do you have information about (.+?)\s*\??$"),
    rx(r"^what is \(are\) (.+?)\s*\?$"),
    rx(r"^what is (.+?)\s*\?$"),

    rx(r"^quels? sont les (?:principaux )?(?:signes|sympt[oô]mes) (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^quels? sont les (?:principaux )?facteurs? de risques? (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^quelles? sont les complications (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^quels? sont les traitements? (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^comment diagnostiquer (.+?)\s*\?$"),
    rx(r"^comment pr[ée]venir (.+?)\s*\?$"),
    rx(r"^quelles? sont les causes (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^quel est le pronostic (?:de|du|des|d['’])\s*(.+?)\s*\?$"),
    rx(r"^qu['’]est[- ]ce que (.+?)\s*\?$")
)