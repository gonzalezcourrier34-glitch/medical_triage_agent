import re


# ## Utilitaire

def rx(pattern: str) -> re.Pattern:
    """Compile une regex clinique."""
    return re.compile(pattern, re.IGNORECASE)


# ## Sources

PATIENT_TEXT_FIELDS = ("context", "question")


# ## Âge

AGE_DAY_PATTERNS = (
    rx(
        r"\b(?:nouveau[- ]né(?:e)?|nourrisson|bébé|bebe|patient(?:e)?|garçon|fille)"
        r"(?:\s+\w+){0,5}\s+âgé(?:e)?\s+de\s+(\d{1,2})\s*jours?\b"
    ),
    rx(
        r"\b(?:nouveau[- ]né(?:e)?|nourrisson|bébé|bebe|(?:petit\s+)?garçon|"
        r"(?:petite\s+)?fille)\s+de\s+(\d{1,2})\s*jours?\b"
    ),
    rx(r"\bâge\s*[:=]?\s*(\d{1,2})\s*jours?\b"),
    rx(r"\b(\d{1,2})[\s-]*days?[\s-]*old\b"),
    rx(
        r"\b(?:newborn|infant|baby|boy|girl|patient)"
        r"(?:\s+\w+){0,5}\s+aged\s+(\d{1,2})\s*days?\b"
    )
)

AGE_MONTH_PATTERNS = (
    rx(
        r"\b(?:nourrisson|bébé|bebe|enfant|patient(?:e)?|garçon|fille)"
        r"(?:\s+\w+){0,5}\s+âgé(?:e)?\s+de\s+(\d{1,2})\s*mois\b"
    ),
    rx(
        r"\b(?:nourrisson|bébé|bebe|(?:jeune\s+)?enfant|(?:petit\s+)?garçon|"
        r"(?:petite\s+)?fille)\s+de\s+(\d{1,2})\s*mois\b"
    ),
    rx(r"\bâge\s*[:=]?\s*(\d{1,2})\s*mois\b"),
    rx(r"\b(\d{1,2})[\s-]*months?[\s-]*old\b"),
    rx(
        r"\b(?:infant|baby|child|boy|girl|patient)"
        r"(?:\s+\w+){0,5}\s+aged\s+(\d{1,2})\s*months?\b"
    )
)

AGE_YEAR_PATTERNS = (
    rx(
        r"\b(?:jeune\s+)?(?:patient(?:e)?|homme|femme|garçon|fille|enfant|"
        r"adolescent(?:e)?|blessé(?:e)?|malade|sujet)\s+de\s+(\d{1,3})\s*ans?\b"
    ),
    rx(
        r"\b(?:patient(?:e)?|homme|femme|garçon|fille|enfant|adolescent(?:e)?|"
        r"blessé(?:e)?|malade|sujet)(?:\s+\w+){0,5}\s+âgé(?:e)?\s+de\s+"
        r"(\d{1,3})\s*ans?\b"
    ),
    rx(r"\bâge\s*[:=]?\s*(\d{1,3})\s*ans?\b"),
    rx(r"\[NAME\]\s*,?\s*(?:âgé(?:e)?\s+de\s+)?(\d{1,3})\s*ans?\b"),
    rx(
        r"\b(\d{1,3})[\s-]*year[\s-]*old\s+"
        r"(?:man|woman|boy|girl|male|female|patient)\b"
    ),
    rx(r"\b(\d{1,3})[\s-]*years?[\s-]*old\b"),
    rx(
        r"\b(?:man|woman|boy|girl|male|female|patient)"
        r"(?:\s+\w+){0,5}\s+aged\s+(\d{1,3})\b"
    ),
    rx(r"\b(\d{1,3})\s*(?:yo|y/o)\b")
)

FRENCH_WORD_AGE_PATTERNS = (
    rx(
        r"\b(?:jeune\s+)?(?:patient(?:e)?|homme|femme|garçon|fille|enfant|"
        r"adolescent(?:e)?|blessé(?:e)?|malade)\s+de\s+"
        r"([a-zàâçéèêëîïôûùüÿ-]+(?:\s+(?:et\s+)?"
        r"[a-zàâçéèêëîïôûùüÿ-]+){0,3})\s+ans\b"
    ),
    rx(
        r"\b(?:patient(?:e)?|homme|femme|garçon|fille|enfant|adolescent(?:e)?|"
        r"blessé(?:e)?|malade)(?:\s+\w+){0,5}\s+âgé(?:e)?\s+de\s+"
        r"([a-zàâçéèêëîïôûùüÿ-]+(?:\s+(?:et\s+)?"
        r"[a-zàâçéèêëîïôûùüÿ-]+){0,3})\s+ans\b"
    )
)

FRENCH_NUMBER_UNITS = {
    "zéro": 0, "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3,
    "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
    "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
    "quinze": 15, "seize": 16, "dix-sept": 17, "dix-huit": 18,
    "dix-neuf": 19
}

FRENCH_NUMBER_TENS = {
    "vingt": 20, "trente": 30, "quarante": 40,
    "cinquante": 50, "soixante": 60
}


# ## Sexe

PATIENT_SEX_PATTERNS = {
    "male": (
        rx(
            r"\b(?:homme|garçon|petit\s+garçon|blessé|nouveau[- ]né|"
            r"patient\s+masculin|sexe\s+masculin|male\s+patient|man|boy|male)\b"
        ),
    ),
    "female": (
        rx(
            r"\b(?:femme|fille|petite\s+fille|patiente|blessée|nouvelle[- ]née|"
            r"patient\s+féminin|sexe\s+féminin|female\s+patient|woman|girl|female)\b"
        ),
    )
}


# ## Pathologies actuelles

CONDITION_MENTION_PATTERNS = (
    rx(
        r"\b(?:diagnostic\s+de|diagnostiqué(?:e)?\s+(?:avec|pour)|"
        r"atteint(?:e)?\s+de|présente\s+une?|présentant\s+une?|"
        r"connu(?:e)?\s+pour)\s+([^.;:?!\n]{3,140})"
    ),
    rx(
        r"\b(?:diagnosed\s+with|known\s+with|known\s+for|has\s+(?:a|an)?)\s+"
        r"([^.;:?!\n]{3,140})"
    )
)

CONDITION_MENTION_EXCLUSION_PATTERNS = (
    rx(
        r"^(?:quel(?:le)?s?|lequel|laquelle|lesquels|lesquelles|"
        r"which|what|who|why|how)\b"
    ),
    rx(
        r"^(?:traitement|prise\s+en\s+charge|conduite\s+à\s+tenir|"
        r"treatment|management)\b"
    )
)


# ## Constantes

TEMPERATURE_PATTERNS = (
    rx(
        r"\b(?:température|temperature|temp|t°|body\s+temperature)\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{2,3}(?:\s*[.,]\s*\d+)?)\s*°?\s*(c|f|celsius|fahrenheit)?\b"
    ),
    rx(r"\b(\d{2}(?:\s*[.,]\s*\d+)?)\s*°\s*(c|f|celsius|fahrenheit)\b")
)

HEART_RATE_PATTERNS = (
    rx(
        r"\b(?:fc|hr|fréquence\s+cardiaque|frequence\s+cardiaque|heart\s+rate|"
        r"pulse|pouls)\s*(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{1,3}(?:[.,]\d+)?)\s*(?:/min|/mn|bpm|beats?\s+per\s+minute|"
        r"battements?\s+par\s+minute)?\b"
    ),
    rx(
        r"\b(\d{1,3}(?:[.,]\d+)?)\s*"
        r"(?:bpm|beats?\s+per\s+minute|battements?\s+par\s+minute)\b"
    )
)

BLOOD_PRESSURE_MMHG_PATTERNS = (
    rx(
        r"\b(?:ta|pa|bp|pression\s+artérielle|pression\s+arterielle|"
        r"tension\s+artérielle|tension\s+arterielle|blood\s+pressure)"
        r"\s*(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{2,3})\s*[/\-]\s*(\d{2,3})\s*(?:mm\s*hg|mmhg)?\b"
    ),
    rx(
        r"\b(?:systolique|systolic)\s*(?:blood\s+pressure|pressure)?\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*(\d{2,3})"
        r".{0,40}?\b(?:diastolique|diastolic)\s*(?:blood\s+pressure|pressure)?\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*(\d{2,3})\b"
    )
)

BLOOD_PRESSURE_CMHG_PATTERNS = (
    rx(
        r"\b(?:ta|pa|bp|pression\s+artérielle|pression\s+arterielle|"
        r"tension\s+artérielle|tension\s+arterielle|blood\s+pressure)"
        r"\s*(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{1,2}(?:[.,]\d+)?)\s*[/\-]\s*(\d{1,2}(?:[.,]\d+)?)\s*"
        r"(?:cm\s*hg|cmhg)\b"
    ),
)

SYSTOLIC_BP_PATTERNS = (
    rx(
        r"\b(?:pression\s+artérielle\s+systolique|pression\s+arterielle\s+systolique|"
        r"tension\s+systolique|pas|systolic\s+blood\s+pressure|"
        r"systolic\s+pressure|systolic\s+bp)\s*(?:est|is)?\s*"
        r"(?:à|a|de|of|at|:|=)?\s*(\d{2,3}(?:[.,]\d+)?)\s*"
        r"(?:mm\s*hg|mmhg)?\b"
    ),
)

DIASTOLIC_BP_PATTERNS = (
    rx(
        r"\b(?:pression\s+artérielle\s+diastolique|pression\s+arterielle\s+diastolique|"
        r"tension\s+diastolique|pad|diastolic\s+blood\s+pressure|"
        r"diastolic\s+pressure|diastolic\s+bp)\s*(?:est|is)?\s*"
        r"(?:à|a|de|of|at|:|=)?\s*(\d{2,3}(?:[.,]\d+)?)\s*"
        r"(?:mm\s*hg|mmhg)?\b"
    ),
)

SPO2_PATTERNS = (
    rx(
        r"\b(?:spo2|spo₂|sao2|sat(?:uration)?(?:\s+(?:en\s+)?o2)?|"
        r"oxygen\s+saturation|oxygen\s+sat|o2\s+sat(?:uration)?)\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{2,3}(?:[.,]\d+)?)\s*%?\b"
    ),
)

RESPIRATORY_RATE_PATTERNS = (
    rx(
        r"\b(?:fr|rr|respiratory\s+rate|respiration\s+rate|breathing\s+rate|"
        r"fréquence\s+respiratoire|frequence\s+respiratoire)\s*(?:est|is)?\s*"
        r"(?:à|a|de|of|at|:|=)?\s*(\d{1,3}(?:[.,]\d+)?)\s*"
        r"(?:/min|/mn|rpm|breaths?\s+per\s+minute|breaths?/min)?\b"
    ),
)

GCS_PATTERNS = (
    rx(
        r"\b(?:gcs|glasgow(?:\s+coma\s+scale)?|glasgow\s+coma\s+score|"
        r"score\s+de\s+glasgow)\s*(?:score\s*)?(?:est|is)?\s*"
        r"(?:à|a|de|of|at|:|=)?\s*(\d{1,2})(?:\s*/\s*15)?\b"
    ),
)

BLOOD_GLUCOSE_PATTERNS = (
    rx(
        r"\b(?:glycémie|glycemie|blood\s+glucose|blood\s+sugar|glucose)\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{1,2}(?:[.,]\d+)?)\s*mmol\s*/?\s*l\b"
    ),
)

KETONE_PATTERNS = (
    rx(
        r"\b(?:cétonémie|cetonemie|cétones?|cetones?|ketones?|ketonemia)\s*"
        r"(?:est|is)?\s*(?:à|a|de|of|at|:|=)?\s*"
        r"(\d{1,2}(?:[.,]\d+)?)\s*mmol\s*/?\s*l\b"
    ),
)

PAIN_SCORE_PATTERNS = (
    rx(
        r"\b(?:douleur|pain)(?:\s+(?:cotée|cotee|évaluée|evaluee|score(?:d)?))?\s*"
        r"(?:à|a|de|at|:|=)?\s*(\d{1,2}(?:[.,]\d+)?)\s*/\s*10\b"
    ),
    rx(r"\b(?:eva|evs|nrs)\s*(?:à|a|:|=)?\s*(\d{1,2}(?:[.,]\d+)?)\b")
)


# ## Symptômes

SYMPTOM_SECTION_PATTERNS = (
    rx(
        r"\b(?:sympt[oô]mes?|signes?\s+fonctionnels?|"
        r"motif(?:\s+de\s+consultation)?)\s*[:\-]\s*([^\n]+)"
    ),
    rx(
        r"\b(?:symptoms?|presenting\s+complaint|chief\s+complaint)\s*[:\-]\s*([^\n]+)"
    )
)

SYMPTOM_NARRATIVE_PATTERNS = (
    rx(r"\b(?:vient\s+)?consult(?:e|er|ait)\s+pour\s+([^.;:?!]{3,160})"),
    rx(r"\bse\s+plaint\s+(?:de|du|des|d[’'])\s*([^.;:?!]{3,160})"),
    rx(
        r"\b(?:est\s+)?(?:amené|amenée|admis|admise|hospitalisé|hospitalisée|"
        r"adressé|adressée)\s+(?:aux\s+urgences\s+)?pour\s+([^.;:?!]{3,160})"
    ),
    rx(
        r"\b(?:patient(?:e)?|malade|blessé(?:e)?|homme|femme|garçon|fille|"
        r"enfant|nourrisson)\b[^.;:?!]{0,80}\b(?:présente|présentant|ayant|avec)\s+"
        r"([^.;:?!]{3,160})"
    ),
    rx(
        r"\b(?:patient|man|woman|boy|girl|child|infant)\b[^.;:?!]{0,80}\b"
        r"(?:presents?|presented|with|experiencing|having)\s+(?:with\s+)?"
        r"([^.;:?!]{3,160})"
    ),
    rx(
        r"\b(?:presents?|presented|comes?|came)\s+"
        r"(?:to\s+(?:the\s+)?(?:ED|emergency\s+department)\s+)?"
        r"(?:with|for)\s+([^.;:?!]{3,160})"
    ),
    rx(r"\bcomplains?\s+of\s+([^.;:?!]{3,160})")
)

SYMPTOM_EXCLUSION_PATTERNS = (
    rx(
        r"^(?:la|le|les|un|une)?\s*(?:prise\s+en\s+charge|traitement|diagnostic|"
        r"conduite\s+à\s+tenir|indication|contre-indication)\b"
    ),
    rx(
        r"^(?:quel(?:le)?s?|laquelle|lesquelles|lequel|lesquels|parmi|"
        r"which|what|who|why|how)\b"
    )
)

CLINICAL_ITEM_SPLIT = rx(r"\s*(?:,|;|\bet\b|\band\b)\s*")


# ## Antécédents

MEDICAL_HISTORY_PATTERNS = (
    rx(r"\b(?:antécédents?(?:\s+médicaux)?|atcd)\s*[:\-]\s*([^\n]+)"),
    rx(
        r"\b(?:medical\s+history|past\s+medical\s+history|pmh)\s*[:\-]\s*([^\n]+)"
    )
)

MEDICAL_HISTORY_NARRATIVE_PATTERNS = (
    rx(r"\bantécédents?\s+(?:de|notamment)\s+([^.;:?!]{3,180})"),
    rx(
        r"\b(?:est|était)\s+(?:suivi|suivie|traité|traitée)\s+pour\s+"
        r"([^.;:?!]{3,180})"
    ),
    rx(r"\b(?:history\s+of|known\s+history\s+of)\s+([^.;:?!]{3,180})")
)

MEDICAL_HISTORY_NEGATION_PATTERNS = (
    rx(
        r"\b(?:sans|aucun|aucune|absence\s+d[’']?)\s*"
        r"antécédents?(?:\s+médicaux)?\b"
    ),
    rx(r"\b(?:sans\s+antécédent\s+notable|aucun\s+antécédent\s+notable)\b"),
    rx(r"\b(?:no|without)\s+(?:significant\s+)?(?:medical\s+)?history\b")
)


# ## Durée

DURATION_PATTERNS = (
    rx(
        r"\b(?:consulte|vient\s+consulter|se\s+plaint|présente)"
        r"[^.;:?!]{0,140}\bdepuis\s+"
        r"(\d+(?:[.,]\d+)?\s*(?:minutes?|heures?|jours?|semaines?|mois|ans?))\b"
    ),
    rx(
        r"\b(?:depuis|évoluant\s+depuis|apparu(?:e)?\s+il\s+y\s+a)\s+"
        r"(\d+(?:[.,]\d+)?\s*(?:minutes?|heures?|jours?|semaines?|mois|ans?))\b"
    ),
    rx(
        r"\b(?:for|for\s+the\s+past|since)\s+"
        r"(\d+(?:[.,]\d+)?\s*(?:minutes?|hours?|days?|weeks?|months?|years?))\b"
    )
)


# ## Évolution

EVOLUTION_PATTERNS = {
    "worsening": (
        rx(
            r"\b(?:sympt[oô]mes?|douleur|état|tableau)[^.;:?!]{0,80}"
            r"(?:s[’']aggrave|s[’']est\s+aggravé|se\s+majore|aggravation)\b"
        ),
        rx(
            r"\b(?:getting\s+worse|progressively\s+worse|"
            r"condition\s+worsening|symptoms?\s+worsening)\b"
        )
    ),
    "improving": (
        rx(
            r"\b(?:sympt[oô]mes?|douleur|état|tableau)[^.;:?!]{0,80}"
            r"(?:s[’']améliore|s[’']est\s+amélioré|régression|amélioration)\b"
        ),
        rx(
            r"\b(?:getting\s+better|condition\s+improving|symptoms?\s+improving)\b"
        )
    ),
    "stable": (
        rx(
            r"\b(?:état|symptômes?|tableau)\s+(?:reste\s+)?"
            r"(?:stable|stationnaire|inchangé)\b"
        ),
        rx(
            r"\b(?:condition|symptoms?)\s+(?:remains?\s+)?(?:stable|unchanged)\b"
        )
    ),
    "sudden": (
        rx(r"\b(?:début|apparition|survenue)\s+(?:brutal(?:e)?|soudain(?:e)?)\b"),
        rx(r"\b(?:sudden|abrupt)\s+onset\b")
    ),
    "progressive": (
        rx(r"\b(?:installation|apparition|évolution)\s+progressive(?:ment)?\b"),
        rx(r"\bgradual(?:ly)?\s+(?:onset|progression)\b")
    )
}


# ## Signes critiques

CRITICAL_SIGN_PATTERNS = {
    "cardiac_arrest": (
        rx(r"\b(?:arrêt\s+cardio(?:-| )?respiratoire|arrêt\s+cardiaque|acr)\b"),
        rx(r"\b(?:cardiac|cardiorespiratory|cardiopulmonary)\s+arrest\b")
    ),
    "unconsciousness": (
        rx(
            r"\b(?:inconscient(?:e)?|perte\s+de\s+connaissance|"
            r"coma(?:teux|teuse)?|état\s+comateux)\b"
        ),
        rx(r"\b(?:unconscious|loss\s+of\s+consciousness|comatose)\b")
    ),
    "altered_consciousness": (
        rx(
            r"\b(?:altération\s+de\s+la\s+conscience|conscience\s+altérée|"
            r"troubles?\s+de\s+la\s+conscience|troubles?\s+de\s+vigilance|"
            r"baisse\s+de\s+vigilance|stuporeux|stuporeuse|somnolent(?:e)?|"
            r"somnolence|obnubilation|obnubilé(?:e)?)\b"
        ),
        rx(
            r"\b(?:altered\s+(?:level\s+of\s+)?consciousness|"
            r"decreased\s+(?:level\s+of\s+)?consciousness|"
            r"impaired\s+consciousness|reduced\s+alertness|stuporous|drowsy|obtunded)\b"
        )
    ),
    "confusion": (
        rx(r"\b(?:confus(?:e)?|confusion|désorientation|désorienté(?:e)?)\b"),
        rx(r"\b(?:confused|confusion|disoriented|disorientation)\b")
    ),
    "severe_respiratory_distress": (
        rx(
            r"\b(?:détresse\s+respiratoire|insuffisance\s+respiratoire\s+aiguë|"
            r"dyspnée\s+(?:majeure|sévère|intense)|"
            r"signes?\s+de\s+lutte\s+respiratoire)\b"
        ),
        rx(
            r"\b(?:severe\s+respiratory\s+distress|acute\s+respiratory\s+failure|"
            r"severe\s+breathing\s+difficulty)\b"
        )
    ),
    "cyanosis": (
        rx(r"\b(?:cyanose|cyanosé(?:e)?|cyanose\s+centrale)\b"),
        rx(r"\b(?:cyanosis|cyanotic|central\s+cyanosis)\b")
    ),
    "retractions": (
        rx(
            r"\b(?:tirage(?:\s+(?:intercostal|sus[- ]sternal|sous[- ]costal))?|"
            r"balancement\s+thoraco[- ]abdominal|battements?\s+des\s+ailes\s+du\s+nez)\b"
        ),
        rx(
            r"\b(?:retractions?|intercostal\s+retractions?|"
            r"chest\s+retractions?|nasal\s+flaring)\b"
        )
    ),
    "orthopnea": (
        rx(r"\b(?:orthopnée|orthopnee)\b"),
        rx(r"\borthopn(?:ea|oea)\b")
    ),
    "massive_bleeding": (
        rx(
            r"\b(?:hémorragie|hemorragie|saignement)\s+"
            r"(?:massif|massive|abondant|abondante)\b"
        ),
        rx(r"\b(?:massive|profuse|heavy)\s+(?:bleeding|hemorrhage|haemorrhage)\b")
    ),
    "active_bleeding": (
        rx(
            r"\b(?:saignement|hémorragie|hemorragie)\s+"
            r"(?:actif|active|en\s+cours)\b"
        ),
        rx(r"\bactive\s+(?:bleeding|hemorrhage|haemorrhage)\b")
    ),
    "hematemesis": (
        rx(r"\b(?:hématémèse|hematemese|vomissement(?:s)?\s+de\s+sang)\b"),
        rx(r"\b(?:hematemesis|haematemesis|vomiting\s+blood|bloody\s+vomit)\b")
    ),
    "rectal_bleeding": (
        rx(
            r"\b(?:rectorragie(?:s)?|méléna|melena|sang\s+dans\s+les\s+selles|"
            r"selles\s+noires)\b"
        ),
        rx(
            r"\b(?:rectal\s+bleeding|melena|melaena|"
            r"blood\s+in\s+(?:the\s+)?stool|black\s+stool)\b"
        )
    ),
    "seizure": (
        rx(
            r"\b(?:convulsion(?:s)?|épisode\s+convulsif|"
            r"crise\s+(?:convulsive|comitiale|épileptique)|"
            r"activité\s+convulsive|syndrome\s+convulsif)\b"
        ),
        rx(
            r"\b(?:seizure(?:s)?|convulsion(?:s)?|convulsive\s+episode|"
            r"epileptic\s+seizure|seizure\s+activity)\b"
        )
    ),
    "shock": (
        rx(r"\b(?:état\s+de\s+choc|choc\s+circulatoire|choc\s+hémodynamique)\b"),
        rx(r"\b(?:circulatory|hemodynamic|haemodynamic)\s+shock\b")
    ),
    "motor_deficit": (
        rx(
            r"\b(?:(?:déficit|deficit)\s+(?:moteur|neurologique)|"
            r"hémiparésie|hemiparesie|hémiplégie|hemiplegie|parésie\s+brutale)\b"
        ),
        rx(
            r"\b(?:(?:motor|neurologic|neurological)\s+deficit|"
            r"hemiparesis|hemiplegia)\b"
        )
    ),
    "aphagia": (
        rx(r"\b(?:aphagie|impossibilité\s+d[’']avaler)\b"),
        rx(r"\b(?:aphagia|unable\s+to\s+swallow|inability\s+to\s+swallow)\b")
    ),
    "hypersalivation": (
        rx(r"\b(?:hypersialorrhée|hypersalivation|sialorrhée)\b"),
        rx(r"\b(?:hypersalivation|sialorrhea|sialorrhoea|excessive\s+salivation)\b")
    ),
    "severe_pain": (
        rx(
            r"\b(?:douleur(?:s)?\s+(?:intense|sévère|violente|insupportable)|"
            r"douleur\s+brutale\s+et\s+intense|douleur\s+à\s+(?:9|10)\s*/\s*10)\b"
        ),
        rx(r"\b(?:severe|intense|excruciating|unbearable)\s+pain\b")
    ),
    "agitation": (
        rx(
            r"\b(?:agitation|agité(?:e)?|très\s+agité(?:e)?|"
            r"agitation\s+psychomotrice|comportement\s+violent)\b"
        ),
        rx(
            r"\b(?:agitation|agitated|markedly\s+agitated|"
            r"psychomotor\s+agitation|violent\s+behavior)\b"
        )
    ),
    "purpura": (
        rx(r"\b(?:purpura|pétéchies?|petechies?|ecchymoses?)\b"),
        rx(r"\b(?:purpura|petechiae|petechial|ecchymoses?)\b")
    ),
    "severe_headache": (
        rx(
            r"\b(?:(?:céphalée|cephalee)s?\s+(?:intense|sévère|brutale|violente)|"
            r"céphalée\s+en\s+coup\s+de\s+tonnerre)\b"
        ),
        rx(r"\b(?:severe|sudden|intense|thunderclap)\s+headache\b")
    ),
    "poor_tolerance": (
        rx(r"\b(?:mauvaise\s+tolérance|mal\s+toléré(?:e)?)\b"),
        rx(r"\b(?:poor(?:ly)?\s+tolerated|poor\s+tolerance)\b")
    )
}


# ## Négations critiques

CRITICAL_SIGN_NEGATION_PATTERNS = {
    "unconsciousness": (
        rx(
            r"\b(?:sans|aucune|absence\s+de|pas\s+de)\s+"
            r"(?:perte\s+de\s+connaissance|coma)\b"
        ),
        rx(r"\b(?:no\s+loss\s+of\s+consciousness|remains?\s+conscious)\b")
    ),
    "altered_consciousness": (
        rx(
            r"\b(?:conscience\s+normale|conscience\s+claire|vigilance\s+normale|"
            r"vigilance\s+conservée|sans\s+trouble\s+de\s+la\s+conscience|"
            r"sans\s+trouble\s+de\s+vigilance)\b"
        ),
        rx(
            r"\b(?:normal\s+consciousness|alert\s+and\s+oriented|"
            r"fully\s+alert|normal\s+alertness)\b"
        )
    ),
    "cyanosis": (
        rx(r"\b(?:sans|aucune|absence\s+de|pas\s+de)\s+cyanose\b"),
        rx(r"\b(?:no|without)\s+cyanosis\b")
    ),
    "seizure": (
        rx(
            r"\b(?:sans|aucune|absence\s+de|pas\s+de)\s+"
            r"(?:convulsions?|crise\s+convulsive|épisode\s+convulsif)\b"
        ),
        rx(r"\b(?:no|without)\s+(?:seizures?|convulsions?|convulsive\s+episode)\b")
    ),
    "motor_deficit": (
        rx(
            r"\b(?:sans|aucun|absence\s+de|pas\s+de)\s+"
            r"(?:déficit|deficit)\s+(?:moteur|neurologique)\b"
        ),
        rx(r"\b(?:no|without)\s+(?:motor|neurological?)\s+deficit\b")
    ),
    "agitation": (
        rx(r"\b(?:sans|aucune|absence\s+d[’']?|pas\s+d[’']?)\s*agitation\b"),
        rx(r"\b(?:no|without)\s+agitation\b")
    )
}


# ## Concepts FRENCH

FRENCH_CONCEPT_PATTERNS = {
    "abdominal_pain": (
        rx(r"\b(?:douleur\s+abdominale|mal\s+au\s+ventre|douleur\s+du\s+ventre)\b"),
        rx(r"\b(?:abdominal\s+pain|stomach\s+pain|belly\s+pain)\b")
    ),
    "amniotic_fluid_loss": (
        rx(
            r"\b(?:perte|écoulement|ecoulement)\s+(?:de\s+)?liquide\s+amniotique\b|"
            r"\brupture\s+de\s+la\s+poche\s+des\s+eaux\b"
        ),
        rx(
            r"\b(?:amniotic\s+fluid\s+(?:loss|leakage)|rupture\s+of\s+membranes)\b"
        )
    ),
    "anaphylaxis": (
        rx(r"\b(?:anaphylaxie|choc\s+anaphylactique)\b"),
        rx(r"\b(?:anaphylaxis|anaphylactic\s+shock)\b")
    ),
    "bowel_obstruction_signs": (
        rx(
            r"\b(?:syndrome|signes?|sympt[oô]mes?)\s+d[’']occlusion\b|"
            r"\barr[êe]t\s+(?:des\s+)?(?:gaz|matières|matieres|transit)\b"
        ),
        rx(
            r"\b(?:bowel|intestinal)\s+obstruction\b|"
            r"\bcessation\s+of\s+bowel\s+transit\b"
        )
    ),
    "chemical_burn": (
        rx(r"\b(?:brûlure|brulure)\s+chimique\b|\bprojection\s+chimique\b"),
        rx(r"\bchemical\s+(?:burn|exposure)\b")
    ),
    "chronic_stable": (
        rx(
            r"\b(?:ancien(?:ne)?s?\s+et\s+stables?|chronique\s+stable|"
            r"stable\s+depuis\s+longtemps)\b"
        ),
        rx(
            r"\b(?:long-standing\s+and\s+stable|chronic\s+stable|"
            r"stable\s+for\s+a\s+long\s+time)\b"
        )
    ),
    "complete_postictal_recovery": (
        rx(
            r"\b(?:récupération|recuperation)\s+complète\s+post(?:-|\s*)critique\b|"
            r"\brécupération\s+complète\s+après\s+la\s+crise\b"
        ),
        rx(
            r"\bcomplete\s+postictal\s+recovery\b|"
            r"\bfully\s+recovered\s+after\s+(?:the\s+)?seizure\b"
        )
    ),
    "distal_thrombosis": (
        rx(r"\b(?:siège|siege|thrombose)\s+distal(?:e)?\b"),
        rx(r"\bdistal\s+(?:location|thrombosis|dvt)\b")
    ),
    "dyspnea_while_speaking": (
        rx(
            r"\b(?:dyspnée|dyspnee)\s+(?:à|a)\s+la\s+parole\b|"
            r"\bincapable\s+de\s+parler\s+en\s+phrases\b"
        ),
        rx(
            r"\bdyspn(?:ea|oea)\s+while\s+speaking\b|"
            r"\bunable\s+to\s+speak\s+in\s+full\s+sentences\b"
        )
    ),
    "fever": (
        rx(r"\b(?:fièvre|fievre|fébrile|febrile|hyperthermie)\b"),
        rx(r"\b(?:fever|febrile|hyperthermia)\b")
    ),
    "first_headache_episode": (
        rx(
            r"\b(?:premier|1er)\s+(?:épisode|episode)\s+de\s+(?:céphalée|cephalee)|"
            r"\bpremière\s+céphalée\b"
        ),
        rx(r"\bfirst\s+(?:episode\s+of\s+)?headache\b|\bfirst-ever\s+headache\b")
    ),
    "head_trauma": (
        rx(r"\b(?:traumatisme\s+crânien|traumatisme\s+cranien|TC)\b"),
        rx(r"\b(?:head\s+trauma|head\s+injury|traumatic\s+brain\s+injury)\b")
    ),
    "high_velocity_trauma": (
        rx(
            r"\b(?:haute|forte)\s+vélocité\b|"
            r"\btraumatisme\s+à\s+haute\s+énergie\b"
        ),
        rx(r"\b(?:high[- ]velocity|high[- ]energy)\s+trauma\b")
    ),
    "high_voltage": (
        rx(r"\bhaute\s+tension\b"),
        rx(r"\bhigh\s+voltage\b")
    ),
    "household_current": (
        rx(r"\b(?:courant|électricité|electricite)\s+domestique\b"),
        rx(r"\bhousehold\s+(?:current|electricity)\b")
    ),
    "hypotonia": (
        rx(r"\b(?:hypotonie|hypotonique)\b"),
        rx(r"\b(?:hypotonia|hypotonic)\b")
    ),
    "large_abscess": (
        rx(r"\babcès\s+(?:volumineux|important|étendu)\b"),
        rx(r"\b(?:large|extensive)\s+abscess\b")
    ),
    "lightning_strike": (
        rx(r"\b(?:foudre|frappé(?:e)?\s+par\s+la\s+foudre)\b"),
        rx(r"\blightning\s+strike\b|\bstruck\s+by\s+lightning\b")
    ),
    "limb_deformity": (
        rx(r"\b(?:déformation|deformation)\s+(?:du\s+)?membre\b|\bmembre\s+déformé\b"),
        rx(r"\b(?:limb\s+deformity|deformed\s+limb)\b")
    ),
    "low_velocity_trauma": (
        rx(r"\b(?:faible|basse)\s+vélocité\b"),
        rx(r"\blow[- ]velocity\s+trauma\b")
    ),
    "multiple_seizures": (
        rx(r"\b(?:crises?|convulsions?)\s+(?:multiples|répétées|récidivantes)\b"),
        rx(r"\b(?:multiple|recurrent|repeated)\s+(?:seizures?|convulsions?)\b")
    ),
    "nasal_flaring": (
        rx(r"\bbattements?\s+des\s+ailes\s+du\s+nez\b"),
        rx(r"\bnasal\s+flaring\b")
    ),
    "ongoing_seizure": (
        rx(
            r"\b(?:crise|convulsion)\s+(?:en\s+cours|persistante)|"
            r"\bétat\s+de\s+mal\s+épileptique\b"
        ),
        rx(r"\b(?:ongoing\s+seizure|active\s+seizure|status\s+epilepticus)\b")
    ),
    "open_fracture": (
        rx(r"\bfracture\s+ouverte\b"),
        rx(r"\bopen\s+fracture\b")
    ),
    "paresthesia": (
        rx(r"\b(?:paresthésie|paresthesie|fourmillements?|engourdissements?)\b"),
        rx(r"\b(?:paresthesia|paresthesias|tingling|numbness)\b")
    ),
    "pregnancy": (
        rx(r"\b(?:grossesse|enceinte|gestante)\b"),
        rx(r"\b(?:pregnancy|pregnant|gestation)\b")
    ),
    "profuse_diarrhea": (
        rx(
            r"\b(?:diarrh[ée]e\s+(?:abondante|profuse|importante)|"
            r"selles\s+(?:très\s+)?abondantes)\b"
        ),
        rx(r"\b(?:profuse|copious|severe)\s+diarrh(?:ea|oea)\b")
    ),
    "profuse_vomiting": (
        rx(
            r"\b(?:vomissements?\s+(?:abondants?|profus|importants?)|"
            r"vomit\s+abondamment)\b"
        ),
        rx(r"\b(?:profuse|copious|heavy)\s+vomiting\b")
    ),
    "proximal_thrombosis": (
        rx(r"\b(?:siège|siege|thrombose)\s+proximal(?:e)?\b"),
        rx(r"\bproximal\s+(?:location|thrombosis|dvt)\b")
    ),
    "referred_by_physician": (
        rx(
            r"\b(?:adressé|adressée|adresse|adressee)\s+"
            r"(?:aux\s+urgences\s+)?par\s+(?:un|le|son)\s+médecin\b"
        ),
        rx(
            r"\breferred\s+(?:to\s+(?:the\s+)?(?:ED|emergency\s+department)\s+)?"
            r"by\s+(?:a|the|his|her)\s+physician\b"
        )
    ),
    "resolved_episode": (
        rx(
            r"\b(?:épisode|episode|sympt[oô]mes?)\s+(?:résolutif|resolutif|"
            r"résolu|resolu)|\bspontanément\s+résolu\b"
        ),
        rx(
            r"\b(?:resolved\s+episode|symptoms?\s+resolved|"
            r"spontaneously\s+resolved)\b"
        )
    ),
    "significant_local_signs": (
        rx(
            r"\b(?:signes?\s+locaux\s+(?:francs?|importants?)|"
            r"importants?\s+signes?\s+locaux)\b"
        ),
        rx(r"\b(?:marked|significant)\s+local\s+signs?\b")
    ),
    "snake_or_scorpion_bite": (
        rx(r"\b(?:morsure\s+de\s+serpent|piqûre\s+de\s+scorpion)\b"),
        rx(r"\b(?:snake\s+bite|scorpion\s+sting)\b")
    ),
    "sudden_headache": (
        rx(
            r"\b(?:céphalée|cephalee)s?\s+(?:brutale|soudaine)|"
            r"\bcéphalée\s+en\s+coup\s+de\s+tonnerre\b"
        ),
        rx(r"\b(?:sudden|thunderclap|abrupt)\s+headache\b")
    ),
    "sudden_hearing_loss": (
        rx(
            r"\b(?:surdité|surdite|baisse\s+d[’']audition)\s+"
            r"(?:brutale|soudaine)\b"
        ),
        rx(r"\bsudden\s+hearing\s+loss\b")
    ),
    "sudden_vision_loss": (
        rx(
            r"\b(?:baisse|perte)\s+(?:brutale|soudaine)\s+"
            r"de\s+(?:la\s+)?vision\b"
        ),
        rx(r"\bsudden\s+(?:vision\s+loss|loss\s+of\s+vision)\b")
    ),
    "vaginal_bleeding": (
        rx(r"\b(?:métrorragie|metrorragie|saignement\s+vaginal)\b"),
        rx(r"\b(?:vaginal\s+bleeding|metrorrhagia)\b")
    ),
    "violent_behavior": (
        rx(r"\b(?:comportement\s+violent|violence|violent(?:e)?)\b"),
        rx(r"\b(?:violent\s+behavior|violence|violent)\b")
    ),
    "withdrawal_state": (
        rx(
            r"\b(?:état|etat|syndrome)\s+de\s+manque\b|"
            r"\bsevrage\s+(?:aigu|sévère|severe)\b"
        ),
        rx(r"\b(?:withdrawal\s+state|acute\s+withdrawal|withdrawal\s+syndrome)\b")
    )
}


# ## Négations concepts FRENCH

FRENCH_CONCEPT_NEGATION_PATTERNS = {
    "fever": (
        rx(r"\b(?:sans|absence\s+de|pas\s+de)\s+fièvre\b"),
        rx(r"\b(?:afebrile|no\s+fever|without\s+fever)\b")
    ),
    "abdominal_pain": (
        rx(r"\b(?:sans|absence\s+de|pas\s+de)\s+douleur\s+abdominale\b"),
        rx(r"\b(?:no|without)\s+abdominal\s+pain\b")
    ),
    "multiple_seizures": (
        rx(r"\b(?:crise\s+unique|épisode\s+unique|episode\s+unique)\b"),
        rx(r"\b(?:single\s+seizure|single\s+episode)\b")
    ),
    "ongoing_seizure": (
        rx(
            r"\b(?:crise|convulsion)\s+(?:terminée|terminee|résolue|resolue)|"
            r"\bplus\s+de\s+convulsions?\b"
        ),
        rx(r"\b(?:seizure\s+resolved|no\s+ongoing\s+seizure)\b")
    ),
    "head_trauma": (
        rx(r"\b(?:sans|absence\s+de|pas\s+de)\s+traumatisme\s+crânien\b"),
        rx(r"\b(?:no|without)\s+head\s+(?:trauma|injury)\b")
    ),
    "dyspnea_while_speaking": (
        rx(r"\bparle\s+en\s+phrases\s+complètes\b"),
        rx(r"\bspeaks?\s+in\s+full\s+sentences\b")
    ),
    "nasal_flaring": (
        rx(
            r"\b(?:sans|absence\s+de|pas\s+de)\s+"
            r"battements?\s+des\s+ailes\s+du\s+nez\b"
        ),
        rx(r"\bno\s+nasal\s+flaring\b")
    ),
    "pregnancy": (
        rx(
            r"\b(?:non\s+enceinte|grossesse\s+exclue|"
            r"test\s+de\s+grossesse\s+négatif)\b"
        ),
        rx(
            r"\b(?:not\s+pregnant|pregnancy\s+excluded|"
            r"negative\s+pregnancy\s+test)\b"
        )
    ),
    "vaginal_bleeding": (
        rx(
            r"\b(?:sans|absence\s+de|pas\s+de)\s+"
            r"(?:métrorragie|metrorragie|saignement\s+vaginal)\b"
        ),
        rx(r"\b(?:no|without)\s+vaginal\s+bleeding\b")
    ),
    "amniotic_fluid_loss": (
        rx(
            r"\b(?:sans|absence\s+de|pas\s+de)\s+"
            r"perte\s+de\s+liquide\s+amniotique\b"
        ),
        rx(r"\b(?:no|without)\s+amniotic\s+fluid\s+(?:loss|leakage)\b")
    ),
    "hypotonia": (
        rx(r"\b(?:sans|absence\s+de|pas\s+d[’']?)\s*hypotonie\b"),
        rx(r"\b(?:no|without)\s+hypotonia\b")
    ),
    "paresthesia": (
        rx(r"\b(?:sans|absence\s+de|pas\s+de)\s+paresthésies?\b"),
        rx(r"\b(?:no|without)\s+paresthesias?\b")
    ),
    "referred_by_physician": (
        rx(r"\b(?:non\s+adressé|non\s+adressee|vient\s+de\s+lui-même)\b"),
        rx(r"\b(?:self-referred|not\s+referred\s+by\s+a\s+physician)\b")
    ),
    "anaphylaxis": (
        rx(r"\b(?:sans|absence\s+d[’']?|pas\s+d[’']?)\s*anaphylaxie\b"),
        rx(r"\b(?:no|without)\s+anaphylaxis\b")
    ),
    "significant_local_signs": (
        rx(
            r"\b(?:sans|absence\s+de|pas\s+de)\s+"
            r"signes?\s+locaux\s+importants?\b"
        ),
        rx(r"\b(?:no|without)\s+significant\s+local\s+signs?\b")
    ),
    "limb_deformity": (
        rx(
            r"\b(?:sans|absence\s+de|pas\s+de)\s+"
            r"déformation\s+(?:du\s+)?membre\b"
        ),
        rx(r"\b(?:no|without)\s+limb\s+deformity\b")
    ),
    "open_fracture": (
        rx(r"\b(?:sans|absence\s+de|pas\s+de)\s+fracture\s+ouverte\b"),
        rx(r"\b(?:no|without)\s+open\s+fracture\b")
    )
}


# ## Plausibilité

PLAUSIBILITY_LIMITS = {
    "age_days": (0, 31),
    "age_months": (0, 23),
    "age_years": (0, 120),
    "temperature": (25.0, 45.0),
    "heart_rate": (20, 300),
    "systolic_bp": (30, 300),
    "diastolic_bp": (10, 200),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (5, 100),
    "gcs": (3, 15),
    "blood_glucose_mmol_l": (0.1, 80.0),
    "ketones_mmol_l": (0.0, 20.0),
    "pain_score": (0.0, 10.0)
}

CMHG_TO_MMHG = 10


# ## Registres

AGE_RULES = (
    ("age_days", AGE_DAY_PATTERNS, "age_days"),
    ("age_months", AGE_MONTH_PATTERNS, "age_months"),
    ("age_years", AGE_YEAR_PATTERNS, "age_years")
)

VITAL_RULES = {
    "heart_rate": (HEART_RATE_PATTERNS, "heart_rate"),
    "spo2": (SPO2_PATTERNS, "spo2"),
    "respiratory_rate": (RESPIRATORY_RATE_PATTERNS, "respiratory_rate"),
    "gcs": (GCS_PATTERNS, "gcs")
}