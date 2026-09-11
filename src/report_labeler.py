#!/usr/bin/env python3
"""
Multilingual knee-MRI report labeler — severity-graded weak supervision.

THE THESIS (WORKING_NOTE.md §4.3, §8 bet A)
-------------------------------------------
The official labels are SEVERITY-THRESHOLDED, not presence-based. The rubric grades
"on the fence" as negative, requires >50% ACL fibre disruption, requires meniscal signal to
REACH THE SURFACE, requires OA to be >50% cartilage thickness over >=1 cm, and requires
effusion to be "moderate or large". A report saying "small effusion" or "mild chondropathy"
is CORRECTLY labelled 0.

And the metric is ROC-AUC, which is purely RANK-BASED. So a binary target is actively
wasteful: it has 2 distinct values and cannot rank within either block — every positive ties
with every other positive. A graded target ranks correctly under ANY threshold, which is
exactly what immunises it against the report-vs-image threshold mismatch.

TWO EXTRACTORS, NOT ONE (§4.4)
------------------------------
The rubric asks two different kinds of question, and the reports answer them with two
different vocabularies:

  MAGNITUDE  ("how much?")  OA x3, Effusion, Synovitis, Baker's
             -> graded by magnitude words and Outerbridge/ICRS grades.
             Severity language co-occurs 28-57% of the time.
  CATEGORICAL("what kind?") ACL, MCL, Meniscus x2, Contusion, Fracture
             -> graded by KIND: complete vs partial vs degenerative; surface-reaching vs
             intrasubstance. Severity language co-occurs only 15-27% of the time.

TWO OUTPUTS, NEVER CONFLATED (§5.5 — this bug cost another team 0.121 AUC on one label)
--------------------------------------------------------------------------------------
  severity   in [0,1] : the RANK estimate. Only its ordering is ever scored.
  confidence in [0,1] : the UNCERTAINTY, used as a per-sample loss weight.

Putting P(positive | unmentioned) into `severity` inverts the evidence: it places every
silent study ABOVE a report that explicitly says "mild synovitis". A mention is evidence FOR
a finding. Uncertainty belongs in `confidence`.

NEGATION IS TESTED FIRST (§5.6 — the other team's most expensive bug)
--------------------------------------------------------------------
A scorer that checks pathology keywords before negation is broken: "medial meniscus: no tear"
matches TEAR and scores positive, because the negation branch becomes unreachable whenever
any pathology word is present — which is nearly always. See the unit tests at the bottom.

STATUS: this is a reproducible rule-based BASELINE, not a clinical-grade NLP labeler. It is
the control and the floor. PLAN.md Phase 2 compares it head-to-head against an open-weights
multilingual LLM pass, and uses labeler DISAGREEMENT as an uncertainty signal (§8 bet F).

RULES: report text must not be sent to a hosted LLM API (§1.6). This module is pure regex and
runs offline in seconds.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

MAGNITUDE_LABELS = {"Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"}
CATEGORICAL_LABELS = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion", "Fracture"}

# Turkish dotless i and friends must fold BEFORE casefolding, or "IZLENMEZ" and "izlenmez"
# become different tokens.
_PRE = str.maketrans({"ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
                      "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae"})


def normalize(text: str) -> str:
    """Fold case, diacritics and separators. Greek and Cyrillic letters survive.

    NFKD strips Latin accents and Greek tonos alike (ά -> α), which is what we want — reports
    are wildly inconsistent about accents. It also folds MICRO SIGN U+00B5 to a real mu, which
    matters because many reports use the wrong codepoint.
    """
    if not isinstance(text, str):
        return ""
    t = text.translate(_PRE).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("­", "")                      # soft hyphen
    t = re.sub(r"intact(?=\d)", " ", t)              # §3.3 substitution artifact: intact9xintact4cm
    t = re.sub(r"[_/\\]+", " ", t)
    return re.sub(r"[ \t]+", " ", t)


_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")
_SUBCLAUSE = re.compile(
    r",|\bbut\b|\bhowever\b|\bwhereas\b|\bpero\b|\bmais\b|\bmaar\b|\baber\b|\bjedoch\b|"
    r"\bancak\b|\bfakat\b|\bομως\b|\bαλλα\b|\bно\b|\bали\b")


def clauses(text: str) -> list[str]:
    """Split into clauses, attaching `heading:` lines to the value beneath them.

    A report reading `Fractures :` / `Aucune.` is ONE statement. Split on punctuation alone
    and the anatomy loses its negation, flipping the label positive.
    """
    raw = [c.strip() for c in _SENT.split(normalize(text)) if c and c.strip()]
    out: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            out.append(c + " " + raw[i + 1])   # merged: heading carries its value's polarity
            i += 1
            out.append(raw[i])                 # and the value stands alone too
        else:
            out.append(c)
        i += 1
    final: list[str] = []
    for c in out:
        final.append(c)
        if len(c.split()) > 18:                # long clauses hide independent assertions
            final.extend(p.strip() for p in _SUBCLAUSE.split(c) if len(p.split()) > 2)
    return final


def rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- polarity
NEG = rx(r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b", r"\bnone\b",
         r"\bunremarkable\b", r"\bfree of\b", r"\babsent\b",
         r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se observ",
         r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
         r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
         r"\bkeine?n?\b", r"\bohne\b", r"\bnicht\b", r"\bkein\b",
         r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bizlenmedi\b",
         r"gozlenmemekte", r"mevcut degil", r"gorulmedi", r"\bsaptanmamistir\b",
         r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
         r"\bδεν\b", r"\bχωρις\b", r"ουδεν", r"\bαρνητικ",
         r"\bбез\b", r"липсва", r"\bняма\b", r"\bотсъств")

NORMAL = rx(r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"within normal limits", r"\bwnl\b",
            r"limites normales", r"\bconservad", r"\bintegr", r"\bhabitual",
            r"\bdoga(l|ll)", r"korunmus", r"\bnormaldir\b", r"olagan", r"\bsalim\b",
            r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt",
            r"φυσιολογικ", r"ακεραι", r"ανευ ευρηματων",
            r"unauffallig", r"regelrecht", r"нормал", r"запазен", r"съхранен",
            r"без особености", r"\bgaaf\b", r"\bnormaal\b")

HEDGE = rx(r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected\b",
           r"cannot (be )?exclude", r"\bquestionable\b", r"\bequivocal\b", r"\blikely\b",
           r"\bposible\b", r"\bdudos", r"\bsugestiv",
           r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim",
           r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b",
           r"πιθαν", r"υποπτ", r"\bmoglich", r"\bverdacht", r"\bfraglich",
           r"\bвъзможно\b", r"\bвероятно\b", r"суспект", r"\bmogelijk\b")

# --------------------------------------------------------------------------- magnitude scale
# Scores are RANKS on a severity continuum, tuned to the rubric's own cut: the rubric calls
# "moderate or large" positive and "mild/small/trace" negative, so the gap between 0.15 and
# 0.62 is deliberately wide — that is where the decision boundary lives.
MAG: list[tuple[float, re.Pattern]] = [
    (0.05, rx(r"\btrace\b", r"\bminimal", r"\bminim", r"\bminiem", r"\bpunctate",
              r"ελαχιστ", r"αμελητε", r"минимал", r"\baz miktarda\b", r"\bcok az\b",
              r"\beser\b", r"\bmalko\b")),
    (0.15, rx(r"\bmild", r"\bslight", r"\bsmall\b", r"\blow.grade", r"\bleve\b", r"\bligera",
              r"\bpequen", r"\bleger", r"\bfaible\b", r"\bpetit", r"\blicht", r"\bgering",
              r"\bklein", r"\bhafif", r"\bkucuk\b", r"ηπι", r"\bμικρ", r"\bлек", r"\bмалк",
              r"\bblag", r"\bmali\b", r"\bmanji\b", r"\bdiskret", r"\bdiscret",
              r"\bsuperficial", r"\byuzeyel", r"\boppervlakkig", r"\boberflachlich")),
    (0.62, rx(r"\bmoderate", r"\bmoderad", r"\bmodere", r"\bmatig", r"\bmassig", r"\bmasig",
              r"\bmittelgradig", r"\borta\b", r"μετρι", r"\bумерен", r"\bсреден",
              r"\bumjeren", r"\bsrednj", r"\bpartial thickness")),
    (0.90, rx(r"\bsevere", r"\bmarked", r"\blarge\b", r"\bmassive", r"\badvanced",
              r"\bextensive", r"\bhigh.grade", r"\bfull.thickness", r"\bcomplete\b",
              r"\bgross\b", r"\bsevero", r"\bgrave", r"\bimportante", r"\bgrande",
              r"\bextens", r"\bavanzad", r"\bernstig", r"\bgroot", r"\buitgebreid",
              r"\bausgepragt", r"\bschwer", r"\bstark", r"\bhochgradig",
              r"\bileri\b", r"\bbelirgin", r"\bbuyuk\b", r"\byaygin", r"\bciddi",
              r"σοβαρ", r"μεγαλ", r"εκτεταμεν", r"\bтежк", r"\bизразен", r"\bголям",
              r"\btezak", r"\bizrazit", r"\bvelik", r"\bopsezn")),
]

_GRADE_NUM = re.compile(
    r"(?:grade?|grado|graad|grad|derece|evre|stadi\w*|βαθμ\w*|степен\w*|gr\.?)\s*([1-4]|i{1,3}v?|iv)\b")
_GRADE_POST = re.compile(r"\b([1-4])\s*(?:степен|derece|\.?\s*grad)")
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
# Outerbridge/ICRS are 1-4 chondral scales. The rubric's ">50% thickness" is grade 3-4.
_GRADE_TO_SCORE = {1: 0.05, 2: 0.15, 3: 0.62, 4: 0.90}


def grade_score(clause: str) -> float | None:
    for m in (_GRADE_NUM.search(clause), _GRADE_POST.search(clause)):
        if m:
            g = m.group(1)
            n = int(g) if g.isdigit() else _ROMAN.get(g)
            if n in _GRADE_TO_SCORE:
                return _GRADE_TO_SCORE[n]
    return None


def magnitude_score(clause: str) -> float | None:
    """Highest magnitude cue in the clause, preferring an explicit grade. None if silent."""
    g = grade_score(clause)
    best = None
    for score, pat in MAG:
        if pat.search(clause):
            best = score if best is None else max(best, score)
    if g is not None and best is not None:
        return max(g, best)
    return g if g is not None else best


# --------------------------------------------------------------------------- categorical scale
# For the "what kind?" labels, severity is a CATEGORY, not a degree. Ordered low -> high.
CAT: list[tuple[float, re.Pattern]] = [
    (0.10, rx(r"\bdegenerat", r"\bmucoid", r"\bintrasubstance", r"\bmyxoid",
              r"\bdegenerativ", r"\bdejenerat", r"\bintrasustancia", r"\bno reaching",
              r"\bnot reach", r"\bdoes not extend", r"\bno alcanza", r"εκφυλιστικ",
              r"дегенератив", r"\bdegenerativn")),
    (0.30, rx(r"\bsprain\b", r"\besguince", r"\bentorse", r"\bzerrung", r"\bdistorsi",
              r"\bburkulma", r"\bthicken", r"\bengrosamiento", r"\bsignal (change|alteration)",
              r"\bedema\b", r"\bodem", r"\bcontusion", r"\bcontusao", r"\bbone bruise",
              r"\bkemik kontuzyon")),
    (0.68, rx(r"\bpartial (tear|rupture|thickness)", r"\brotura parcial", r"\bdesgarro parcial",
              r"\brupture partielle", r"\bteilruptur", r"\bpartiele ruptuur",
              r"\bparsiyel", r"\bmerkezi yirtik", r"\bμερικη ρηξη", r"\bчастичн",
              r"\bparcijaln", r"\breaches the (articular )?surface", r"\bsurface.reaching",
              r"\bextends to the (articular )?surface", r"\bcontacts the",
              r"\bllega a la superficie")),
    (0.92, rx(r"\b(complete|full.thickness|total) (tear|rupture)", r"\bcomplete disruption",
              r"\btear\b", r"\btorn\b", r"\brupture", r"\brotura", r"\bdesgarro",
              r"\bdechirure", r"\bruptur", r"\bscheur", r"\byirtik", r"\brupturu",
              r"\bρηξη", r"\bразрыв", r"\bруптура", r"\bскъсван", r"\bpuknuce",
              r"\bfracture", r"\bfractura", r"\bfraktur", r"\bkirik", r"\bκαταγμα",
              r"\bфрактур", r"\bприлом", r"\bprijelom", r"\bavulsion")),
]


# A "partial"/"incomplete" qualifier CAPS the categorical score. Without this, "partial tear"
# matches the 0.68 tier AND the bare \btear\b in the 0.92 tier, and max() promotes it to
# "complete" — collapsing the very distinction the ACL and meniscus rubrics turn on
# (>50% fibre disruption; signal reaching the surface). Caught by the ACL-ladder unit test.
PARTIAL_QUAL = rx(r"\bpartial", r"\bincomplete", r"\bparcial", r"\bpartielle", r"\bteil",
                  r"\bparsiyel", r"\bpartiele", r"\bμερικ", r"\bчастичн", r"\bparcijaln",
                  r"\bnepotpun", r"\bdelvis")

# Likewise, a low-grade qualifier on a categorical finding pulls it back down: the MCL rubric
# grades "low-grade sprain" negative outright (§2.4).
LOWGRADE_QUAL = rx(r"\blow.grade", r"\bminor\b", r"\bbajo grado", r"\bgeringgradig",
                   r"\bdusuk dereceli", r"\bniedriggradig")


def categorical_score(clause: str) -> float | None:
    best = None
    for score, pat in CAT:
        if pat.search(clause):
            best = score if best is None else max(best, score)
    if best is None:
        return None
    if PARTIAL_QUAL.search(clause):
        best = min(best, 0.68)        # cannot be graded "complete"
    if LOWGRADE_QUAL.search(clause):
        best = min(best, 0.30)        # rubric grades low-grade injury negative
    return best


# --------------------------------------------------------------------------- anatomy
ANAT: dict[str, re.Pattern] = {
    "ACL": rx(r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blca\b",
              r"croise anterieur", r"voorste kruisband", r"\bvkb\b",
              r"vordere[sn]? kreuzband", r"vorderen kreuzband",
              r"on capraz", r"\bocb\b", r"prednj\w* krizn", r"προσθι\w* χιαστ",
              r"предна кръстна", r"передн\w* крестообраз"),
    "MCL": rx(r"medial collateral", r"\bmcl\b", r"tibial collateral",
              r"colateral (medial|interno)", r"\blcm\b", r"collateral (medial|interne)",
              r"mediale collaterale", r"binnenband", r"innenband", r"mediales? kollateral",
              r"ic yan bag", r"medial kollateral", r"\biyb\b",
              r"medijaln\w* kolateraln", r"εσω πλαγι", r"εσωτερικο πλαγι",
              r"медиален колатерал", r"вътрешн\w* странич"),
    "Medial Meniscus": rx(r"medial meniscus", r"menisco (medial|interno)", r"menisque (medial|interne)",
                          r"mediale meniscus", r"binnenmeniscus", r"innenmeniskus",
                          r"medial meniskus", r"ic menisk", r"medijaln\w* meniskus",
                          r"εσω μηνισκ", r"медиален мениск", r"вътрешен мениск"),
    "Lateral Meniscus": rx(r"lateral meniscus", r"menisco (lateral|externo)", r"menisque (lateral|externe)",
                           r"laterale meniscus", r"buitenmeniscus", r"aussenmeniskus",
                           r"lateral meniskus", r"dis menisk", r"lateraln\w* meniskus",
                           r"εξω μηνισκ", r"латерален мениск", r"външен мениск"),
    "Medial OA": rx(r"medial (compartment|femorotibial|tibiofemoral)", r"compartimento (medial|interno)",
                    r"compartiment (medial|interne)", r"mediale[ns]? kompartiment",
                    r"medial kompartman", r"ic kompartman", r"εσω διαμερισμα",
                    r"медиален компартм"),
    "Lateral OA": rx(r"lateral (compartment|femorotibial|tibiofemoral)", r"compartimento (lateral|externo)",
                     r"compartiment (lateral|externe)", r"laterale[ns]? kompartiment",
                     r"lateral kompartman", r"dis kompartman", r"εξω διαμερισμα",
                     r"латерален компартм"),
    "PF OA": rx(r"patellofemoral", r"patelo.?femoral", r"retropatell", r"\bpatella\w* cartilag",
                r"femoropatel", r"patellofemoraal", r"patellofemoral", r"patellofemoralen",
                r"επιγονατιδομηρια", r"пателофеморал"),
    "Effusion": rx(r"\beffusion", r"joint fluid", r"derrame", r"epanchement", r"\bhydrops",
                   r"gelenkerguss", r"\berguss", r"gewrichtsvocht", r"\bvocht in",
                   r"eklem (sivisi|efuzyon)", r"\bizliv\b", r"\bvyliv", r"αρθρικη συλλογη",
                   r"υγρο στην αρθρωση", r"излив", r"изливи", r"течност в"),
    "Synovitis": rx(r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
                    r"synoviale", r"synovialis", r"υμενιτ", r"синовит"),
    "Baker's": rx(r"baker", r"popliteal cyst", r"quiste de baker", r"kyste popl",
                  r"poplitealzyste", r"bakerzyste", r"popliteale cyste",
                  r"popliteal kist", r"κυστη baker", r"κυστη τ(ου|ης) baker",
                  r"поплитеал\w* киста", r"бейкер"),
    "Contusion": rx(r"contusion", r"bone (bruise|marrow (o|oe)dema)", r"marrow (o|oe)dema",
                    r"contusao", r"contusion osea", r"edema (oseo|de medula)",
                    r"knochenmark(o|oe)dem", r"\bbone marrow edema", r"botcontusie",
                    r"kemik (kontuzyon|ilik odem)", r"οστικ\w* (θλαση|οιδημα)",
                    r"костн\w* контузия", r"оток на костния"),
    "Fracture": rx(r"fracture", r"fractura", r"fraktur", r"\bkirik\b", r"καταγμα",
                   r"фрактур", r"прелом", r"prijelom", r"\bbreuk\b", r"avulsion"),
}

# A clause mentioning the compartment is evidence for that compartment's OA only if it also
# talks about cartilage/arthrosis. "medial compartment meniscal tear" is not OA.
OA_CONTEXT = rx(r"chondr", r"cartilag", r"cartilago", r"kraakbeen", r"knorpel", r"kikirdak",
                r"χονδρ", r"хрущял", r"хрящ", r"arthros", r"artros", r"arthrit", r"osteoarthr",
                r"gonarthros", r"gonartroz", r"οστεοαρθρ", r"артроз", r"osteofit", r"osteophyt")


@dataclass
class LabelOut:
    severity: float      # rank estimate in [0,1]. ONLY its ordering is scored.
    confidence: float    # uncertainty -> per-sample loss weight
    state: str           # for auditing: which branch produced this


# Priors for "the report never mentions this finding". These are RANK positions, and they sit
# BELOW any explicit mention and ABOVE an explicit negation — see the ordering assertion in
# assert_ordering(). They are NOT P(positive | unmentioned); that conflation is the §5.5 bug.
UNMENTIONED_SEVERITY = 0.03
NEGATED_SEVERITY = 0.01


def report_completeness(text: str) -> float:
    """How much a silence in this report actually means. §3.3.

    A 30-word report does not enumerate negatives, so "not mentioned" is uninformative.
    A 300-word structured report with section headers does enumerate them, so silence is
    strong evidence of absence. Because report FORMAT tracks language, which tracks SITE,
    getting this wrong injects site-correlated bias straight into the training labels.
    """
    n = len(normalize(text).split())
    has_headers = bool(re.search(r"^[a-z][a-z \t]{3,}:", normalize(text), re.M))
    c = min(1.0, n / 150.0)
    if has_headers:
        c = min(1.0, c + 0.25)
    return round(c, 3)


def label_report(text: str) -> dict[str, LabelOut]:
    """Score all 12 labels for one report."""
    cls = clauses(text)
    completeness = report_completeness(text)
    out: dict[str, LabelOut] = {}

    for label in LABELS:
        anat = ANAT[label]
        best: tuple[float, float, str] | None = None   # (severity, confidence, state)

        for c in cls:
            if not anat.search(c):
                continue
            if label.endswith("OA") and label != "PF OA" and not OA_CONTEXT.search(c):
                continue      # compartment named, but not about cartilage

            # --- NEGATION FIRST. This ordering is the whole ballgame (§5.6). ---------
            if NEG.search(c) or NORMAL.search(c):
                cand = (NEGATED_SEVERITY, 0.9 * max(completeness, 0.5), "explicit_negative")
            else:
                score = (magnitude_score(c) if label in MAGNITUDE_LABELS
                         else categorical_score(c))
                if score is None:
                    # Mentioned, not negated, but ungraded. A bare mention is weak evidence
                    # FOR the finding, so it must rank above silence — but well below a
                    # graded positive.
                    cand = (0.45, 0.45, "mentioned_ungraded")
                else:
                    conf = 0.9
                    if HEDGE.search(c):
                        score = score * 0.75          # hedged findings rank lower...
                        conf = 0.5                    # ...and carry less weight
                    cand = (score, conf, "graded")

            # Keep the strongest assertion in the report. A report that says "no tear" in one
            # clause and "complete rupture" in another is describing two structures or
            # correcting itself; the positive assertion is the informative one.
            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:
            out[label] = LabelOut(UNMENTIONED_SEVERITY,
                                  round(0.15 + 0.55 * completeness, 3),
                                  "unmentioned")
        else:
            out[label] = LabelOut(round(best[0], 4), round(best[1], 3), best[2])
    return out


def label_presence(text: str) -> dict[str, float]:
    """The CONTROL: binary presence extraction, exactly what everyone else does.

    Phase 0 scores this against label_report() on the 58 gold studies. If severity does not
    beat presence on >=8 of 12 labels, bet A is falsified and we fall back to this (PLAN §5).
    """
    cls = clauses(text)
    out = {}
    for label in LABELS:
        anat = ANAT[label]
        pos = 0.0
        for c in cls:
            if not anat.search(c):
                continue
            if label.endswith("OA") and label != "PF OA" and not OA_CONTEXT.search(c):
                continue
            if NEG.search(c) or NORMAL.search(c):
                continue
            pos = 1.0
            break
        out[label] = pos
    return out


def assert_ordering() -> None:
    """Guard the rank invariant that §5.5 shows is easy to break.

    explicit_negative < unmentioned < hedged/graded-positive.
    If this ever fails, the labeler is inverting evidence and every downstream AUC is wrong.
    """
    assert NEGATED_SEVERITY < UNMENTIONED_SEVERITY, "negated must rank below silence"
    assert UNMENTIONED_SEVERITY < MAG[0][0], "silence must rank below even a 'trace' mention"
    assert UNMENTIONED_SEVERITY < CAT[0][0], "silence must rank below even a 'degenerative' mention"
    assert MAG[0][0] < MAG[1][0] < MAG[2][0] < MAG[3][0], "magnitude scale must be monotone"
    assert CAT[0][0] < CAT[1][0] < CAT[2][0] < CAT[3][0], "categorical scale must be monotone"


# --------------------------------------------------------------------------- tests
def _tests() -> None:
    assert_ordering()
    fail = 0

    def want(text, label, lo, hi, why):
        nonlocal fail
        got = label_report(text)[label].severity
        ok = lo <= got <= hi
        if not ok:
            fail += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label:<17} {got:<7} want [{lo},{hi}]  {why}")

    print("negation must be tested BEFORE pathology keywords (§5.6):")
    want("Medial meniscus: no tear.", "Medial Meniscus", 0.0, 0.05,
         "the bug that cost another team an entire label")
    want("There is no evidence of an anterior cruciate ligament tear.", "ACL", 0.0, 0.05, "English")
    want("Menisco interno: sin rotura.", "Medial Meniscus", 0.0, 0.05, "Spanish heading+value")
    want("Kein Nachweis einer Ruptur des vorderen Kreuzbandes.", "ACL", 0.0, 0.05, "German")
    want("On capraz bagda yirtik izlenmedi.", "ACL", 0.0, 0.05, "Turkish")

    print("\nheading attachment — `Fractures :` / `Aucune.` is ONE statement:")
    want("Fractures :\nAucune.", "Fracture", 0.0, 0.05, "French heading on its own line")

    print("\nseverity must be ORDERED, because AUC is rank-only (§4.3):")
    sev = [label_report(t)["Effusion"].severity for t in
           ["No joint effusion.", "Trace joint effusion.", "Small joint effusion.",
            "Moderate joint effusion.", "Large joint effusion."]]
    ok = sev == sorted(sev) and len(set(sev)) == 5
    fail += 0 if ok else 1
    print(f"  [{'ok ' if ok else 'FAIL'}] effusion ladder {sev} strictly increasing, 5 distinct")

    print("\nthe rubric's own cut: mild/small NEGATIVE, moderate/large POSITIVE (§2.4):")
    want("Small joint effusion.", "Effusion", 0.10, 0.20, "rubric grades this 0")
    want("Moderate joint effusion.", "Effusion", 0.55, 0.70, "rubric grades this 1")
    want("Grade 2 chondromalacia of the patellofemoral joint.", "PF OA", 0.10, 0.20,
         "Outerbridge 2 is <50% thickness -> negative")
    want("Grade 4 chondral loss in the patellofemoral compartment.", "PF OA", 0.85, 0.95,
         "Outerbridge 4 -> positive")

    print("\ncategorical ladder for the 'what kind?' labels (§4.4):")
    cat = [label_report(t)["ACL"].severity for t in
           ["Intrasubstance degenerative signal in the ACL.",
            "Low-grade ACL sprain.",
            "Partial tear of the anterior cruciate ligament.",
            "Complete rupture of the anterior cruciate ligament."]]
    ok = cat == sorted(cat) and len(set(cat)) == 4
    fail += 0 if ok else 1
    print(f"  [{'ok ' if ok else 'FAIL'}] ACL ladder {cat} strictly increasing, 4 distinct")

    print("\nsilence ranks BELOW any mention and ABOVE an explicit negation (§5.5):")
    silent = label_report("MRI of the knee. Unremarkable patellar tendon.")["Synovitis"]
    mild = label_report("Mild synovitis.")["Synovitis"]
    neg = label_report("No synovitis.")["Synovitis"]
    ok = neg.severity < silent.severity < mild.severity
    fail += 0 if ok else 1
    print(f"  [{'ok ' if ok else 'FAIL'}] negated {neg.severity} < silent {silent.severity} "
          f"< mild {mild.severity}   (the inversion here cost 0.121 AUC on Synovitis)")

    print("\nOA context gate — a compartment mention is not automatically OA:")
    want("Tear of the medial compartment meniscus.", "Medial OA", 0.0, 0.10,
         "meniscal tear is not cartilage loss")
    want("Severe cartilage loss in the medial compartment.", "Medial OA", 0.85, 0.95, "this is")

    print("\nconfidence tracks report completeness, severity does not (§3.3):")
    short = label_report("Knie links. Erguss.")
    long_ = label_report("MRI LEFT KNEE.\nMENISCI: Normal.\nLIGAMENTS: Normal.\n"
                         "CARTILAGE: Normal.\nJOINT: Moderate effusion is present.\n"
                         "BONES: No marrow oedema. No fracture line is identified.\n"
                         "SOFT TISSUES: Unremarkable. No popliteal cyst. " + "filler word " * 40)
    ok = long_["Fracture"].confidence > short["Fracture"].confidence
    fail += 0 if ok else 1
    print(f"  [{'ok ' if ok else 'FAIL'}] unmentioned-Fracture confidence: "
          f"short={short['Fracture'].confidence} < long={long_['Fracture'].confidence}")

    print("\nthe substitution artifact must not poison grade parsing (§3.3):")
    want("Baker cyst measuring intact9xintact4cm, large.", "Baker's", 0.85, 0.95,
         "'intact9x...' must not read as a normality cue")

    print("\npresence control disagrees with severity exactly where the rubric does:")
    t = "Small joint effusion. Mild chondropathy."
    p, s = label_presence(t)["Effusion"], label_report(t)["Effusion"].severity
    ok = p == 1.0 and s < 0.3
    fail += 0 if ok else 1
    print(f"  [{'ok ' if ok else 'FAIL'}] presence={p} (calls it positive) vs "
          f"severity={s} (ranks it low) — this gap IS the thesis")

    print(f"\n{'ALL TESTS PASSED' if fail == 0 else f'{fail} TEST(S) FAILED'}")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    _tests()
