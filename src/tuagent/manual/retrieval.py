import re
import unicodedata
from collections.abc import Iterator


_WORDS = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[\u3400-\u9fff]+")
_COMMON = set(
    (
        "a an the to of is was are on in by at for it or and as has have this that can "
        "be been with from not if when while what which why how does do may could would "
        "reach find show manual check checks cause causes possible remedy remedies "
        "fault alarm list described happen troubleshooting occurs"
    ).split()
)
_CONCEPT_PATTERNS = {
    "dc_link": r"\bdc[ -](?:link|bus)\b|母线|直流中间回路",
    "overvoltage": r"\bover[ -]?voltage\b|电压(?:升高|过高|上升)|过电压|过压",
    "undervoltage": r"\bunder[ -]?voltage\b|欠压|电压过低",
    "deceleration": r"\bdecelerat\w*\b|\bramp[ -]down\b|\bslow(?:ing|s)? down\b|减速|降速",
    "regeneration": r"\bregenerat\w*\b|\benergy feedback\b|回馈|再生制动|发电制动",
    "braking_resistor": r"\b(?:brake|braking) resist\w*\b|制动电阻",
    "supply_voltage": r"\b(?:line|mains|supply)(?: supply)? voltage\b|供电电压|电网电压",
    "phase_loss": r"\b(?:line |supply )?phase (?:interrupted|failure|loss)\b|缺相|断相",
    "motor_overtemperature": (
        r"\bmotor (?:over[ -]?temperature|overheat\w*)\b|\bmotor (?:ambient )?temperature\b"
        r"|电机(?:过热|超温|温升|温度)"
    ),
    "blocked_motor": r"\bmotor (?:is )?(?:block\w*|stall\w*)\b|\b(?:blocked|stalled) motor\b|堵转|卡滞",
    "torque_limit": r"\btorque limit\w*\b|力矩限值|转矩限值|转矩上限",
    "overload": r"\boverload\w*\b|过载",
    "temperature_sensor": r"\btemperature sensor\b|\bKTY84\b|\bPT1000\b|温度传感器",
}
_CONCEPTS = {
    name: re.compile(pattern, re.I) for name, pattern in _CONCEPT_PATTERNS.items()
}
_RELATED = {"deceleration": {"regeneration"}, "regeneration": {"deceleration"}}
_OPPOSITES = {"overvoltage": "undervoltage", "undervoltage": "overvoltage"}
_INTENTS = (
    (re.compile(r"\b(?:cause[sd]?|why|reasons?)\b|原因|为什么"), "cause"),
    (
        re.compile(r"\b(?:remed\w*|check\w*|troubleshoot\w*)\b|核查|检查|处理方法"),
        "remedy",
    ),
)


def keywords(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return {
        word
        for word in _WORDS.findall(normalized)
        if len(word) >= 2 and word not in _COMMON
    }


def concepts(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return {name for name, pattern in _CONCEPTS.items() if pattern.search(normalized)}


def index_terms(text: str, title: str, code: str) -> Iterator[str]:
    for word in sorted(keywords(f"{code} {title} {text}")):
        yield f"kw:{word}"
    for concept in sorted(concepts(f"{title} {text}")):
        yield f"concept:{concept}"


class SearchPlan:
    def __init__(self, query: str, semantic: bool) -> None:
        self.keywords = set(
            sorted(keywords(query), key=lambda word: (-len(word), word))[:32]
        )
        self.concepts = concepts(query) if semantic else set()
        self.related = (
            set().union(*(_RELATED.get(name, set()) for name in self.concepts))
            - self.concepts
        )
        self.intent = next(
            (kind for pattern, kind in _INTENTS if pattern.search(query.casefold())),
            None,
        )
        self.intent_only = bool(
            self.intent
            and (
                not self.keywords
                or query.strip(" ?？。")
                in {
                    "原因",
                    "有哪些原因",
                    "什么原因",
                    "核查",
                    "如何排查",
                    "检查",
                }
            )
        )

    @property
    def terms(self) -> tuple[str, ...]:
        return (
            *(f"kw:{word}" for word in sorted(self.keywords)),
            *(f"concept:{name}" for name in sorted(self.concepts | self.related)),
        )

    def rank(
        self, code: str, title: str, kind: str, text: str, *, code_filter: bool
    ) -> float:
        title_keywords = keywords(f"{code} {title}")
        body_keywords = keywords(text)
        literal = self.keywords & (title_keywords | body_keywords)
        title_concepts = concepts(title) if self.concepts else set()
        body_concepts = concepts(text) if self.concepts else set()
        direct = self.concepts & (title_concepts | body_concepts)
        related = self.related & body_concepts
        explicit_code = code.casefold() in self.keywords
        intent_only = code_filter and kind == self.intent and self.intent_only
        if not (
            direct
            or explicit_code
            or intent_only
            or len(literal) >= 2
            or (len(self.keywords) == 1 and literal and len(next(iter(literal))) >= 4)
        ):
            return 0
        if any(_OPPOSITES.get(name) in title_concepts for name in self.concepts):
            return 0
        score = (
            4 * len(direct)
            + 2 * len(direct & title_concepts)
            + len(direct & body_concepts)
            + 2 * len(related)
            + 2 * len(literal & body_keywords)
            + len(literal & title_keywords)
        )
        return score + (4 if self.intent == kind else 0) + (3 if explicit_code else 0)
