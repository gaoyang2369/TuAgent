"""Opt-in evidence retrieval features for indexed fault and alarm passages."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Protocol

from .retrieval import SearchPlan, keywords


class RewriteBackend(Protocol):
    def rewrite(self, question: str, terms: dict[str, str]) -> str: ...


class RerankBackend(Protocol):
    def score(self, question: str, documents: list[str]) -> list[float]: ...


class EvidenceServiceError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class EvidenceOptions:
    depth: int = 20
    complete_excerpts: bool = True
    rewrite: Literal["none", "fixed", "automatic"] = "none"
    rewriter: RewriteBackend | None = None
    english_vector: bool = False
    reranker: RerankBackend | None = None
    min_anchors: int = 1
    fallback: bool = False

    def __post_init__(self) -> None:
        if (
            self.depth not in (20, 40)
            or self.min_anchors not in (1, 2)
            or self.rewrite not in {"none", "fixed", "automatic"}
            or (self.rewrite == "automatic") != (self.rewriter is not None)
            or (self.english_vector and self.rewrite == "none")
        ):
            raise ValueError("Invalid manual evidence options")


# Source-triggered terminology. The mapping describes words in the question, without
# adding fault codes, inferred root causes, or corrective instructions.
_TERMS = (
    (r"直流(?:侧|母线|中间回路)|母线", "DC link"),
    (r"过电压|过压|电压.{0,6}(?:太高|偏高|过高|往上窜|升高|上升)", "overvoltage"),
    (r"欠压|电压.{0,6}(?:太低|过低|降到|跌落)", "undervoltage"),
    (r"急刹|刹车|减速|降速|斜坡下降", "ramp-down"),
    (r"回馈|再生", "regenerative"),
    (r"制动电阻", "braking resistor"),
    (r"电压上限(?:控制器|控制)?", "voltage maximum controller"),
    (r"供电|进线|配电端|交流电压|电源电压", "line supply"),
    (r"缺相|少了.{0,3}(?:相|路电)|断相", "phase failure"),
    (r"三相", "three-phase"),
    (r"保险(?:丝)?", "fuses"),
    (r"纹波", "ripple"),
    (r"不均衡|不平衡|不对称", "asymmetry"),
    (r"电机", "motor"),
    (r"堵转|卡滞|卡死|卡住|失速", "motor stalled"),
    (r"转矩|扭矩|力矩", "torque"),
    (r"转速|转得|调速", "speed"),
    (r"转速.{0,5}(?:过高|太高)|超速", "overspeed"),
    (r"过电流|过流", "overcurrent"),
    (r"过载", "overload"),
    (r"过温|超温|过热|高温", "overtemperature"),
    (r"温度|热度|太烫", "temperature"),
    (r"测温头|温度探头|温度传感器|测温探头", "temperature sensor"),
    (r"传感器", "sensor"),
    (r"接头|连接", "connection"),
    (r"内部估算|温度模型", "temperature model"),
    (r"散热片|散热器|热沉", "heat sink"),
    (r"风扇|风机", "fan"),
    (r"报警|预警|警戒", "alarm"),
    (r"跳闸|故障", "fault"),
    (r"断线|断开|中断", "interrupted"),
    (r"短路|短接", "short-circuit"),
    (r"接地", "ground fault"),
    (r"电缆", "cable"),
    (r"长度|太长|过长", "length"),
    (r"开环|U/f", "U/f operation"),
    (r"斜坡|加速", "ramp"),
    (r"外接编码器|外部编码器", "external encoder"),
    (r"整流", "rectifier"),
    (r"调试", "commissioning"),
    (r"辨识", "identification"),
    (r"重启|再启动|重新启动", "restart"),
    (r"捕捉再启动|飞车重启", "flying restart"),
    (r"启动|开机|上电|接通", "switch-on"),
    (r"停机|停机确认|停止状态", "stop"),
    (r"出厂设定", "factory setting"),
    (r"确认复位|确认故障|能确认|才能确认|可确认", "acknowledged"),
    (r"时限|超时|延时|时长|等待时间|计时|延长|监视周期", "time"),
    (r"阈值|门槛|上限|允许值", "threshold limit"),
    (r"参数", "parameter"),
    (r"控制单元", "control unit"),
    (r"功率单元|功率模块", "power unit"),
    (r"通讯|通信|总线", "bus communication"),
    (r"预充电", "precharging"),
    (r"生命信号", "sign-of-life"),
    (r"给定|设定值", "setpoint"),
    (r"处理器", "processor"),
    (r"半导体", "semiconductor"),
    (r"光纤", "fiber-optic cable"),
    (r"芯片", "chip"),
)

_ATTRIBUTES = (
    (
        r"价格|报价|收费|费用|多少钱|餐.{0,4}菜单|\b(?:cost|price|pricing|fee)\b",
        "price",
        r"\b(?:price|fee|cost)\b.{0,48}(?:[$€¥]\s*\d|\d\s*(?:EUR|USD|CNY))",
    ),
    (
        r"寿命|使用几年|\b(?:lifespan|service life)\b",
        "lifetime",
        r"\b(?:lifespan|service life)\b.{0,48}\d\s*(?:years?|hours?|cycles?)\b",
    ),
    (r"序列号|\bserial number\b", "site_serial", None),
    (
        r"昨天|昨晚|今天|最近.{0,4}(?:次|发生)|历史.{0,6}次|实际触发时间|(?:报了|发生了)几次|"
        r"\b(?:yesterday|today|telemetry)\b",
        "site_events",
        None,
    ),
    (
        r"保修|质保|\bwarranty\b",
        "warranty",
        r"\bwarranty\b.{0,48}\d\s*(?:years?|months?)\b",
    ),
    (r"库存|库房|\bin stock\b", "site_stock", None),
    (r"产量损失|财务|理赔|\binsurance\b", "site_economics", None),
    (r"密码|\b(?:password|credential)\b", "site_credentials", None),
    (
        r"预约|排班|到场|维修点地址|许可证|SHA-256|\b(?:booking|appointment)\b",
        "site_service",
        None,
    ),
    (r"IP 地址|IP 防护等级|实际 IP|\bIP address\b", "site_network_or_enclosure", None),
)
_KIND = (
    (r"原因|为什么|为何|可能.{0,6}(?:触发|导致)|\b(?:causes?|why|reasons?)\b", "cause"),
    (
        r"排查|核查|如何处理|怎么处理|处理方式|处理方法|要查|要排除|检查什么|"
        r"要检查|哪些保险|\b(?:check(?:s|ed|ing)?|remed\w*|troubleshoot\w*)\b",
        "remedy",
    ),
    (r"故障值|报警值|单位是|单位是什么|\bdiagnostic value\b", "diagnostic_value"),
)
_SHORT = re.compile(
    r"^(?:请问)?(?:是|的)?(?:有|列了|有哪些|会有|是什么|什么|可能的|可能|哪些|有哪几种|"
    r"应该|该|怎么|如何|有什么|怎么看|要)?(?:可能的?|哪些)?(?:的)?"
    r"(?:原因|排查|核查|处理|检查|处理方法)(?:是什么|有哪些)?[？?。！!]*$"
)
_PARAM = re.compile(r"(?<![a-z0-9_])[pr]\d{4}(?:\[\d+\])?(?![a-z0-9_])", re.I)
_CODE = re.compile(r"(?<![a-z0-9_])[FA]\d{5}(?![a-z0-9_])", re.I)
_NUMBER = re.compile(
    r"(?<![\w.])[+−-]?\d+(?:\.\d+)?(?![\d.])|(?<=[\u3400-\u9fff])[+−-]?\d+(?:\.\d+)?(?![\d.])"
)
_UNITS = re.compile(
    r"(?i)\b(?:ms|s|A|V|Hz|kW|rpm|ohm|K)\b|°\s*C|%|毫秒|秒|伏|安培|摄氏度|欧姆"
)
_NEGATION = re.compile(
    r"[不未无没]|\b(?:no|not|never|without|cannot|isn't|doesn't|lacks?)\b",
    re.I,
)
_COMPARE = re.compile(
    r"比较|对比|分别|区别|还是|相较|\b(?:compare|versus|vs|difference|rather than)\b",
    re.I,
)
_COMPARE_CONDITION = re.compile(
    r"门槛|阈值|判据|条件|触发|断开|短路|threshold|condition|open circuit|short.circuit",
    re.I,
)
_CHINESE_QUANTITY = re.compile(
    r"[零一二两三四五六七八九十百千万点]+\s*(?:毫秒|秒|伏|安培|摄氏度|欧姆|年|次)"
)
_ALARM_ONLY = re.compile(r"哪个报警|哪条报警|什么报警|which alarm", re.I)
_FAULT_ONLY = re.compile(r"哪个故障|哪条故障|什么故障|which fault", re.I)
_CJK = re.compile(r"[\u3400-\u9fff]")
_COMMON_ANCHORS = {
    "drive",
    "unit",
    "power",
    "fault",
    "alarm",
    "motor",
    "check",
    "time",
    "parameter",
    "possible",
    "could",
    "too",
}
_INFLECTIONS = {
    "long": "length",
    "lengths": "length",
    "cables": "cable",
    "fuses": "fuse",
    "stalled": "stall",
    "stalling": "stall",
    "blocked": "block",
    "regenerates": "regenerat",
    "regenerative": "regenerat",
    "regeneration": "regenerat",
    "phases": "phase",
    "interruptions": "interrupt",
    "interrupted": "interrupt",
    "interruption": "interrupt",
    "restarts": "restart",
    "attempts": "attempt",
    "acknowledged": "acknowledge",
    "acknowledgment": "acknowledge",
    "acknowledgement": "acknowledge",
    "temperatures": "temperature",
    "sensors": "sensor",
    "settings": "setting",
    "controllers": "controller",
    "voltages": "voltage",
    "over-temperature": "overtemperature",
    "over-voltage": "overvoltage",
    "short-circuited": "short-circuit",
}
_UNIT_ENGLISH = {
    "毫秒": "ms",
    "秒": "s",
    "伏": "v",
    "安培": "a",
    "摄氏度": "°c",
    "欧姆": "ohm",
}
_ENGLISH_GLUE = set(
    (
        "what which how why does do did is are was were can could may should will would the a an "
        "at by for from of on in into or and to exceed too only between cause causes reason "
        "trigger triggers mean value found say described documented manual about find check checked against when if no "
        "not without cannot never compare versus separately"
    ).split()
)
_TECHNICAL_WORD = re.compile(r"[A-Za-z][A-Za-z0-9/-]+")


def _features(text: str) -> set[str]:
    return (
        {_INFLECTIONS.get(token, token) for token in keywords(text) if token.isascii()}
        - _COMMON_ANCHORS
        - _ENGLISH_GLUE
    )


class QuestionEvidence:
    def __init__(self, query: str):
        self.query = unicodedata.normalize("NFKC", query)
        self.terms = {p: term for p, term in _TERMS if re.search(p, self.query, re.I)}
        self.attribute, self.required_pattern = next(
            (
                (name, proof)
                for regex, name, proof in _ATTRIBUTES
                if re.search(regex, self.query, re.I)
            ),
            (None, None),
        )
        self.kind = next(
            (kind for regex, kind in _KIND if re.search(regex, self.query, re.I)), None
        )
        self.params = set(_PARAM.findall(self.query.casefold()))
        self.numbers = set(_NUMBER.findall(_PARAM.sub("", _CODE.sub("", self.query))))
        self.literals = tuple(
            dict.fromkeys(
                word
                for word in _TECHNICAL_WORD.findall(self.query)
                if word.casefold() not in _ENGLISH_GLUE
            )
        )
        self.features = _features(self.query + " " + " ".join(self.terms.values()))
        self.short = bool(_SHORT.fullmatch(self.query.strip()))
        self.negative = bool(_NEGATION.search(self.query))
        self.comparison = bool(_COMPARE.search(self.query))
        if (
            self.kind is None
            and self.comparison
            and _COMPARE_CONDITION.search(self.query)
        ):
            self.kind = "cause"

    def fixed_rewrite(self) -> str | None:
        if not _CJK.search(self.query) or len(self.terms) < 2 or self.attribute:
            return None
        # A fragment rewrite cannot preserve the attachment of these operators.
        if (
            self.negative
            or self.comparison
            or _NUMBER.search(self.query)
            or _CHINESE_QUANTITY.search(self.query)
            or self.params
            or _UNITS.search(self.query)
        ):
            return None
        prefix = {
            "cause": "What causes",
            "remedy": "What should be checked for",
            "diagnostic_value": "What diagnostic value is documented for",
        }.get(self.kind, "What does the manual say about")
        translated = " ".join(dict.fromkeys((*self.terms.values(), *self.literals)))
        return f"{prefix} {translated}?"

    def validate_rewrite(self, rewritten: str) -> bool:
        if (
            not rewritten
            or len(rewritten) > 500
            or _CJK.search(rewritten)
            or "\n" in rewritten
            or rewritten.count("?") != 1
            or not rewritten.endswith("?")
        ):
            return False
        if (
            not self.terms
            or self.attribute
            or self.negative
            or self.comparison
            or _NEGATION.search(rewritten)
            or _COMPARE.search(rewritten)
        ):
            return False
        # Arbitrary additions cannot pass the vocabulary check. The source phrase
        # mappings are required in full rather than accepting related symptoms.
        source = self.query.casefold()
        target = rewritten.casefold()
        for pattern in (_CODE, _PARAM, _NUMBER):
            if sorted(pattern.findall(source)) != sorted(pattern.findall(target)):
                return False
        if not {word.casefold() for word in self.literals} <= set(
            _TECHNICAL_WORD.findall(target)
        ):
            return False
        source_units = sorted(_UNIT_ENGLISH.get(u, u) for u in _UNITS.findall(source))
        if source_units != sorted(_UNITS.findall(target)):
            return False
        allowed = keywords(source + " " + " ".join(self.terms.values())) | _ENGLISH_GLUE
        if not keywords(target) <= allowed:
            return False
        target_features = _features(target)
        return (
            all(_features(term) <= target_features for term in self.terms.values())
            and QuestionEvidence(rewritten).kind == self.kind
        )


class EvidenceRulePlan(SearchPlan):
    def __init__(
        self,
        question: QuestionEvidence,
        code_filter: bool,
        comparison: bool,
        rewrite: str | None = None,
    ):
        # Decode only the colloquial DC-link query deterministically in this arm.
        original = question.query
        if (
            re.search(r"刹车|急刹", original)
            and re.search(r"直流侧|母线", original)
            and re.search(r"电压往上窜|电压蹿高", original)
        ):
            original += " 减速 母线过压 电压升高 原因"
        super().__init__(
            f"{original} {rewrite or ''}".strip(),
            True,
            code_filter=code_filter,
            comparison=comparison,
        )
        self.params = question.params
        if question.kind:
            self.intent = question.kind
        if code_filter and question.short and self.intent:
            self.intent_only = True

    def rank(
        self, code: str, title: str, kind: str, text: str, *, code_filter: bool
    ) -> float:
        score = super().rank(code, title, kind, text, code_filter=code_filter)
        literal = set(_PARAM.findall(text.casefold())) & self.params
        return score + 10 * len(literal)


_BULLET = re.compile(r"(?<=\s)(?=-(?:\s*[A-Za-z])|\d{1,2}\)\s)")
_SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")
_DIAGNOSIS_HEADER = re.compile(
    r"^(?:The .+ has detected|A .+ was detected|Possible causes:)", re.I
)


def segments(text: str) -> list[str]:
    return [
        segment.strip().removeprefix("-").strip()
        for item in _BULLET.split(text)
        for segment in (
            [item] if re.match(r"\s*(?:-|\d{1,2}\))", item) else _SENTENCE.split(item)
        )
        if segment.strip()
    ]


def select_evidence(
    question: QuestionEvidence,
    title: str,
    code: str,
    kind: str,
    text: str,
    *,
    scoped: bool,
    min_anchors: int,
) -> tuple[str | None, str]:
    if question.attribute and not question.required_pattern:
        return None, f"missing_{question.attribute}_evidence"
    if (
        question.kind
        and kind != question.kind
        and not question.params
        and not question.attribute
    ):
        return None, "wrong_passage_type"
    if (_ALARM_ONLY.search(question.query) and not code.startswith("A")) or (
        _FAULT_ONLY.search(question.query) and not code.startswith("F")
    ):
        return None, "wrong_alarm_fault_type"
    topical = question.features
    heading = _features(title) & topical
    bare_scope = scoped and (
        question.short
        or not question.query
        or (question.comparison and not topical and kind == "title")
    )
    choices = []
    parts = segments(text)
    passage_params = set(_PARAM.findall(text.casefold()))
    if question.params and not question.params & passage_params:
        return None, "requested_parameter_absent"
    list_parts = _BULLET.split(text)
    if (
        kind == "remedy"
        and len(list_parts) == 2
        and not question.params
        and not question.numbers
        and text.strip().startswith("-")
        and all(len(item.split()) <= 12 for item in list_parts)
    ):
        parts = [text.strip(), *parts]
    for index, excerpt in enumerate(parts):
        if len(excerpt) > 280 or len(excerpt.split()) > 24:
            continue
        if question.required_pattern and not re.search(
            question.required_pattern, excerpt, re.I
        ):
            continue
        if (
            question.params
            and not (kind == "diagnostic_value" and question.numbers)
            and not question.params & set(_PARAM.findall(excerpt.casefold()))
        ):
            continue
        if (
            question.numbers
            and kind == "diagnostic_value"
            and not question.numbers & set(_NUMBER.findall(_PARAM.sub("", excerpt)))
        ):
            continue
        if (
            question.params
            and kind == "cross_reference"
            and not re.search(r"[pr]\d{4}(?:\[\d+\])?\s*\([^)]*[A-Za-z]", excerpt, re.I)
        ):
            continue
        anchors = _features(excerpt) & topical
        braking_evidence = (
            kind == "cause"
            and "ramp-down" in topical
            and "regenerat" in _features(excerpt)
            and not question.negative
        )
        if braking_evidence:
            anchors.add("ramp-down")
        if not (bare_scope or question.params or question.required_pattern) and (
            len(anchors | heading) < min_anchors
            or (
                not anchors
                and kind != "title"
                and not (kind == "cause" and heading and len(list_parts) > 1)
            )
        ):
            continue
        # Prefer the clause with body-level anchors, preserving any negative
        # wording, parameter, numeric threshold, and units in the source.
        score = 3 * len(anchors) + len(heading) + (12 if braking_evidence else 0)
        if (
            question.kind == "cause"
            and len(list_parts) > 1
            and _DIAGNOSIS_HEADER.search(excerpt)
        ):
            score -= 12
        choices.append((score, -index, excerpt))
    if not choices:
        return None, (
            f"missing_{question.attribute}_evidence"
            if question.attribute
            else "no_complete_answer_item"
        )
    return max(choices)[2], "accepted"
