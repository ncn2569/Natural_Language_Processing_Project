"""Phan II (Classical NLP): nhan dien entity, intent va bieu dien ngu nghia.

Pipeline: cau (+ cay cu phap tu Phan I) -> entities -> intent -> semantic frame.
Intent/entity dung luat + tu dien (entities.txt) + keyword cue; khong dung LLM.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Nhan nguon cua cay cu phap (Phan I) -> intent. Sua theo grammar cua nhom.
TREE_LABEL_TO_INTENT = {
    "Q_INFO": "GET_COURSE_INFO",
    "Q_TOPIC": "GET_TOPIC",
    "Q_SCHEDULE": "GET_SCHEDULE",
    "Q_ASSIGN": "GET_ASSIGNMENT",
    "Q_DEADLINE": "GET_DEADLINE",
    "Q_RES": "GET_RESOURCE",
    "Q_RULE": "GET_RULE",
    "Q_LO": "GET_LO",
    "Q_INSTRUCTOR": "GET_INSTRUCTOR",
}

ROMAN = {
    "i": "I",
    "ii": "II",
    "iii": "III",
    "iv": "IV",
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
}

# Chu de dung de tim tai lieu (khong co trong entities.txt)
EXTRA_TOPICS = {
    "STATISTICAL_NLP": ["statistical nlp", "nlp thống kê", "statistical methods"],
    "PYTHON_NLP": ["python nlp", "nlp with python", "python"],
    "PARSING": ["parsing"],
}
# Tu khoa tim trong KB (ngoai ten chuan va ve phai tieng Anh)
EXTRA_SEARCH = {
    "QUESTION_ANSWERING": ["QA"],
    "PARSING": ["parsing"],
    "WSD": ["WSD"],
    "ANAPHORA": ["anaphora"],
    "REFERENCE": ["reference"],
}

PRONOUN_RE = re.compile(
    r"\bnó\b|\b(?:phần|chương|nội dung|mục|bài|tuần|chủ đề|vấn đề|kiến thức)\s+(?:này|đó|ấy|sau|trước|tiếp theo|tiếp)\b|^\s*(?:thế\s+|vậy\s+)?còn\b"
)


def norm(s):
    return unicodedata.normalize("NFC", s).strip().lower()


@dataclass
class Entity:
    type: str
    value: str
    text: str
    start: int
    end: int


@dataclass
class Frame:
    sentence: str
    intent: str
    predicate: str
    args: dict
    entities: list = field(default_factory=list)
    primary: str = "-"
    yes_no: bool = False
    attr: str = ""
    intent_source: str = "rules"
    parse_status: str = "MISSING"
    notes: list = field(default_factory=list)

    def semantic(self):
        a = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.predicate}({a})"


# ---------------------------------------------------------------- cay cu phap
def parse_bracket(line):
    """'(S (Q_TOPIC tuan (WEEK 5) ...))' -> (label, [children]); '()' hoac rong -> None."""
    line = line.strip()
    if not line or line == "()":
        return None
    toks = re.findall(r"\(|\)|[^\s()]+", line)
    pos = 0

    def node():
        nonlocal pos
        pos += 1  # '('
        label = toks[pos]
        pos += 1
        kids = []
        while toks[pos] != ")":
            if toks[pos] == "(":
                kids.append(node())
            else:
                kids.append(toks[pos])
                pos += 1
        pos += 1  # ')'
        return (label, kids)

    try:
        return node()
    except IndexError:
        return None


# def tree_intent(tree):
#     stack = [tree]
#     while stack:
#         label, kids = stack.pop(0)
#         if label in TREE_LABEL_TO_INTENT:
#             return TREE_LABEL_TO_INTENT[label]
#         stack.extend(k for k in kids if isinstance(k, tuple))
#     return None


def collect_tree_features(tree):
    """Rút trích tập nhãn nonterminal và từ vựng lá theo từng loại nút từ cây cú pháp Phần I."""
    labels = set()
    by_label = {}
    stack = [tree]
    while stack:
        label, kids = stack.pop(0)
        labels.add(label)
        leaves = [k for k in kids if isinstance(k, str)]
        if leaves:
            by_label.setdefault(label, []).append(" ".join(leaves))
        stack.extend(k for k in kids if isinstance(k, tuple))
    return labels, by_label


def tree_intent(tree):
    """Suy diễn Intent từ các nhánh cú pháp của cây Phần I (Syntactic -> Semantic)."""
    labels, by_label = collect_tree_features(tree)
    # 1. Nếu vẫn dùng grammar có nhãn Q_*
    for lb in labels:
        if lb in TREE_LABEL_TO_INTENT:
            return TREE_LABEL_TO_INTENT[lb]
    # 2. Đọc theo cấu trúc Syntactic Grammar của nhóm
    nouns = set(
        by_label.get("Noun", [])
        + by_label.get("IndexedNoun", [])
        + by_label.get("UnitNoun", [])
    )
    verbs = set(by_label.get("ObjectVerb", []) + by_label.get("IntransitiveVerb", []))
    if "OutcomeName" in by_label:
        return "GET_LO"
    if nouns & {"deadline", "hạn nộp", "thời hạn nộp"} or (
        "WhTime" in labels and "nộp" in verbs
    ):
        return "GET_DEADLINE"
    if "giảng viên" in nouns or "ai" in by_label.get("WhPronoun", []):
        return "GET_INSTRUCTOR"
    if nouns & {"tài liệu", "tài liệu tham khảo", "giáo trình"}:
        return "GET_RESOURCE"
    if (
        "ToolName" in by_label
        or nouns & {"quy định", "công cụ ai", "thư viện công khai"}
        or "được phép" in by_label.get("Modal", [])
    ):
        return "GET_RULE"
    if "AssignmentName" in by_label or "bài tập lớn" in nouns or (
        "file" in nouns and "xuất" in verbs
    ):
        return "GET_ASSIGNMENT"
    if "ExamName" in by_label or nouns & {
        "tín chỉ",
        "phút",
        "giờ",
        "thi giữa kỳ",
        "thi cuối kỳ",
        "phần chính",
        "thông tin",
    }:
        return "GET_COURSE_INFO"
    if "tuần" in nouns or "tuần" in by_label.get("TimeNoun", []) or "lịch học" in nouns:
        return "GET_SCHEDULE"
    if "TopicName" in by_label or nouns & {
        "chương",
        "nội dung",
        "chủ đề",
        "phương pháp",
    }:
        return "GET_TOPIC"
    if "CourseName" in by_label:
        return "GET_COURSE_INFO"
    return None


# ---------------------------------------------------------------- entity
class EntityLinker:
    def __init__(self, kb):
        self.kb = kb
        self.rules = []  # (compiled_regex, type, value_fn)
        add = self._add
        add(
            r"\b(?:chương|chapter|ch)\s*(\d{1,2})\b",
            "CHAPTER",
            lambda m: "CH%02d" % int(m.group(1)),
        )
        add(
            r"\b(?:tuần|week)\s*(\d{1,2})\b",
            "WEEK",
            lambda m: "WEEK_%02d" % int(m.group(1)),
        )
        add(
            r"\bl\.?o\.?\s*(\d)\.(\d)\b",
            "LO",
            lambda m: "LO%s.%s" % (m.group(1), m.group(2)),
        )
        add(r"\bl\.?o\.?\s*(\d)\b(?![.\d])", "LO", lambda m: "LO%s" % m.group(1))
        add(r"\bllm\s*/?\s*rag\b|\bllm\b|\brag\b", "RULE", lambda m: "LLM_RAG")
        add(
            r"(?:sử dụng|dùng|công cụ|khai báo)\s+(?:công cụ\s+)?ai\b|\bchatgpt\b",
            "RULE",
            lambda m: "AI_USAGE",
        )
        add(r"\bmidterm\b|giữa kỳ|giữa kì", "EXAM", lambda m: "MIDTERM")
        add(r"\bfinal\b|cuối kỳ|cuối kì", "EXAM", lambda m: "FINAL")
        add(r"bài tập lớn|\bbtl\s*0?1?\b|major assignment", "ASSIGN", lambda m: "BTL01")
        add(
            r"\b(?:phần|part)\s*(iv|iii|ii|i|[1-4])\b",
            "PARTREF",
            lambda m: ROMAN[m.group(1)],
        )
        add(
            r"co3085|\bnlp\b|xử lý ngôn ngữ tự nhiên|natural language processing|môn học|\bmôn\b",
            "COURSE",
            lambda m: "CO3085",
        )
        # chu de tu entities.txt (+ extra)
        for name, rhs in kb.entities.get("TOPICS", []):
            val = re.sub(r"\W+", "_", name.upper()).strip("_")
            aliases = {name.lower()}
            for a in re.split(r"\s*/\s*", rhs):
                if len(a) >= 4:
                    aliases.add(a.lower())
            self._topic(val, aliases)
        for val, al in EXTRA_TOPICS.items():
            self._topic(
                val,
                set(al),
                rtype="RESOURCE_TOPIC"
                if val in ("STATISTICAL_NLP", "PYTHON_NLP")
                else "TOPIC",
            )
        # tim kiem trong KB
        self.search_terms = {}
        for name, rhs in kb.entities.get("TOPICS", []):
            val = re.sub(r"\W+", "_", name.upper()).strip("_")
            terms = [name]
            terms += [
                a
                for a in re.split(r"\s*/\s*", rhs)
                if re.search(r"[a-z]", a) and a.isascii()
            ]
            self.search_terms[val] = terms + EXTRA_SEARCH.get(val, [])
        for val, al in EXTRA_SEARCH.items():
            self.search_terms.setdefault(val, al)

    def _add(self, pat, typ, fn):
        self.rules.append((re.compile(pat), typ, fn))

    def _topic(self, val, aliases, rtype="TOPIC"):
        pat = "|".join(
            r"(?<!\w)%s(?!\w)" % re.escape(a)
            for a in sorted(aliases, key=len, reverse=True)
        )
        self.rules.append((re.compile(pat), rtype, lambda m, v=val: v))

    def extract(self, text):
        t = norm(text)
        cands = []
        for rx, typ, fn in self.rules:
            for m in rx.finditer(t):
                cands.append(
                    (
                        m.end() - m.start(),
                        m.start(),
                        Entity(typ, fn(m), t[m.start() : m.end()], m.start(), m.end()),
                    )
                )
        cands.sort(key=lambda c: (-c[0], c[1]))
        taken, out = [], []
        for _, _, e in cands:
            if any(not (e.end <= s or e.start >= f) for s, f in taken):
                continue
            taken.append((e.start, e.end))
            out.append(e)
        return sorted(out, key=lambda e: e.start)


# ---------------------------------------------------------------- intent
CUES = {
    "deadline": r"deadline|hạn nộp|hạn chót|thời hạn|khi nào (?:phải )?nộp|ngày nộp|nộp trước|mốc|checkpoint",
    "instructor": r"giảng viên|giáo viên|\bthầy\b|\bcô giáo\b|ai dạy|người dạy",
    "resource": r"tài liệu|giáo trình|\bsách\b|tham khảo|textbook",
    "rule": r"quy định|nội quy|có được|được phép|cho phép|cấm|điểm danh|liêm chính|khai báo|gian lận|sao chép|chia sẻ",
    "course": r"tín chỉ|\bects\b|môn học gồm|phần nào|những phần|các phần|chính nào|thành phần điểm|trọng số|bao nhiêu phút|mấy phút|thời lượng|bao lâu|\bkhoa\b|học kỳ|học kì",
    "schedule": r"\btuần\b|\blịch\b|tiết|thứ mấy|\bweek\b",
    "topic": r"\bchương\b|nói về|học về|nội dung|phương pháp|liên quan đến|được học|học ở đâu|trình bày",
    "lo": r"chuẩn đầu ra|mục tiêu học|learning outcome",
}
CUES = {k: re.compile(v) for k, v in CUES.items()}


def classify(text, entities, parse_intent=None):
    """Tra ve (intent, ly do). Luat uu tien co thu tu."""
    t = norm(text)
    types = {e.type for e in entities}
    has = lambda k: bool(CUES[k].search(t))
    primary_types = {
        "CHAPTER",
        "WEEK",
        "LO",
        "ASSIGN",
        "PARTREF",
        "COURSE",
        "EXAM",
        "RULE",
    }
    if PRONOUN_RE.search(t) and not (types & primary_types):
        return "CONTEXT_DEPENDENT", "co dai tu/tham chieu, khong co entity chinh"
    domain = bool(types) or any(has(k) for k in CUES)
    if not domain:
        return "UNKNOWN", "khong co tu khoa/entity thuoc mien mon hoc"
    if "LO" in types or has("lo"):
        return "GET_LO", "entity LO / cue chuan dau ra"
    assign_ctx = "ASSIGN" in types
    if has("deadline"):
        return "GET_DEADLINE", "cue deadline/han nop"
    if has("instructor"):
        return "GET_INSTRUCTOR", "cue giang vien"
    if has("resource"):
        return "GET_RESOURCE", "cue tai lieu"
    if has("rule") or "RULE" in types:
        return "GET_RULE", "cue quy dinh / entity RULE"
    if assign_ctx or ("PARTREF" in types and re.search(r"\b(?:file|output|xuất|nộp)\b", t)):
        return "GET_ASSIGNMENT", "entity bai tap lon"
    if "EXAM" in types or has("course"):
        return "GET_COURSE_INFO", "cue thong tin mon hoc / ky thi"
    if has("schedule") or "WEEK" in types:
        return "GET_SCHEDULE", "cue tuan/lich"
    if has("topic") or types & {"CHAPTER", "TOPIC", "PARTREF"}:
        return "GET_TOPIC", "cue chuong/chu de"
    if "COURSE" in types:
        return "GET_COURSE_INFO", "entity mon hoc"
    return "UNKNOWN", "khong khop luat nao"


# ---------------------------------------------------------------- semantic
PRIMARY_ORDER = [
    "CHAPTER",
    "WEEK",
    "LO",
    "PART",
    "ASSIGN",
    "RULE",
    "EXAM",
    "TOPIC",
    "RESOURCE_TOPIC",
    "COURSE",
]

COURSE_ATTRS = [
    ("credits", r"tín chỉ"),
    ("ects", r"\bects\b"),
    ("hours", r"số giờ|tổng giờ|bao nhiêu giờ"),
    ("exam", r"bao nhiêu phút|mấy phút|thời lượng|bao lâu|hình thức thi"),
    ("assessment", r"đánh giá|điểm|trọng số|thành phần"),
    ("parts", r"phần nào|những phần|các phần|gồm|chính nào|main part"),
    ("department", r"\bkhoa\b"),
    ("semester", r"học kỳ|học kì"),
    ("format", r"hình thức|mấy tiết|tiết"),
    ("description", r"mô tả|giới thiệu|là gì"),
]
ASSIGN_PART_CUES = [
    ("I", r"parser|grammar|văn phạm|cú pháp|sinh câu|phân tích cú pháp"),
    ("II", r"hỏi đáp|ngữ nghĩa|semantic|\bqa\b|intent|entity|knowledge base|rag"),
    ("III", r"ngữ cảnh|hội thoại|diễn ngôn|dialogue"),
    ("IV", r"đánh giá|evaluation|kiểm thử"),
]
RULE_CUES = [
    ("attendance", r"điểm danh|tham dự|vắng|nghỉ"),
    ("assignment_sharing", r"sao chép|chia sẻ|copy"),
    ("external_libraries", r"thư viện"),
    ("academic_integrity", r"liêm chính|gian lận|đạo văn"),
    ("source_code", r"mã nguồn|source code|giải thích code"),
]


def build_frame(sentence, linker, tree_state=("MISSING", None)):
    status, tree = tree_state
    t = norm(sentence)
    ents = linker.extract(sentence)
    by = {}
    for e in ents:
        by.setdefault(e.type, []).append(e)

    # phan I/II/... : bai tap lon hay phan cua mon hoc
    part_val = None
    if "PARTREF" in by:
        n = by["PARTREF"][0].value
        is_assign_part = "ASSIGN" in by or bool(re.search(r"\b(?:file|output|xuất|nộp)\b", t))
        part_val = ("PART_" if is_assign_part else "COURSE_PART_") + n
        by.setdefault("PART", []).append(
            Entity("PART", part_val, by["PARTREF"][0].text, 0, 0)
        )
    elif "ASSIGN" in by:
        for roman, cue in ASSIGN_PART_CUES:
            if re.search(cue, t):
                part_val = "PART_" + roman
                by.setdefault("PART", []).append(Entity("PART", part_val, cue, 0, 0))
                break

    intent, why = classify(sentence, ents, tree_intent(tree) if tree else None)
    frame = Frame(
        sentence=sentence,
        intent=intent,
        predicate="",
        args={},
        entities=ents,
        parse_status=status,
    )
    frame.notes.append("intent: " + why)
    # doi chieu nhan tu cay cu phap
    if tree:
        ti = tree_intent(tree)
        if ti:
            frame.intent_source = (
                "rules+tree(agree)" if ti == intent else f"rules(tree={ti} khac)"
            )
    elif status == "REJECTED":
        frame.intent_source = "rules(parser tu choi cau)"

    # frame.yes_no = bool(re.search(r"\bcó\b.*\bkhông\b\s*\??\s*$", t)) and not re.search(r"có (?:bao nhiêu|mấy|những)", t)
    tree_labels = collect_tree_features(tree)[0] if tree else set()
    frame.yes_no = ("YesNoQuestion" in tree_labels) or (
        bool(re.search(r"\bcó\b.*\bkhông\b\s*\??\s*$", t))
        and not re.search(r"có (?:bao nhiêu|mấy|những)", t)
    )

    def first(*types):
        for ty in types:
            if ty in by:
                return by[ty][0].value
        return None

    # primary entity (de danh gia)
    prim = None
    for ty in PRIMARY_ORDER:
        if ty in by:
            prim = by[ty][0].value
            break
    a = frame.args

    if intent == "GET_COURSE_INFO":
        exam = first("EXAM")
        course = first("COURSE") or "CO3085"
        attr = next((n for n, rx in COURSE_ATTRS if re.search(rx, t)), "")
        if exam:
            attr = "exam"
            frame.predicate, a["exam"] = (
                "GET_EXAM_DURATION"
                if re.search(r"phút|bao lâu|thời lượng", t)
                else "GET_EXAM_FORMAT",
                exam,
            )
        else:
            frame.predicate = {
                "credits": "GET_CREDIT",
                "ects": "GET_ECTS",
                "hours": "GET_HOURS",
                "assessment": "GET_ASSESSMENT",
                "parts": "GET_MAIN_PARTS",
                "department": "GET_DEPARTMENT",
                "semester": "GET_SEMESTER",
                "format": "GET_FORMAT",
                "description": "GET_COURSE_DESCRIPTION",
                "exam": "GET_EXAM_FORMAT",
            }.get(attr, "GET_COURSE_INFO")
            a["course"] = course
        frame.attr = attr
        prim = exam or course
    elif intent == "GET_SCHEDULE":
        if first("WEEK"):
            frame.predicate, a["week"] = "GET_WEEK_CONTENT", first("WEEK")
        elif first("CHAPTER"):
            frame.predicate, a["chapter"] = "GET_CHAPTER_WEEKS", first("CHAPTER")
        elif first("TOPIC"):
            frame.predicate, a["topic"] = "GET_TOPIC_WEEKS", first("TOPIC")
        else:
            frame.predicate, a["course"] = "GET_SCHEDULE", "CO3085"
    elif intent == "GET_TOPIC":
        if first("CHAPTER") and first("TOPIC"):
            frame.predicate, a["chapter"], a["topic"] = (
                "CHECK_CHAPTER_COVERS",
                first("CHAPTER"),
                first("TOPIC"),
            )
        elif first("CHAPTER"):
            frame.predicate, a["chapter"] = "GET_CHAPTER_CONTENT", first("CHAPTER")
        elif first("TOPIC"):
            frame.predicate, a["topic"] = "GET_TOPIC_CHAPTER", first("TOPIC")
        elif part_val and part_val.startswith("COURSE_PART_"):
            frame.predicate, a["part"] = "GET_PART_CHAPTERS", part_val
        else:
            frame.predicate, a["course"] = "GET_TOPIC", "CO3085"
    elif intent == "GET_LO":
        lo = first("LO")
        if lo and re.search(r"chương\s+nào|ở\s+chương|thuộc\s+chương", t):
            frame.predicate, a["lo"] = "GET_LO_CHAPTER", lo
        else:
            frame.predicate, a["lo"] = ("GET_LO", lo) if lo else ("GET_LO_ALL", "ALL")
        prim = lo or prim
    elif intent == "GET_ASSIGNMENT":
        if part_val and part_val.startswith("PART_"):
            if frame.yes_no:
                frame.predicate = "CHECK_ASSIGNMENT_PART"
            else:
                frame.predicate = "GET_ASSIGNMENT_PART"
            a["part"] = part_val
            prim = part_val
        else:
            frame.predicate, a["assignment"] = "GET_ASSIGNMENT", "BTL01"
    elif intent == "GET_DEADLINE":
        frame.predicate, a["assignment"] = "GET_DEADLINE", "BTL01"
        prim = prim or "BTL01"
    elif intent == "GET_RESOURCE":
        rt = first("RESOURCE_TOPIC")
        topic = first("TOPIC")
        ch = first("CHAPTER")
        if rt:
            frame.predicate, a["topic"] = "GET_RESOURCE", rt
            prim = rt
        elif topic:
            frame.predicate, a["topic"] = "GET_RESOURCE", topic
            prim = topic
        elif ch:
            frame.predicate, a["chapter"] = "GET_RESOURCE", ch
            prim = ch
        else:
            frame.predicate, a["scope"] = "GET_RESOURCE", "ALL"
            prim = prim or "ALL"
    elif intent == "GET_RULE":
        rule = first("RULE") or next(
            (
                n.upper() if n != "AI_usage" else "AI_USAGE"
                for n, rx in RULE_CUES
                if re.search(rx, t)
            ),
            None,
        )
        frame.predicate, a["rule"] = "GET_RULE", rule or "ALL"
        prim = rule or prim or "ALL"
    elif intent == "GET_INSTRUCTOR":
        frame.predicate, a["course"] = "GET_INSTRUCTOR", "CO3085"
        prim = "CO3085"
    elif intent == "CONTEXT_DEPENDENT":
        m = PRONOUN_RE.search(t)
        frame.predicate, a["expr"] = (
            "NEEDS_CONTEXT",
            '"%s"' % (m.group().strip() if m else "?"),
        )
        prim = "-"
    else:
        frame.predicate, a["reason"] = "UNKNOWN", "OUT_OF_SCOPE"
        prim = "OUT_OF_SCOPE"
    frame.primary = prim or "-"
    return frame
