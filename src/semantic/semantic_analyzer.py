
"""Rule-based semantic interpretation for the Course Assistant domain.

This module complements (and does not replace) the CFG parser from Part I.
It identifies intents, extracts entities, and prepares retrieval queries.
"""

from dataclasses import dataclass, asdict
import re
import unicodedata


INTENT_KEYWORDS = {
    "GET_DEFINITION": [
        "là gì", "nghĩa là gì", "được hiểu là gì",
        "định nghĩa", "khái niệm",
    ],
    "GET_LO": [
        "l.o.", "lo1", "lo2", "lo3", "lo4",
        "learning outcome", "chuẩn đầu ra",
    ],
    "GET_DEADLINE": [
        "deadline", "hạn nộp", "thời hạn",
        "khi nào nộp", "ngày nộp",
    ],
    "GET_RESOURCE": [
        "tài liệu", "giáo trình", "tham khảo",
        "statistical nlp", "python nlp", "sách",
    ],
    "GET_ASSIGNMENT": [
        "bài tập lớn", "btl", "phần i", "phần ii",
        "yêu cầu xây dựng", "parser",
    ],
    "GET_RULE": [
        "quy định", "được phép", "sử dụng ai",
        "llm", "rag", "liêm chính", "điểm danh",
        "code of ethics", "chia sẻ bài",
    ],
    "GET_SCHEDULE": [
        "tuần", "lịch học", "kế hoạch học", "học vào tuần",
    ],
    "GET_TOPIC": [
        "chương", "cfg", "pcfg", "semantic grammar",
        "reference", "parsing", "question answering",
        "selectional", "features", "knowledge representation",
        "anaphora", "centering",
    ],
    "GET_COURSE_INFO": [
        "tín chỉ", "ects", "học phần", "môn học",
        "midterm", "final", "bao nhiêu phút",
        "mấy phút", "phần chính", "môn nlp", "co3085",
        "học kỳ", "semester", "thuộc khoa nào",
        "bao nhiêu giờ",
    ],
}


@dataclass
class SemanticResult:
    intent: str
    entities: list[str]
    search_query: str
    parser_status: str
    parse_tree: str

    def to_dict(self):
        return asdict(self)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_intent(question: str) -> str:
    q = _normalise(question)

    # Definition questions must be checked before broad topic categories.
    if re.search(r"\b(nlp|cfg|pcfg|llm|rag)\b.*\b(là gì|nghĩa là gì)\b", q):
        return "GET_DEFINITION"

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            return intent

    return "UNKNOWN"


def _extract_entities(question: str) -> list[str]:
    q = _normalise(question)
    entities = []

    if re.search(r"\bco\s*3085\b", q) or "nlp" in q or "môn học" in q:
        entities.append("CO3085")

    chapter = re.search(r"chương\s*(\d+)", q)
    if chapter:
        entities.append(f"CH{int(chapter.group(1)):02d}")

    week = re.search(r"tuần\s*(\d+)", q)
    if week:
        entities.append(f"WEEK_{int(week.group(1)):02d}")

    lo = re.search(r"\blo\.?\s*(\d+\.\d+|\d+)\b", q)
    if lo:
        entities.append("LO" + lo.group(1))

    aliases = {
        "cfg": "CFG",
        "pcfg": "PCFG",
        "llm": "LLM",
        "rag": "RAG",
        "nlp": "NLP",
        "semantic grammar": "SEMANTIC_GRAMMAR",
        "question answering": "QUESTION_ANSWERING",
        "selectional restriction": "SELECTIONAL_RESTRICTION",
        "reference": "REFERENCE",
        "python nlp": "PYTHON_NLP",
        "statistical nlp": "STATISTICAL_NLP",
        "bài tập lớn": "BTL01",
    }

    for phrase, entity in aliases.items():
        if phrase in q and entity not in entities:
            entities.append(entity)

    return entities


def analyze_semantics(question: str, tree=None) -> SemanticResult:
    """Build a semantic representation from a question and parse tree."""

    intent = _detect_intent(question)
    entities = _extract_entities(question)
    parse_text = str(tree) if tree is not None else "()"

    return SemanticResult(
        intent=intent,
        entities=entities,
        search_query=question.strip(),
        parser_status="PARSED" if tree is not None else "NOT_PARSED",
        parse_tree=parse_text,
    )
