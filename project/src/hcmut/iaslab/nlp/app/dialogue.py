"""Phan III: Xu ly ngu canh hoi thoai cuc bo (Local Discourse Context & Reference).

Ap dung ly thuyet Chuong 7 (Logical Form -> Contextual Interpretation) va
Chuong 12 (Discourse Context and Reference: History List, Centering,
Definite Descriptions, Ellipsis).

Pipeline moi luot hoi thoai:
  User Turn + DiscourseState
    -> Tree/Lexical Reference Detection
    -> History List + Centering + Type Compatibility Resolution
    -> Query Rewriting (+ Ellipsis Expansion)
    -> Part I Parser (CourseParser) -> Semantic Frame -> Query -> KB Answer
    -> Update DiscourseState (entities tu cau hoi U_n va cau tra loi A_n)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from src.grammar import CourseParser, load_cfg

from .nlu import (
    Entity,
    EntityLinker,
    Frame,
    build_frame,
    collect_tree_features,
    norm,
)
from .part2 import PROJECT_ROOT, CourseAssistant, default_dirs
from .qa import NEED_CONTEXT, NOT_FOUND, Query, Result, build_query

TRACKED_ENTITY_TYPES = (
    "PART",
    "CHAPTER",
    "WEEK",
    "LO",
    "ASSIGN",
    "EXAM",
    "RULE",
    "TOPIC",
    "RESOURCE_TOPIC",
    "COURSE",
)

GENERIC_ENTITY_TYPES = {"COURSE"}

ROMAN_BY_NUM = {1: "I", 2: "II", 3: "III", 4: "IV"}
NUM_BY_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}

# Tuong thich loai thuc the cho tung dang bieu thuc tham chieu (Chapter 12.2, 12.4).
DEFINITE_TYPE_MAP: list[tuple[re.Pattern[str], tuple[str, ...], bool]] = [
    (
        re.compile(r"\b(?:chương|chapter)\s+(?:này|đó|ấy)\b", re.I),
        ("CHAPTER",),
        True,
    ),
    (
        re.compile(r"\b(?:tuần|week)\s+(?:này|đó|ấy)\b", re.I),
        ("WEEK",),
        True,
    ),
    (
        re.compile(r"\b(?:phần|mục)\s+(?:này|đó|ấy)\b", re.I),
        ("PART", "CHAPTER", "TOPIC", "LO", "WEEK"),
        True,
    ),
    (
        re.compile(r"\b(?:nội dung|chủ đề|vấn đề|kiến thức)\s+(?:này|đó|ấy)\b", re.I),
        ("LO", "TOPIC", "CHAPTER", "WEEK", "PART"),
        True,
    ),
    (
        re.compile(r"\b(?:bài tập|btl)\s+(?:này|đó|ấy)\b", re.I),
        ("PART", "ASSIGN"),
        True,
    ),
    (
        re.compile(r"\b(?:môn|môn học|học phần)\s+(?:này|đó|ấy)\b", re.I),
        ("COURSE",),
        True,
    ),
    (
        re.compile(r"\b(?:kỳ thi|bài thi)\s+(?:này|đó|ấy)\b", re.I),
        ("EXAM",),
        True,
    ),
    (
        re.compile(r"\b(?:quy định)\s+(?:này|đó|ấy)\b", re.I),
        ("RULE",),
        True,
    ),
]

ORDINAL_REF_RE = re.compile(
    r"\b(chương|chapter|tuần|week|phần|part)\s+(sau|tiếp theo|kế tiếp|trước)\b",
    re.I,
)
ELLIPSIS_RE = re.compile(r"^\s*(?:thế\s+|vậy\s+)?còn\s+(.+?)\s*\??\s*$")
PRONOUN_N_RE = re.compile(r"\bnó\b", re.I)


@dataclass
class DiscourseEntity:
    type: str
    value: str
    label: str
    turn_id: int
    source: str  # "user_primary" | "user_secondary" | "answer"
    salience: int


@dataclass
class Resolution:
    expression: str
    ref_kind: str
    target: DiscourseEntity | None
    resolved_value: str
    surface: str
    source_note: str


@dataclass
class DialogueTurnResult:
    turn_index: int
    user_input: str
    context_snapshot: str
    context_used: str
    resolutions: list[Resolution]
    resolved_input: str
    parse_status: str
    frame: Frame
    query: Query
    result: Result


def entity_to_surface(etype: str, value: str) -> str:
    """Chuyen ma thuc the ve cum tu tieng Viet hop le trong grammar/NLU."""
    if etype == "CHAPTER" and value.startswith("CH"):
        return f"Chương {int(value[2:])}"
    if etype == "WEEK" and value.startswith("WEEK_"):
        return f"Tuần {int(value.split('_')[1])}"
    if etype == "LO" and value.startswith("LO"):
        return f"L.O.{value[2:]}"
    if etype == "PART":
        if value.startswith("PART_"):
            return f"Phần {value.split('_', 1)[1]} của bài tập lớn"
        if value.startswith("COURSE_PART_"):
            return f"Phần {value.rsplit('_', 1)[1]}"
    if etype == "ASSIGN":
        return "bài tập lớn"
    if etype == "EXAM":
        return value.lower()
    if etype == "COURSE":
        return "môn NLP"
    if etype == "RULE":
        return "LLM/RAG" if value == "LLM_RAG" else "công cụ AI"
    return value.replace("_", " ").lower()


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    """Thay the cum tham chieu giu hoa chu dau cau neu o vi tri dau."""
    rep = replacement
    if start == 0 and rep:
        rep = rep[0].upper() + rep[1:]
    elif start > 0 and rep.startswith(("Chương ", "Tuần ", "Phần ")):
        # Giu viet hoa chuan tieng Viet cho "Chương 3", "Tuần 5", "Phần I".
        pass
    return text[:start] + rep + text[end:]


def _tree_has_pronoun_or_dem(tree_tuple) -> tuple[bool, list[str]]:
    """Doc cay cu phap Phan I de kiem tra dai tu hoac cum Danh tu + Chi dinh tu."""
    if not tree_tuple:
        return False, []
    _, by_label = collect_tree_features(tree_tuple)
    has_pronoun = "nó" in by_label.get("Pronoun", [])
    dem_phrases: list[str] = []

    def walk(node):
        if not isinstance(node, tuple):
            return
        label, kids = node
        if label == "Nominal":
            sub_labels, sub_by = collect_tree_features(node)
            if "Demonstrative" in sub_labels:
                leaves = _collect_leaves(node)
                if leaves:
                    dem_phrases.append(" ".join(leaves))
        for k in kids:
            walk(k)

    walk(tree_tuple)
    return has_pronoun, dem_phrases


def _collect_leaves(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    out: list[str] = []
    for k in node[1]:
        out.extend(_collect_leaves(k))
    return out


class DiscourseState:
    """Trang thai hoi thoai cuc bo: History List + Center (Cb) + luot gan nhat."""

    def __init__(self) -> None:
        self.history: list[DiscourseEntity] = []
        self.center: DiscourseEntity | None = None
        self.last_user_text: str = ""
        self.last_frame: Frame | None = None
        self.last_answer: str = ""
        self.turn_count: int = 0

    def reset(self) -> None:
        self.history.clear()
        self.center = None
        self.last_user_text = ""
        self.last_frame = None
        self.last_answer = ""
        self.turn_count = 0

    def snapshot(self) -> str:
        if not self.history:
            return "(rỗng - chưa có ngữ cảnh trước đó)"
        center_str = (
            f"{self.center.value}({self.center.type}, T{self.center.turn_id})"
            if self.center
            else "-"
        )
        hist_str = ", ".join(
            f"{e.value}({e.type}, T{e.turn_id}, {e.source})"
            for e in self.history[:6]
        )
        return f"center={center_str} | history=[{hist_str}]"

    def _upsert(self, entity: DiscourseEntity) -> None:
        for i, old in enumerate(self.history):
            if old.type == entity.type and old.value == entity.value:
                # Chi nang cap neu luot moi co salience cao hon hoac turn moi hon.
                self.history.pop(i)
                break
        self.history.insert(0, entity)

    def find_candidate(
        self,
        allowed_types: tuple[str, ...],
        exclude_values: set[str] | None = None,
        prefer_type_tier: bool = False,
    ) -> DiscourseEntity | None:
        """Chon thuc the phu hop nhat theo History List, Centering va rang buoc kieu."""
        excluded = exclude_values or set()
        candidates = [
            e
            for e in self.history
            if e.type in allowed_types and e.value not in excluded
        ]
        if not candidates:
            return None

        type_rank = {t: idx for idx, t in enumerate(allowed_types)}
        if prefer_type_tier:
            candidates.sort(
                key=lambda e: (
                    type_rank[e.type],
                    -e.turn_id,
                    0 if (self.center and e.value == self.center.value) else 1,
                    -e.salience,
                )
            )
            return candidates[0]

        # Voi dai tu "no": uu tien luot gan nhat -> tam diem Center -> salience -> loai dac thu.
        candidates.sort(
            key=lambda e: (
                -e.turn_id,
                0 if (self.center and e.value == self.center.value) else 1,
                -e.salience,
                type_rank[e.type],
            )
        )
        return candidates[0]


class DialogueManager:
    """Quan ly hoi thoai nhieu luot ket hop Phan I Parser + Phan II Classical QA."""

    def __init__(
        self,
        kb_dir: str | Path | None = None,
        grammar_path: str | Path | None = None,
    ) -> None:
        resolved_kb = kb_dir or (PROJECT_ROOT / "data")
        resolved_grammar = Path(grammar_path) if grammar_path else (
            PROJECT_ROOT / "src" / "grammar" / "grammar.txt"
        )
        self.assistant = CourseAssistant(resolved_kb)
        self.kb = self.assistant.kb
        self.linker: EntityLinker = self.assistant.linker
        self.parser = (
            CourseParser(load_cfg(resolved_grammar))
            if resolved_grammar.is_file()
            else None
        )
        self.state = DiscourseState()

    def reset(self) -> None:
        self.state.reset()

    def _parse_tree_tuple(self, sentence: str):
        if not self.parser:
            return None, "MISSING", None
        tree = self.parser.parse(sentence)
        if tree is None:
            return None, "REJECTED", None
        from .nlu import parse_bracket

        bracket = tree.pformat(margin=1_000_000)
        return parse_bracket(bracket), "OK", bracket

    @staticmethod
    def _shift_ordinal(entity: DiscourseEntity, delta: int) -> tuple[str, str]:
        """Dich chi so chuong/tuan/phan (+1 cho 'sau/tiep theo', -1 cho 'truoc')."""
        if entity.type == "CHAPTER" and entity.value.startswith("CH"):
            num = int(entity.value[2:]) + delta
            val = f"CH{num:02d}"
            return val, f"Chương {num}"
        if entity.type == "WEEK" and entity.value.startswith("WEEK_"):
            num = int(entity.value.split("_")[1]) + delta
            val = f"WEEK_{num:02d}"
            return val, f"Tuần {num}"
        if entity.type == "PART":
            prefix, roman = entity.value.rsplit("_", 1)
            num = NUM_BY_ROMAN.get(roman, 0) + delta
            new_roman = ROMAN_BY_NUM.get(num, str(num))
            val = f"{prefix}_{new_roman}"
            return val, entity_to_surface("PART", val)
        return entity.value, entity.label

    def _extract_turn_entities(
        self, sentence: str, frame: Frame, result: Result, turn_id: int
    ) -> list[DiscourseEntity]:
        """Trich cac thuc the tu cau hoi U_n (user) va cau tra loi A_n (answer)."""
        collected: list[DiscourseEntity] = []
        seen: set[tuple[str, str]] = set()

        def add_ent(etype: str, val: str, source: str, salience: int) -> None:
            if etype not in TRACKED_ENTITY_TYPES:
                return
            if not val or val in ("ALL", "OUT_OF_SCOPE", "-"):
                return
            key = (etype, val)
            if key in seen:
                return
            seen.add(key)
            collected.append(
                DiscourseEntity(
                    type=etype,
                    value=val,
                    label=entity_to_surface(etype, val),
                    turn_id=turn_id,
                    source=source,
                    salience=salience,
                )
            )

        # 1. Thuc the trong Frame cau hoi nguoi dung (Uu tien primary > secondary).
        if frame.primary and frame.primary not in ("-", "ALL", "OUT_OF_SCOPE"):
            for e in frame.entities:
                if e.value == frame.primary:
                    add_ent(e.type, e.value, "user_primary", 4)
            if "part" in frame.args and frame.args["part"] == frame.primary:
                add_ent("PART", frame.args["part"], "user_primary", 4)

        for k, v in frame.args.items():
            if k == "part" and isinstance(v, str):
                add_ent("PART", v, "user_primary" if v == frame.primary else "user_secondary", 3)
            elif k == "chapter" and isinstance(v, str):
                add_ent("CHAPTER", v, "user_primary" if v == frame.primary else "user_secondary", 3)
            elif k == "week" and isinstance(v, str):
                add_ent("WEEK", v, "user_primary" if v == frame.primary else "user_secondary", 3)
            elif k == "lo" and isinstance(v, str):
                add_ent("LO", v, "user_primary" if v == frame.primary else "user_secondary", 3)
            elif k == "assignment" and isinstance(v, str):
                add_ent("ASSIGN", v, "user_secondary", 2)

        for e in frame.entities:
            sal = 1 if e.type in GENERIC_ENTITY_TYPES else 2
            add_ent(e.type, e.value, "user_secondary", sal)

        # 2. Thuc the moi sinh ra tu cau tra loi cua he thong (vd: hoi Tuan 3 -> tra loi Chuong 3).
        if result.found and result.answer:
            for e in self.linker.extract(result.answer):
                sal = 0 if e.type in GENERIC_ENTITY_TYPES else 1
                add_ent(e.type, e.value, "answer", sal)
        return collected

    def _update_state(
        self, sentence: str, frame: Frame, result: Result
    ) -> list[DiscourseEntity]:
        self.state.turn_count += 1
        turn_id = self.state.turn_count
        ents = self._extract_turn_entities(sentence, frame, result, turn_id)
        # Day entity theo thu tu: answer truoc, user sau -> user nam o dau history list.
        for e in sorted(ents, key=lambda x: x.salience):
            self.state._upsert(e)

        # Cap nhat Center (Cb): uu tien thuc the dac thu co salience cao nhat o luot hien tai.
        specific = [e for e in ents if e.type not in GENERIC_ENTITY_TYPES]
        if specific:
            specific.sort(
                key=lambda e: (
                    -e.salience,
                    0
                    if (self.state.center and e.value == self.state.center.value)
                    else 1,
                )
            )
            self.state.center = specific[0]
        elif ents and self.state.center is None:
            self.state.center = ents[0]

        self.state.last_user_text = sentence
        if frame.intent not in ("UNKNOWN", "CONTEXT_DEPENDENT"):
            self.state.last_frame = frame
        self.state.last_answer = result.answer
        return ents

    def _expand_ellipsis(
        self, target_entity_type: str, target_surface: str
    ) -> str | None:
        """Khoi phuc cau hoi day du tu khung ngu nghia luot truoc cho cau tinh luoc 'Con ...?'."""
        last = self.state.last_frame
        if not last:
            return None
        p = last.predicate
        if p == "CHECK_CHAPTER_COVERS" and target_entity_type == "CHAPTER":
            topic = last.args.get("topic", "")
            topic_surf = entity_to_surface("TOPIC", topic)
            return f"{target_surface} có {topic_surf} không?"
        if p == "GET_CHAPTER_CONTENT" and target_entity_type == "CHAPTER":
            return f"{target_surface} nói về gì?"
        if p == "GET_CHAPTER_WEEKS" and target_entity_type == "CHAPTER":
            return f"{target_surface} học vào tuần nào?"
        if p == "GET_WEEK_CONTENT" and target_entity_type == "WEEK":
            return f"{target_surface} học gì?"
        if p == "GET_EXAM_DURATION" and target_entity_type == "EXAM":
            return f"{target_surface} bao nhiêu phút?"
        if p == "GET_EXAM_FORMAT" and target_entity_type == "EXAM":
            return f"{target_surface} thi như thế nào?"
        if p in ("GET_LO", "GET_LO_CHAPTER") and target_entity_type == "LO":
            suffix = "nằm ở chương nào?" if p == "GET_LO_CHAPTER" else "là gì?"
            return f"{target_surface} {suffix}"
        if p == "GET_ASSIGNMENT_PART" and target_entity_type == "PART":
            return f"{target_surface} cần xuất file gì?"
        if p == "CHECK_ASSIGNMENT_PART" and target_entity_type == "PART":
            return f"{target_surface} là gì?"
        if target_entity_type == "CHAPTER":
            return f"{target_surface} nói về gì?"
        if target_entity_type == "WEEK":
            return f"{target_surface} học gì?"
        return None

    def _resolve_inline_references(
        self, sentence: str, tree_tuple
    ) -> tuple[str, list[Resolution], bool]:
        """Giai tham chieu dai tu ('no') va mo ta xac dinh ('chuong nay', 'chuong sau').

        Tra ve: (cau sau giai tham chieu, danh sach Resolution, co bieu thuc tham chieu that bai khong).
        """
        t_norm = norm(sentence)
        current_ents = self.linker.extract(sentence)
        current_vals = {e.value for e in current_ents}
        has_tree_pronoun, tree_dems = _tree_has_pronoun_or_dem(tree_tuple)
        resolutions: list[Resolution] = []
        working = unicodedata.normalize("NFC", sentence)
        unresolved_ref = False

        # 1. Tham chieu thu tu tuong doi: "chuong sau", "tuan truoc", ...
        while True:
            m = ORDINAL_REF_RE.search(working)
            if not m:
                break
            noun_kw, rel_kw = m.group(1).lower(), m.group(2).lower()
            delta = -1 if rel_kw == "trước" else 1
            base_type = (
                "CHAPTER"
                if noun_kw in ("chương", "chapter")
                else ("WEEK" if noun_kw in ("tuần", "week") else "PART")
            )
            cand = self.state.find_candidate((base_type,), prefer_type_tier=True)
            expr = working[m.start() : m.end()]
            if not cand:
                resolutions.append(
                    Resolution(
                        expression=expr,
                        ref_kind="ordinal_reference",
                        target=None,
                        resolved_value="-",
                        surface="",
                        source_note="khong co thuc the goc trong lich su",
                    )
                )
                unresolved_ref = True
                break
            new_val, new_surf = self._shift_ordinal(cand, delta)
            resolutions.append(
                Resolution(
                    expression=expr,
                    ref_kind="ordinal_reference",
                    target=cand,
                    resolved_value=new_val,
                    surface=new_surf,
                    source_note=f"history({cand.type}={cand.value}, T{cand.turn_id}) -> {delta:+d}",
                )
            )
            working = _replace_span(working, m.start(), m.end(), new_surf)

        # 2. Mo ta xac dinh: "chuong nay", "phan nay", "noi dung nay", ...
        for rx, allowed_types, tier_mode in DEFINITE_TYPE_MAP:
            while True:
                m = rx.search(working)
                if not m:
                    break
                expr = working[m.start() : m.end()]
                cand = self.state.find_candidate(
                    allowed_types,
                    exclude_values=current_vals,
                    prefer_type_tier=tier_mode,
                )
                detect_via = (
                    "syntax_tree+rules"
                    if any(expr.lower() == d for d in tree_dems)
                    else "rules"
                )
                if not cand:
                    resolutions.append(
                        Resolution(
                            expression=expr,
                            ref_kind="definite_description",
                            target=None,
                            resolved_value="-",
                            surface="",
                            source_note=f"{detect_via}: khong tim thay trong history",
                        )
                    )
                    unresolved_ref = True
                    break
                surf = cand.label
                resolutions.append(
                    Resolution(
                        expression=expr,
                        ref_kind="definite_description",
                        target=cand,
                        resolved_value=cand.value,
                        surface=surf,
                        source_note=(
                            f"{detect_via}; history:{cand.source}(T{cand.turn_id})"
                        ),
                    )
                )
                working = _replace_span(working, m.start(), m.end(), surf)

        # 3. Dai tu "no"
        while True:
            m = PRONOUN_N_RE.search(working)
            if not m:
                break
            expr = working[m.start() : m.end()]
            # Selectional preference don gian cho dai tu "no".
            if re.search(r"\b(?:file|output|xuất|nộp|deadline)\b", t_norm):
                allowed = ("PART", "ASSIGN", "CHAPTER", "WEEK")
            elif re.search(r"\b(?:liên quan|nói về|học về|hỏi đáp|parsing)\b", t_norm):
                allowed = ("CHAPTER", "PART", "LO", "TOPIC", "WEEK", "ASSIGN", "COURSE")
            else:
                allowed = (
                    "CHAPTER",
                    "PART",
                    "LO",
                    "WEEK",
                    "ASSIGN",
                    "TOPIC",
                    "EXAM",
                    "RULE",
                    "COURSE",
                )
            cand = self.state.find_candidate(
                allowed, exclude_values=current_vals, prefer_type_tier=False
            )
            detect_via = "syntax_tree+rules" if has_tree_pronoun else "rules"
            if not cand:
                resolutions.append(
                    Resolution(
                        expression=expr,
                        ref_kind="anaphora",
                        target=None,
                        resolved_value="-",
                        surface="",
                        source_note=f"{detect_via}: khong tim thay trong history",
                    )
                )
                unresolved_ref = True
                break
            surf = cand.label
            is_center = self.state.center and cand.value == self.state.center.value
            resolutions.append(
                Resolution(
                    expression=expr,
                    ref_kind="anaphora",
                    target=cand,
                    resolved_value=cand.value,
                    surface=surf,
                    source_note=(
                        f"{detect_via}; {'center' if is_center else 'history'}:{cand.source}(T{cand.turn_id})"
                    ),
                )
            )
            working = _replace_span(working, m.start(), m.end(), surf)

        return working, resolutions, unresolved_ref

    def ask(self, sentence: str) -> DialogueTurnResult:
        """Xu ly mot luot hoi thoai co su dung va cap nhat ngu canh cuc bo."""
        snapshot = self.state.snapshot()
        raw_tree, raw_status, _ = self._parse_tree_tuple(sentence)
        resolutions: list[Resolution] = []
        resolved_text = sentence
        unresolved_flag = False
        ellipsis_fallback_chapter: str | None = None

        # Kiem tra tinh luoc dang "Con <X>?" truoc, sau do giai tham chieu ben trong.
        ell_match = ELLIPSIS_RE.match(norm(sentence))
        if ell_match:
            fragment = sentence.strip()
            # Giai tham chieu tren phan doan sau "Con ..." (vd: "chuong sau" -> "Chuong 12").
            inner = ell_match.group(1)
            resolved_inner, inner_res, inner_fail = self._resolve_inline_references(
                inner, raw_tree
            )
            resolutions.extend(inner_res)
            if inner_fail:
                unresolved_flag = True
            else:
                inner_ents = self.linker.extract(resolved_inner)
                # Uu tien thuc the chinh trong phan tinh luoc.
                new_ent = next(
                    (e for e in inner_ents if e.type not in GENERIC_ENTITY_TYPES),
                    inner_ents[0] if inner_ents else None,
                )
                if new_ent is not None and self.state.last_frame is not None:
                    expanded = self._expand_ellipsis(
                        new_ent.type, entity_to_surface(new_ent.type, new_ent.value)
                    )
                    if expanded:
                        if (
                            self.state.last_frame.predicate == "CHECK_CHAPTER_COVERS"
                            and new_ent.type == "CHAPTER"
                        ):
                            ellipsis_fallback_chapter = new_ent.value
                        resolutions.append(
                            Resolution(
                                expression=fragment,
                                ref_kind="ellipsis",
                                target=self.state.center,
                                resolved_value=new_ent.value,
                                surface=expanded,
                                source_note=(
                                    f"ellipsis tu luot T{self.state.turn_count} ({self.state.last_frame.predicate})"
                                ),
                            )
                        )
                        resolved_text = expanded
                    else:
                        unresolved_flag = True
                else:
                    unresolved_flag = True
        else:
            resolved_text, inline_res, inline_fail = self._resolve_inline_references(
                sentence, raw_tree
            )
            resolutions.extend(inline_res)
            unresolved_flag = inline_fail

        # Neu tham chieu that bai -> tra ve thong bao thieu ngu canh, khong doan mo.
        if unresolved_flag:
            frame = build_frame(
                sentence, self.linker, (raw_status, raw_tree)
            )
            query = build_query(frame)
            result = Result(NEED_CONTEXT, [], found=False)
            turn_res = DialogueTurnResult(
                turn_index=self.state.turn_count + 1,
                user_input=sentence,
                context_snapshot=snapshot,
                context_used="Khong du ngu canh phu hop trong lich su hoi thoai",
                resolutions=resolutions,
                resolved_input=sentence,
                parse_status=raw_status,
                frame=frame,
                query=query,
                result=result,
            )
            self._update_state(sentence, frame, result)
            return turn_res

        # Phan tich lai cau da giai tham chieu bang Parser Phan I -> Semantic -> Query -> KB.
        res_tree, res_status, res_bracket = self._parse_tree_tuple(resolved_text)
        frame, query, result = self.assistant.ask(resolved_text, res_bracket)

        # Truong hop "Con chuong sau?" sau CHECK_CHAPTER_COVERS (Dialog 04):
        # Neu chuong moi khong de cap chu de cu, bo sung tom tat chuong moi de cau tra loi tu nhien.
        if (
            ellipsis_fallback_chapter
            and frame.predicate == "CHECK_CHAPTER_COVERS"
            and result.found
            and not result.answer.startswith("Có.")
            and ellipsis_fallback_chapter in self.kb.chapters
        ):
            ch_frame = build_frame(
                f"{entity_to_surface('CHAPTER', ellipsis_fallback_chapter)} nói về gì?",
                self.linker,
                (res_status, res_tree),
            )
            ch_query = build_query(ch_frame)
            ch_res = self.assistant.engine.execute(ch_frame, ch_query)
            if ch_res.found:
                result = Result(
                    answer=f"{result.answer} Nội dung chính của chương này: {ch_res.answer}",
                    sources=list(dict.fromkeys(result.sources + ch_res.sources)),
                    found=True,
                )

        if resolutions:
            used_items = [
                f"{r.expression} -> {r.resolved_value} ({r.source_note})"
                for r in resolutions
            ]
            context_used = "; ".join(used_items)
        else:
            context_used = "Không sử dụng ngữ cảnh (câu hỏi độc lập)"

        turn_res = DialogueTurnResult(
            turn_index=self.state.turn_count + 1,
            user_input=sentence,
            context_snapshot=snapshot,
            context_used=context_used,
            resolutions=resolutions,
            resolved_input=resolved_text,
            parse_status=res_status,
            frame=frame,
            query=query,
            result=result,
        )
        self._update_state(resolved_text, frame, result)
        return turn_res

    def run_dialogue(self, utterances: list[str]) -> list[DialogueTurnResult]:
        """Chay mot doan hoi thoai hoan chinh va reset trang thai truoc khi bat dau."""
        self.reset()
        return [self.ask(u) for u in utterances]


def load_dialogues(path: str | Path) -> list[dict]:
    """Doc cac hoi thoai mau tu data/dialogues.txt (chi lay loi thoai U_k lam input)."""
    raw = Path(path).read_text(encoding="utf-8")
    dialogues: list[dict] = []
    for block in re.split(r"(?m)^(?=DIALOGUE \d+)", raw):
        m = re.match(r"^(DIALOGUE \d+)", block.strip())
        if not m:
            continue
        did = m.group(1)
        utterances: list[str] = []
        expected_refs: list[str] = []
        for line in block.splitlines():
            um = re.match(r"^\s*U\d+:\s*(.+?)\s*$", line)
            if um:
                utterances.append(um.group(1))
            em = re.match(r"^\s*EXPECTED_REFERENCE:\s*(.+?)\s*$", line)
            if em:
                expected_refs.append(em.group(1))
        if utterances:
            dialogues.append(
                {
                    "id": did,
                    "utterances": utterances,
                    "expected_references": expected_refs,
                }
            )
    return dialogues


def format_dialogue_report(
    dialogue_id: str,
    turns: list[DialogueTurnResult],
    expected_refs: list[str] | None = None,
) -> str:
    """Dinh dang ket qua hoi thoai du 4 muc theo yeu cau Muc 5 cua de bai."""
    lines = ["=" * 72, dialogue_id, "=" * 72]
    if expected_refs:
        lines.append("Tham chiếu kỳ vọng (reference): " + "; ".join(expected_refs))
    for t in turns:
        lines.append(f"\n[Lượt {t.turn_index}]")
        lines.append(f"  Hội thoại đầu vào : U{t.turn_index}: {t.user_input}")
        lines.append(f"  Ngữ cảnh sử dụng  : {t.context_snapshot}")
        if t.resolutions:
            interp_parts = [
                f'"{r.expression}" -> {r.resolved_value} ({r.ref_kind}, {r.source_note})'
                for r in t.resolutions
            ]
            lines.append("  Kết quả diễn giải : " + "; ".join(interp_parts))
        else:
            lines.append("  Kết quả diễn giải : Không có tham chiếu ngữ cảnh")
        lines.append(f"    Câu sau diễn giải : {t.resolved_input}")
        lines.append(
            f"    Cú pháp & Ngữ nghĩa: parse={t.parse_status} | {t.frame.semantic()}"
        )
        lines.append(f"    Truy vấn KB       : {t.query.text}")
        lines.append(f"  Câu trả lời       : A{t.turn_index}: {t.result.answer}")
        lines.append(
            f"    Nguồn KB          : {', '.join(t.result.sources) or '-'}"
        )
    lines.append("")
    return "\n".join(lines)


def run(**kwargs) -> None:
    """Chay Phan III tren data/dialogues.txt (+ tinh huong mo rong) -> output/dialogue.txt."""
    _, out_dir = default_dirs()
    out_dir = Path(kwargs.get("output_dir", out_dir))
    kb_dir = kwargs.get("kb_dir", str(PROJECT_ROOT / "data"))
    dialogues_path = Path(
        kwargs.get("dialogues_path", PROJECT_ROOT / "data" / "dialogues.txt")
    )

    dm = DialogueManager(kb_dir=kb_dir, grammar_path=kwargs.get("grammar_path"))
    sections: list[str] = []

    sample_dialogues = load_dialogues(dialogues_path) if dialogues_path.is_file() else []
    for d in sample_dialogues:
        turns = dm.run_dialogue(d["utterances"])
        sections.append(
            format_dialogue_report(d["id"], turns, d.get("expected_references"))
        )

    # Bo sung kich ban vi du trong de bai (Trang 2 & 7) + tinh huong 3 luot + guardrails.
    extra_cases = [
        (
            "EXTRA 01 - Ví dụ đề bài (Tuần 5 -> phần này)",
            ["Tuần 5 học những gì?", "Tài liệu của phần này ở đâu?"],
        ),
        (
            "EXTRA 02 - Hội thoại 3 lượt với Centering (Cb duy trì qua nhiều lượt)",
            [
                "Tuần 3 học gì?",
                "Tài liệu của chương này là gì?",
                "Nó có liên quan đến PCFG không?",
            ],
        ),
        (
            "EXTRA 03 - Kiểm tra an toàn khi thiếu ngữ cảnh / ngoài phạm vi KB",
            [
                "Chương 12 nói về gì?",
                "Còn chương sau?",
            ],
        ),
    ]
    for eid, utts in extra_cases:
        turns = dm.run_dialogue(utts)
        sections.append(format_dialogue_report(eid, turns))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dialogue.txt"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[III] {len(sample_dialogues) + len(extra_cases)} hoi thoai -> {out_path}")
