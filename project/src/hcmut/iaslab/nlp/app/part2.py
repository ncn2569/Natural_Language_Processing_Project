"""Phan II (Lua chon 1: Classical NLP) - pipeline:
Input -> Parsing (Phan I) -> Semantic -> Intent/Entity -> Query -> KB -> Answer

Ghi: semantic.txt, intent-entity.txt, query.txt, answer.txt
"""

import os
from pathlib import Path

# from .generate import PROJECT_ROOT, default_dirs
from .kb import KnowledgeBase
from .nlu import EntityLinker, build_frame, parse_bracket
from .qa import Engine, build_query

PROJECT_ROOT = Path(__file__).parents[5]


def default_dirs():
    """Docker cham bai mount /nlp/input va /nlp/output; chay local thi dung thu muc trong project."""
    in_dir = os.environ.get("NLP_INPUT") or (
        "/nlp/input"
        if os.path.isdir("/nlp/input")
        else str(PROJECT_ROOT / "input")
    )
    out_dir = os.environ.get("NLP_OUTPUT") or (
        "/nlp/output"
        if os.path.isdir("/nlp/output")
        else str(PROJECT_ROOT / "output")
    )
    return in_dir, out_dir


class CourseAssistant:
    def __init__(self, kb_dir=None):
        # self.kb = KnowledgeBase(kb_dir or PROJECT_ROOT / "data" / "course")
        self.kb = KnowledgeBase(kb_dir or PROJECT_ROOT / "data")

        self.linker = EntityLinker(self.kb)
        self.engine = Engine(self.kb, self.linker)

    def ask(self, sentence, parse_line=None):
        """parse_line: dong tuong ung trong parse-results.txt (None neu chua co Phan I)."""
        if parse_line is None:
            state = ("MISSING", None)
        else:
            tree = parse_bracket(parse_line)
            state = ("OK", tree) if tree else ("REJECTED", None)
        frame = build_frame(sentence, self.linker, state)
        query = build_query(frame)
        result = self.engine.execute(frame, query)
        return frame, query, result


def _read_lines(path):
    p = Path(path)
    if not p.exists():
        return []
    return [l.rstrip("\n") for l in p.read_text(encoding="utf-8").splitlines()]


def run(**kwargs):
    in_dir, out_dir = default_dirs()
    in_dir = kwargs.get("input_dir", in_dir)
    out_dir = kwargs.get("output_dir", out_dir)
    sentences = [s for s in _read_lines(Path(in_dir) / "sentences.txt") if s.strip()]
    parse_lines = _read_lines(Path(out_dir) / "parse-results.txt")
    use_trees = len(parse_lines) == len(sentences) and len(sentences) > 0
    if not use_trees:
        print(
            "[II] Khong co parse-results.txt khop input -> chay o che do luat tu vung (parse=MISSING)."
        )

    ca = CourseAssistant(kwargs.get("kb_dir"))
    os.makedirs(out_dir, exist_ok=True)
    sem, ie, qq, aa = [], [], [], []
    for i, s in enumerate(sentences):
        frame, query, res = ca.ask(s, parse_lines[i] if use_trees else None)
        tag = f"[{i + 1:03d}] {s}"
        sem.append(f"{tag}\n  {frame.semantic()}\n")
        ents = ", ".join(f"{e.type}={e.value}" for e in frame.entities) or "-"
        ie.append(
            f"{tag}\n  intent   : {frame.intent}\n  entities : {ents}\n  primary  : {frame.primary}\n"
            f"  yes/no   : {frame.yes_no}\n  parse    : {frame.parse_status}\n  source   : {frame.intent_source}\n"
            f"  note     : {'; '.join(frame.notes)}\n"
        )
        qq.append(f"{tag}\n  {query.text}\n")
        aa.append(
            f"{tag}\n  A: {res.answer}\n  sources: {', '.join(res.sources) or '-'}\n  found: {res.found}\n"
        )
    for name, rows in (
        ("semantic.txt", sem),
        ("intent-entity.txt", ie),
        ("query.txt", qq),
        ("answer.txt", aa),
    ):
        (Path(out_dir) / name).write_text("\n".join(rows), encoding="utf-8")
    print(
        f"[II] {len(sentences)} cau -> semantic/intent-entity/query/answer.txt trong {out_dir}"
    )
