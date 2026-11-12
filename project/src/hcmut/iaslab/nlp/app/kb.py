"""Knowledge base loader cho bo du lieu Course Assistant (data/course/*.txt).

Khong chua noi dung mon hoc: moi thong tin deu doc tu file du lieu.
"""
import re
from pathlib import Path


def _read(path):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _join(lines):
    return " ".join(l.strip() for l in lines if l.strip())


class KnowledgeBase:
    def __init__(self, kb_dir):
        d = Path(kb_dir)
        self.dir = d
        self.info = {}        # course_info: key -> text
        self.los = {}         # 'LO2.5' -> text
        self.chapters = {}    # 'CH03' -> dict
        self.weeks = {}       # 3 -> dict
        self.resources = []   # list of dict
        self.regs = {}        # key -> text
        self.faq = []         # list of dict(question, answer)
        self.entities = {}    # section -> list[(lhs, rhs)]
        self.intents = []
        self.assign = {}      # 'BTL01' / 'PART_I' -> dict
        self._load_info(_read(d / "course_info.txt"))
        self._load_topics(_read(d / "topics.txt"))
        self._load_schedule(_read(d / "schedule.txt"))
        self._load_resources(_read(d / "resources.txt"))
        self._load_regs(_read(d / "regulations.txt"))
        self._load_faq(_read(d / "faq.txt"))
        self._load_entities(_read(d / "entities.txt"))
        self._load_assign(_read(d / "assignments.txt"))

    # ---------- course_info ----------
    def _load_info(self, text):
        key, buf = None, []
        raw = {}
        for line in text.splitlines():
            m = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
            if m:
                if key:
                    raw[key] = buf
                key, buf = m.group(1), [m.group(2)] if m.group(2) else []
            elif key is not None:
                buf.append(line)
        if key:
            raw[key] = buf
        self.info = {k: _join(v) for k, v in raw.items()}
        self.info_raw = raw
        for sec in ("learning_outcomes", "detailed_learning_outcomes"):
            cur, parts = None, {}
            for line in raw.get(sec, []):
                m = re.match(r"^\s*(LO\d(?:\.\d)?)\s*-\s*(.*)$", line)
                if m:
                    cur = m.group(1)
                    parts[cur] = [m.group(2)]
                elif cur and line.strip():
                    parts[cur].append(line.strip())
            for k, v in parts.items():
                self.los[k] = " ".join(v)

    # ---------- topics ----------
    def _load_topics(self, text):
        for block in re.split(r"(?m)^(?=CHAPTER \d+:)", text):
            m = re.match(r"CHAPTER (\d+):\s*(.*)", block)
            if not m:
                continue
            ch = {"title_raw": m.group(2).strip(), "subtopics": [], "keywords": []}
            mode, summ = None, []
            for line in block.splitlines()[1:]:
                s = line.strip()
                if s.startswith("chapter_id:"):
                    ch["id"] = s.split(":", 1)[1].strip()
                elif s.startswith("part:"):
                    ch["part"] = s.split(":", 1)[1].strip()
                elif s.startswith("keywords:"):
                    ch["keywords"] = [k.strip() for k in s.split(":", 1)[1].split(",") if k.strip()]
                elif s == "summary:":
                    mode = "sum"
                elif s == "subtopics:":
                    mode = "sub"
                elif s and mode == "sum":
                    summ.append(s)
                elif s and mode == "sub":
                    ch["subtopics"].append(s)
            ch["summary"] = " ".join(summ)
            ch["num"] = int(m.group(1))
            self.chapters[ch["id"]] = ch

    def chapter_title(self, cid):
        t = self.chapters[cid]["title_raw"]
        if not t.isupper():
            return t
        small = {"and", "for", "of", "the", "to", "in"}
        words = [w.lower() for w in t.split()]
        return " ".join(w if (k and w in small) else w.capitalize() for k, w in enumerate(words))

    # ---------- schedule ----------
    def _load_schedule(self, text):
        for block in re.split(r"(?m)^(?=WEEK \d+)", text):
            m = re.match(r"WEEK (\d+)", block)
            if not m:
                continue
            wk = {"week": int(m.group(1)), "topics": [], "chapter": None}
            mode = None
            for line in block.splitlines()[1:]:
                s = line.strip()
                if s.startswith("topics:"):
                    mode = "t"
                elif s.startswith("- ") and mode == "t":
                    wk["topics"].append(s[2:])
                elif ":" in s:
                    k, v = s.split(":", 1)
                    mode = None
                    if k in ("part", "activity", "milestone"):
                        wk[k] = v.strip()
                    elif k == "chapter":
                        n = re.search(r"\d+", v)
                        wk["chapter"] = "CH%02d" % int(n.group()) if n else None
            self.weeks[wk["week"]] = wk

    def weeks_of_chapter(self, cid):
        return [w for w, d in sorted(self.weeks.items()) if d["chapter"] == cid]

    # ---------- resources ----------
    def _load_resources(self, text):
        for block in re.split(r"(?m)^(?=(?:TEXTBOOK|REFERENCE) \d+)", text):
            m = re.match(r"((?:TEXTBOOK|REFERENCE) \d+)", block)
            if not m:
                continue
            r = {"id": m.group(1)}
            for line in block.splitlines()[1:]:
                mm = re.match(r"^(\w+):\s*(.*)$", line.strip())
                if mm:
                    r[mm.group(1)] = mm.group(2)
                    if mm.group(1) == "study_note":
                        break
            self.resources.append(r)
        m = re.search(r"study_note:\s*(.*)\Z", text, re.S)
        self.study_note = " ".join(m.group(1).split()) if m else ""

    # ---------- regulations ----------
    def _load_regs(self, text):
        for m in re.finditer(r"(?ms)^([A-Za-z_]+):\s*\n(.*?)(?=^\w+:\s*$|\Z)", text):
            self.regs[m.group(1)] = " ".join(m.group(2).split())

    # ---------- faq ----------
    def _load_faq(self, text):
        for block in re.split(r"(?m)^(?=Q\d+:)", text):
            q = re.search(r"question:\s*(.*)", block)
            a = re.search(r"answer:\s*(.*?)\s*\Z", block, re.S)
            if q and a:
                self.faq.append({"question": q.group(1).strip(), "answer": " ".join(a.group(1).split()),
                                 "id": re.match(r"(Q\d+)", block).group(1)})

    # ---------- entities ----------
    def _load_entities(self, text):
        sec = None
        for line in text.splitlines():
            s = line.strip()
            if not s or set(s) <= {"="}:
                continue
            if re.fullmatch(r"[A-Z ]+", s) and "=" not in s and s != "COURSE ASSISTANT ENTITY LEXICON":
                sec = s
                continue
            if sec == "INTENTS":
                self.intents.append(s)
            elif sec and "=" in s:
                l, r = s.split("=", 1)
                self.entities.setdefault(sec, []).append((l.strip(), r.strip()))

    # ---------- assignments ----------
    def _load_assign(self, text):
        for block in re.split(r"(?m)^(?=(?:ASSIGNMENT|PART) \S+\s*$)", text):
            m = re.match(r"(ASSIGNMENT|PART) (\S+)[ \t]*(?:\n|\Z)", block)
            if not m:
                continue
            key = m.group(2) if m.group(1) == "ASSIGNMENT" else "PART_" + m.group(2)
            d = {}
            for line in block.splitlines()[1:]:
                mm = re.match(r"^(\w+):\s*(.*)$", line.strip())
                if mm:
                    d[mm.group(1)] = mm.group(2)
            self.assign[key] = d

    @property
    def has_assignments(self):
        return bool(self.assign)
