"""Phan II (Classical NLP): xay dung truy van toi KB, truy xuat va sinh cau tra loi.

Moi cau tra loi chi duoc tao tu du lieu trong KB; khong co du lieu -> thong bao khong tim thay.
"""
import re
from dataclasses import dataclass, field

NOT_FOUND = "Xin lỗi, tôi không tìm thấy thông tin này trong dữ liệu của môn học."
OUT_OF_SCOPE = ("Xin lỗi, câu hỏi này nằm ngoài phạm vi của Course Assistant. "
                "Tôi không tìm thấy thông tin này trong dữ liệu của môn học.")
NEED_CONTEXT = ("Câu hỏi này tham chiếu tới lượt hội thoại trước, nhưng chưa có ngữ cảnh để xác định "
                "đối tượng được nhắc tới (xử lý ở Phần III).")


@dataclass
class Query:
    op: str
    text: str
    params: dict = field(default_factory=dict)


@dataclass
class Result:
    answer: str
    sources: list = field(default_factory=list)
    found: bool = True


def build_query(f):
    """Semantic frame -> truy van (pseudo-SQL) tren KB."""
    p, a = f.predicate, f.args
    q = lambda op, text, **kw: Query(op, text, kw)
    if p == "GET_CREDIT":
        return q("info", f"SELECT credits FROM course_info WHERE course_id='{a['course']}'", field="credits")
    if p == "GET_ECTS":
        return q("info", f"SELECT ects FROM course_info WHERE course_id='{a['course']}'", field="ects")
    if p == "GET_HOURS":
        return q("hours", f"SELECT *_hours FROM course_info WHERE course_id='{a['course']}'")
    if p == "GET_ASSESSMENT":
        return q("assessment", f"SELECT assessment FROM course_info WHERE course_id='{a['course']}'")
    if p == "GET_MAIN_PARTS":
        return q("parts", f"SELECT main_parts FROM course_info WHERE course_id='{a['course']}'")
    if p == "GET_DEPARTMENT":
        return q("info", f"SELECT department FROM course_info WHERE course_id='{a['course']}'", field="department")
    if p == "GET_SEMESTER":
        return q("info", f"SELECT semester FROM course_info WHERE course_id='{a['course']}'", field="semester")
    if p == "GET_FORMAT":
        return q("info", f"SELECT format FROM course_info WHERE course_id='{a['course']}'", field="format")
    if p == "GET_COURSE_DESCRIPTION":
        return q("info", f"SELECT course_description FROM course_info WHERE course_id='{a['course']}'", field="course_description")
    if p == "GET_COURSE_INFO":
        return q("summary", f"SELECT course_name, credits, semester FROM course_info WHERE course_id='{a['course']}'")
    if p in ("GET_EXAM_DURATION", "GET_EXAM_FORMAT"):
        return q("exam", f"SELECT {a['exam'].lower()} FROM course_info.exam_format", exam=a["exam"], duration=p.endswith("DURATION"))
    if p == "GET_WEEK_CONTENT":
        n = int(a["week"].split("_")[1])
        return q("week", f"SELECT part, chapter, topics, activity, milestone FROM schedule WHERE week={n}", week=n)
    if p == "GET_CHAPTER_WEEKS":
        return q("chapter_weeks", f"SELECT week FROM schedule WHERE chapter='{a['chapter']}'", chapter=a["chapter"])
    if p == "GET_TOPIC_WEEKS":
        return q("topic_weeks", f"SELECT week FROM schedule JOIN topics ON chapter WHERE topic MATCH '{a['topic']}'", topic=a["topic"])
    if p == "GET_CHAPTER_CONTENT":
        return q("chapter", f"SELECT title, part, summary, subtopics FROM topics WHERE chapter_id='{a['chapter']}'", chapter=a["chapter"])
    if p == "GET_TOPIC_CHAPTER":
        return q("topic_chapter", f"SELECT chapter_id, title FROM topics WHERE keywords|subtopics|summary MATCH '{a['topic']}' ORDER BY match_tier DESC", topic=a["topic"])
    if p == "CHECK_CHAPTER_COVERS":
        return q("covers", f"SELECT subtopics FROM topics WHERE chapter_id='{a['chapter']}' AND subtopics|summary MATCH '{a['topic']}'", chapter=a["chapter"], topic=a["topic"])
    if p == "GET_PART_CHAPTERS":
        return q("part_chapters", f"SELECT chapter_id, title FROM topics WHERE part=<{a['part']}>", part=a["part"])
    if p == "GET_LO":
        return q("lo", f"SELECT text FROM learning_outcomes WHERE id='{a['lo']}'", lo=a["lo"])
    if p == "GET_LO_CHAPTER":
        return q("lo_chapter", f"SELECT chapter_id, title FROM topics WHERE lo_content MATCH '{a['lo']}'", lo=a["lo"])
    if p == "GET_LO_ALL":
        return q("lo_all", "SELECT id, text FROM learning_outcomes")
    if p == "GET_ASSIGNMENT":
        return q("assignment", f"SELECT title, description, parts FROM assignments WHERE id='{a['assignment']}'")
    if p in ("GET_ASSIGNMENT_PART", "CHECK_ASSIGNMENT_PART"):
        return q("assign_part", f"SELECT name, description, outputs FROM assignments WHERE part='{a['part']}'", part=a["part"], check=p.startswith("CHECK"))
    if p == "GET_DEADLINE":
        return q("deadline", f"SELECT week, milestone FROM schedule WHERE milestone LIKE 'BTL%'  -- assignment='{a['assignment']}'")
    if p == "GET_RESOURCE":
        if "chapter" in a:
            return q("resource", f"SELECT id, title, author, year, usage FROM resources WHERE chapter='{a['chapter']}'", topic="*", chapter=a["chapter"])
        flt = a.get("topic", "*")
        return q("resource", f"SELECT id, title, author, year, usage FROM resources WHERE usage|title MATCH '{flt}'", topic=flt)
    if p == "GET_RULE":
        return q("rule", f"SELECT text FROM regulations WHERE key='{a['rule']}'", rule=a["rule"])
    if p == "GET_INSTRUCTOR":
        return q("info", f"SELECT instructor FROM course_info WHERE course_id='{a['course']}'", field="instructor")
    return Query("none", "-- khong tao truy van (" + p + ")")


# ------------------------------------------------------------------ thuc thi
SEARCH_RULES = {"RESOURCE": {"STATISTICAL_NLP": r"statistical", "PYTHON_NLP": r"python"}}

LO_VI_EN_GLOSSARY = {
    "tổng quan": ["introduction", "organization", "overview"],
    "tổ chức": ["organization"],
    "cấu trúc": ["organization", "structure"],
    "ứng dụng": ["applications", "application"],
    "đánh giá": ["evaluating", "evaluation"],
    "cú pháp tiếng anh": ["english syntax"],
    "cú pháp": ["syntax"],
    "văn phạm gia tố": ["augmented grammar"],
    "văn phạm ngữ nghĩa": ["semantic grammar"],
    "quan hệ văn phạm": ["grammatical relations"],
    "văn phạm cho ngôn ngữ tự nhiên": ["grammar for natural language"],
    "văn phạm": ["grammar"],
    "phân tích cú pháp": ["parsing", "parser"],
    "nét": ["feature", "features"],
    "ngôn ngữ tự nhiên": ["natural language"],
    "xác suất": ["probability", "statistical"],
    "thống kê": ["statistical"],
    "nhập nhằng ngữ nghĩa": ["semantic ambiguity", "ambiguity resolution"],
    "nhập nhằng": ["ambiguity", "disambiguation"],
    "mức câu": ["sentence level"],
    "dạng luận lý": ["logical form"],
    "tạo tự động": ["compositionality", "semantic interpretation"],
    "diễn dịch ngữ nghĩa": ["semantic interpretation"],
    "câu tự nhiên": ["sentences", "natural language"],
    "cơ sở tri thức": ["knowledge-based", "knowledge representation"],
    "biểu diễn tri thức": ["knowledge representation"],
    "suy diễn": ["reasoning", "inference"],
    "hỏi đáp": ["question answering", "qa"],
    "ngữ cảnh diễn ngôn cục bộ": ["local discourse context", "discourse context"],
    "tham chiếu": ["reference", "anaphora"],
}

LO_PART_MAP = {
    "1": "Introduction",
    "2": "Syntactic Processing",
    "3": "Semantic Interpretation",
    "4": "Context and World Knowledge",
}


class Engine:
    def __init__(self, kb, linker):
        self.kb, self.linker = kb, linker

    # ----- tim chuong theo chu de (keywords > subtopic > summary)
    def chapters_for(self, topic):
        terms = self.linker.search_terms.get(topic, [topic.replace("_", " ")])
        hits = {}
        for cid, ch in self.kb.chapters.items():
            tier = 0
            for term in terms:
                rx = re.compile(r"(?<!\w)%s(?!\w)" % re.escape(term), re.I)
                if any(rx.fullmatch(k.strip()) or rx.search(k) for k in ch["keywords"]):
                    tier = max(tier, 3)
                if any(rx.search(re.sub(r"^[\d.]+\s*", "", s)) for s in ch["subtopics"]):
                    tier = max(tier, 2)
                if rx.search(ch["summary"]):
                    tier = max(tier, 1)
            if tier:
                hits[cid] = tier
        if not hits:
            return [], []
        top = max(hits.values())
        main = sorted(c for c, t in hits.items() if t == top)
        rest = sorted(c for c, t in hits.items() if t < top and t >= 2)
        return main, rest

    @staticmethod
    def disp(topic):
        t = topic.replace("_", " ")
        return t if len(t) <= 5 else t.lower()

    def ch_label(self, cid):
        ch = self.kb.chapters[cid]
        return f"Chương {ch['num']} ({self.kb.chapter_title(cid)})"

    @staticmethod
    def _list(items):
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " và " + items[-1]

    def execute(self, f, q):
        kb = self.kb
        if f.intent == "UNKNOWN":
            return Result(OUT_OF_SCOPE, ["regulations.txt:OUT_OF_SCOPE"], found=False)
        if f.intent == "CONTEXT_DEPENDENT":
            return Result(NEED_CONTEXT, [], found=False)
        fn = getattr(self, "_x_" + q.op, None)
        res = fn(f, q) if fn else None
        if res is None:
            res = self._faq_fallback(f)
        return res or Result(NOT_FOUND, [], found=False)

    # ----- thong tin mon hoc
    def _x_info(self, f, q):
        v = self.kb.info.get(q.params["field"])
        if not v:
            return None
        name = self.kb.info.get("course_name", "")
        cid = self.kb.info.get("course_id", "")
        fld = q.params["field"]
        if fld == "credits":
            return Result(f"Môn {name} ({cid}) có {v} tín chỉ.", ["course_info.txt:credits"])
        return Result(f"{fld}: {v}", [f"course_info.txt:{fld}"])

    def _x_summary(self, f, q):
        i = self.kb.info
        return Result(f"{i['course_name']} ({i['course_id']}), {i['credits']} tín chỉ, học kỳ {i['semester']}, {i['department']}.",
                      ["course_info.txt"])

    def _x_hours(self, f, q):
        i = self.kb.info
        if "total_course_hours" not in i:
            return None
        return Result(f"Tổng {i['total_course_hours']} giờ: {i['lecture_hours']} giờ lý thuyết, {i['major_assignment_hours']} giờ bài tập lớn, "
                      f"{i['self_study_hours']} giờ tự học.", ["course_info.txt:hours"])

    def _x_assessment(self, f, q):
        i = self.kb.info
        keys = [("class_attention_and_assignments", "chuyên cần và bài tập"), ("major_assignment", "bài tập lớn"),
                ("midterm_exam", "thi giữa kỳ"), ("final_exam", "thi cuối kỳ")]
        if not all(k in i for k, _ in keys):
            return None
        parts = [f"{lbl} {i[k].replace(' percent', '%')}" for k, lbl in keys]
        return Result("Thành phần điểm: " + "; ".join(parts) + ".", ["course_info.txt:assessment"])

    def _x_parts(self, f, q):
        raw = self.kb.info_raw.get("main_parts")
        if not raw:
            return None
        lines = [l.strip() for l in raw if l.strip()]
        return Result("Môn học gồm các phần chính: " + "; ".join(lines) + ".", ["course_info.txt:main_parts"])

    def _x_exam(self, f, q):
        ex = q.params["exam"]
        v = self.kb.info.get(ex.lower())
        if not v:
            return None
        label = "Midterm" if ex == "MIDTERM" else "Final"
        m = re.search(r"(\d+)\s*minutes", v)
        if q.params["duration"] and m:
            return Result(f"Theo đề cương, {label.lower()} có thời lượng {m.group(1)} phút ({v}).", ["course_info.txt:exam_format"])
        return Result(f"{label}: {v}.", ["course_info.txt:exam_format"])

    # ----- lich hoc
    def _x_week(self, f, q):
        w = self.kb.weeks.get(q.params["week"])
        if not w:
            return None
        s = f"Tuần {w['week']}"
        if w["chapter"]:
            s += f" học {self.ch_label(w['chapter'])}"
        s += f" (phần {w.get('part', '?')}). Nội dung: " + "; ".join(w["topics"]) + "."
        if w.get("activity"):
            s += f" Hoạt động: {w['activity']}."
        if w.get("milestone"):
            s += f" Mốc: {w['milestone']}."
        return Result(s, [f"schedule.txt:WEEK {w['week']:02d}"])

    def _x_chapter_weeks(self, f, q):
        cid = q.params["chapter"]
        if cid not in self.kb.chapters:
            return None
        ws = self.kb.weeks_of_chapter(cid)
        if not ws:
            return None
        return Result(f"{self.ch_label(cid)} được học vào tuần {self._list(str(w) for w in ws)}.",
                      [f"schedule.txt:WEEK {w:02d}" for w in ws])

    def _x_topic_weeks(self, f, q):
        main, _ = self.chapters_for(q.params["topic"])
        ws = sorted({w for c in main for w in self.kb.weeks_of_chapter(c)})
        if not ws:
            return None
        return Result(f"{self.disp(q.params['topic'])} nằm ở {self._list(self.ch_label(c) for c in main)}, học vào tuần {self._list(str(w) for w in ws)}.",
                      [f"schedule.txt:WEEK {w:02d}" for w in ws])

    # ----- chu de / chuong
    def _x_chapter(self, f, q):
        cid = q.params["chapter"]
        ch = self.kb.chapters.get(cid)
        if not ch:
            return None
        subs = "; ".join(ch["subtopics"])
        return Result(f"{self.ch_label(cid)}, phần {ch['part']}. Tóm tắt: {ch['summary']} Các mục: {subs}.", [f"topics.txt:{cid}"])

    def _x_topic_chapter(self, f, q):
        topic = q.params["topic"]
        main, rest = self.chapters_for(topic)
        if not main:
            return None
        s = f"{self.disp(topic)} được trình bày chủ yếu trong {self._list(self.ch_label(c) for c in main)}."
        if rest:
            s += f" Ngoài ra còn được nhắc đến trong {self._list(self.ch_label(c) for c in rest)}."
        return Result(s, [f"topics.txt:{c}" for c in main + rest])

    def _x_covers(self, f, q):
        cid, topic = q.params["chapter"], q.params["topic"]
        ch = self.kb.chapters.get(cid)
        if not ch:
            return None
        terms = self.linker.search_terms.get(topic, [topic.replace("_", " ")])
        rx = re.compile("|".join(r"(?<!\w)%s(?!\w)" % re.escape(t) for t in terms), re.I)
        subs = [s for s in ch["subtopics"] if rx.search(s)]
        in_kw = any(rx.search(k) for k in ch["keywords"])
        in_sum = bool(rx.search(ch["summary"]))
        if subs or in_kw or in_sum:
            ev = ("Mục liên quan: " + "; ".join(subs) + ". ") if subs else ""
            return Result(f"Có. {self.ch_label(cid)} đề cập đến {self.disp(topic)}. {ev}Tóm tắt chương: {ch['summary']}",
                          [f"topics.txt:{cid}"])
        return Result(f"Trong dữ liệu, {self.ch_label(cid)} không có mục nào nhắc đến {self.disp(topic)}.", [f"topics.txt:{cid}"])

    def _x_part_chapters(self, f, q):
        roman = q.params["part"].rsplit("_", 1)[1]
        order = {"I": "Syntactic Processing", "II": "Semantic Interpretation", "III": "Context and World Knowledge"}
        part = self.kb.info_raw.get("main_parts", [])
        name = next((re.sub(r"^Part \w+ - ", "", l.strip()) for l in part if l.strip().startswith(f"Part {roman} ")), None)
        if not name:
            return None
        cs = [c for c, ch in self.kb.chapters.items() if ch.get("part") == name]
        if not cs:
            return None
        return Result(f"Phần {roman} ({name}) gồm {self._list(self.ch_label(c) for c in sorted(cs))}.", ["course_info.txt:main_parts", "topics.txt"])

    # ----- LO
    def _x_lo(self, f, q):
        lo = q.params["lo"]
        t = self.kb.los.get(lo)
        if not t:
            return None
        return Result(f"L.O.{lo[2:]}: {t}", ["course_info.txt:detailed_learning_outcomes" if "." in lo else "course_info.txt:learning_outcomes"])

    def _x_lo_all(self, f, q):
        if not self.kb.los:
            return None
        top = {k: v for k, v in self.kb.los.items() if "." not in k}
        return Result("Chuẩn đầu ra: " + " ".join(f"L.O.{k[2:]}: {v}" for k, v in top.items()), ["course_info.txt:learning_outcomes"])

    def _x_lo_chapter(self, f, q):
        lo = q.params["lo"]
        lo_text = self.kb.los.get(lo)
        if not lo_text or not self.kb.chapters:
            return None
        lo_low = lo_text.lower()
        part_name = LO_PART_MAP.get(lo[2:3], "")

        # Top-level LO (vd LO2): tra ve cac chuong trong phan tuong ung.
        if "." not in lo and part_name:
            cs = sorted(c for c, ch in self.kb.chapters.items() if ch.get("part") == part_name)
            if not cs:
                return None
            return Result(
                f"L.O.{lo[2:]} ({lo_text}) bao phủ {self._list(self.ch_label(c) for c in cs)}.",
                ["course_info.txt:learning_outcomes"] + [f"topics.txt:{c}" for c in cs],
            )

        # Detailed LO (vd LO2.5): doi chieu tu khoa song ngu + phan mon hoc voi topics.txt.
        en_terms = []
        for vi, ens in LO_VI_EN_GLOSSARY.items():
            if vi in lo_low:
                en_terms.extend(ens)
        en_terms.extend(re.findall(r"[a-z]{3,}", lo_low))

        scored = {}
        for cid, ch in self.kb.chapters.items():
            score = 4 if part_name and ch.get("part") == part_name else 0
            title = ch["title_raw"].lower()
            kws = " ".join(ch["keywords"]).lower()
            summ = ch["summary"].lower()
            subs = " ".join(ch["subtopics"]).lower()
            for term in set(en_terms):
                rx = re.compile(r"(?<!\w)%s(?!\w)" % re.escape(term), re.I)
                if rx.search(title):
                    score += 3
                if rx.search(kws):
                    score += 3
                if rx.search(summ):
                    score += 2
                if rx.search(subs):
                    score += 1
            scored[cid] = score

        best_score = max(scored.values(), default=0)
        if best_score <= 0:
            return None
        best = sorted(c for c, s in scored.items() if s == best_score)
        return Result(
            f"L.O.{lo[2:]} ({lo_text}) thuộc {self._list(self.ch_label(c) for c in best)}, phần {self.kb.chapters[best[0]]['part']}.",
            ["course_info.txt:detailed_learning_outcomes"] + [f"topics.txt:{c}" for c in best],
        )

    # ----- bai tap lon
    def _x_assignment(self, f, q):
        a = self.kb.assign.get("BTL01")
        if not a:
            return None
        parts = [f"Phần {k[5:]}: {v['name']}" for k, v in self.kb.assign.items() if k.startswith("PART_")]
        return Result(f"Bài tập lớn: {a['title']}. {a['description']} Gồm: " + "; ".join(parts) + ".", ["assignments.txt:BTL01"])

    def _x_assign_part(self, f, q):
        key = q.params["part"]
        a = self.kb.assign.get(key)
        if not a:
            return None
        s = f"Phần {key[5:]} của bài tập lớn ({a['name']}): {a['description']} Output: {a['outputs']}."
        return Result(("Có. " if q.params["check"] else "") + s, [f"assignments.txt:{key}"])

    def _x_deadline(self, f, q):
        ms = [(w, d["milestone"]) for w, d in sorted(self.kb.weeks.items()) if d.get("milestone", "").upper().startswith("BTL")]
        if not ms:
            return None
        s = "Dữ liệu chỉ có các mốc theo tuần (không có ngày nộp cụ thể): " + "; ".join(f"tuần {w}: {m}" for w, m in ms) + "."
        return Result(s, [f"schedule.txt:WEEK {w:02d}" for w, _ in ms])

    # ----- tai lieu
    def _x_resource(self, f, q):
        topic = q.params["topic"]
        chapter = q.params.get("chapter")
        rx = SEARCH_RULES["RESOURCE"].get(topic)
        items, note = self.kb.resources, ""
        extra_sources = []
        if chapter:
            ch = self.kb.chapters.get(chapter)
            if not ch:
                return None
            note = (
                f"Dữ liệu không gắn tài liệu riêng cho {self.ch_label(chapter)}; "
                "có thể tham khảo các tài liệu nền tảng trong danh sách môn học: "
            )
            extra_sources.append(f"topics.txt:{chapter}")
        elif rx:
            items = [r for r in items if re.search(rx, r.get("title", "") + " " + r.get("usage", ""), re.I)]
            if not items:
                return None
        elif topic != "*":
            note = f"Dữ liệu không gắn tài liệu theo từng chủ đề ({topic.replace('_', ' ')}); danh sách tài liệu chung của môn học: "
        else:
            note = "Danh sách tài liệu của môn học: "
        def fmt(r):
            who = r.get("author") or r.get("authors") or ""
            y = f", {r['year']}" if r.get("year") else ""
            return f"{r['id']}: {r['title']}" + (f" ({who}{y})" if who else (f" ({r['year']})" if r.get("year") else "")) + f" - {r.get('usage', '')}"
        return Result(note + "; ".join(fmt(r) for r in items) + ".", extra_sources + ["resources.txt:" + r["id"] for r in items])

    # ----- quy dinh
    def _x_rule(self, f, q):
        key = q.params["rule"]
        regs = {k.lower(): k for k in self.kb.regs}
        if key == "ALL":
            ks = [k for k in self.kb.regs if k != "OUT_OF_SCOPE"]
            return Result("Các quy định: " + " ".join(f"[{k}] {self.kb.regs[k]}" for k in ks), ["regulations.txt"])
        real = regs.get(key.lower())
        if not real:
            return None
        return Result(f"Quy định ({real}): {self.kb.regs[real]}", [f"regulations.txt:{real}"])

    # ----- FAQ (du phong: cung bo entity + tu trung nhau)
    def _faq_fallback(self, f):
        want = {e.value for e in f.entities if e.type not in ("PARTREF",)}
        qt = set(re.findall(r"\w+", f.sentence.lower()))
        best, score = None, 0.0
        for item in self.kb.faq:
            have = {e.value for e in self.linker.extract(item["question"]) if e.type not in ("PARTREF",)}
            if not want or have != want:
                continue
            it = set(re.findall(r"\w+", item["question"].lower()))
            j = len(qt & it) / max(1, len(qt | it))
            if j > score:
                best, score = item, j
        if best and score >= 0.4:
            return Result(best["answer"], [f"faq.txt:{best['id']}"])
        return None
