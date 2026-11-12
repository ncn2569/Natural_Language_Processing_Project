"""Phan II - Lua chon 2: Retrieval-Augmented Generation (RAG).

User Question -> Retrieve (BM25 tu cai dat, mo rong truy van bang entity/tu dien cua Phan I)
-> Retrieved Context -> Prompt -> LLM (neu co API key) / trich xuat cau co can cu (neu khong) -> Answer + nguon.

Chong bia (hallucination):
  - cong ngoai pham vi (dung bo phan loai UNKNOWN cua Phan II-Classical) + nguong diem retrieval;
  - prompt bat buoc chi dung ngu canh va phai noi "khong tim thay" neu thieu;
  - kiem tra hau kiem: so lieu/ma trong cau tra loi phai xuat hien trong ngu canh.

LLM: bien moi truong ANTHROPIC_API_KEY (+ RAG_MODEL, mac dinh claude-haiku-5-5). Khong co key -> che do extractive.
"""

import json
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from .kb import KnowledgeBase
from .nlu import EntityLinker, build_frame, norm

# from .generate import PROJECT_ROOT, default_dirs
from .part2 import PROJECT_ROOT, default_dirs
from .qa import NEED_CONTEXT, NOT_FOUND, OUT_OF_SCOPE, Engine

STOP = set(
    "môn ai là gì của có những nào không được về và các cho tôi mình bạn hãy thì ở này đó khi nào bao nhiêu mấy một the of and is a an to in what which how".split()
)
TOKEN_RE = re.compile(r"\w+(?:\.\w+)*", re.U)
TOP_K = 4
MIN_SCORE = 1.0


def tokenize(text):
    t = norm(text)
    t = re.sub(r"\bl\.?o\.?\s*(\d)", r"lo\1", t)
    return [w for w in TOKEN_RE.findall(t) if w not in STOP]


# ------------------------------------------------------------------ chunking
def build_chunks(kb):
    ch = []
    add = lambda cid, src, text: ch.append(
        {"id": cid, "source": src, "text": " ".join(text.split())}
    )
    i = kb.info
    add(
        "course_info:general",
        "course_info.txt",
        f"Thông tin môn học. Mã môn {i.get('course_id')}. Tên {i.get('course_name')} ({i.get('course_name_en')}). Học kỳ {i.get('semester')}. "
        f"Số tín chỉ credits: {i.get('credits')}. ECTS: {i.get('ects')}. Loại: {i.get('course_type')}. Khoa: {i.get('department')}. "
        f"Hình thức: {i.get('format')}. Tổng giờ {i.get('total_course_hours')}, lý thuyết {i.get('lecture_hours')}, bài tập lớn {i.get('major_assignment_hours')}, tự học {i.get('self_study_hours')}.",
    )
    add(
        "course_info:description",
        "course_info.txt",
        f"Mô tả môn học: {i.get('course_description')} Mục tiêu: {i.get('Mục tiêu', '')} Các phần chính main parts: "
        + "; ".join(l.strip() for l in kb.info_raw.get("main_parts", []) if l.strip()),
    )
    add(
        "course_info:assessment",
        "course_info.txt",
        f"Đánh giá và thi. Chuyên cần và bài tập {i.get('class_attention_and_assignments')}; bài tập lớn {i.get('major_assignment')}; "
        f"giữa kỳ midterm {i.get('midterm_exam')}; cuối kỳ final {i.get('final_exam')}. Hình thức thi: midterm {i.get('midterm')}; final {i.get('final')}.",
    )
    for k, t in kb.los.items():
        add(f"lo:{k}", "course_info.txt", f"Chuẩn đầu ra L.O.{k[2:]} ({k}): {t}")
    for cid, c in kb.chapters.items():
        add(
            f"topic:{cid}",
            "topics.txt",
            f"Chương {c['num']} ({cid}): {kb.chapter_title(cid)}. Phần {c['part']}. Từ khóa: {', '.join(c['keywords'])}. Tóm tắt: {c['summary']} Các mục: "
            + "; ".join(c["subtopics"]),
        )
    for w, d in kb.weeks.items():
        add(
            f"schedule:WEEK_{w:02d}",
            "schedule.txt",
            f"Tuần {w} week {w}. Phần {d.get('part')}. "
            + (f"Chương {int(d['chapter'][2:])}. " if d["chapter"] else "")
            + "Nội dung: "
            + "; ".join(d["topics"])
            + ". "
            + (f"Hoạt động: {d['activity']}. " if d.get("activity") else "")
            + (f"Mốc: {d['milestone']}." if d.get("milestone") else ""),
        )
    for r in kb.resources:
        add(
            f"resource:{r['id']}",
            "resources.txt",
            f"Tài liệu {r['id']}: {r.get('title')}. Tác giả: {r.get('author') or r.get('authors', '')}. Năm {r.get('year', '')}. Mục đích: {r.get('usage', '')}.",
        )
    if kb.study_note:
        add(
            "resource:study_note",
            "resources.txt",
            "Ghi chú học tập tài liệu: " + kb.study_note,
        )
    for k, t in kb.regs.items():
        add(f"regulation:{k}", "regulations.txt", f"Quy định {k}: {t}")
    for f in kb.faq:
        add(f"faq:{f['id']}", "faq.txt", f"Hỏi: {f['question']} Đáp: {f['answer']}")
    for k, a in kb.assign.items():
        add(
            f"assignment:{k}",
            "assignments.txt",
            f"Bài tập lớn {k}: " + " ".join(f"{n}: {v}." for n, v in a.items()),
        )
    return ch


# ------------------------------------------------------------------ BM25
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d["text"]) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, q_tokens):
        out = []
        for tf, toks in zip(self.tf, self.toks):
            s = 0.0
            for w in q_tokens:
                if w in tf:
                    f = tf[w]
                    s += (
                        self.idf[w]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg))
                    )
            out.append(s)
        return out


# ------------------------------------------------------------------ prompt / LLM
SYSTEM = (
    "Bạn là Course Assistant của môn Xử lý ngôn ngữ tự nhiên (CO3085). Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp. "
    'Nếu ngữ cảnh không có thông tin để trả lời, hãy nói đúng câu: "'
    + NOT_FOUND
    + '" và không suy đoán. '
    "Trả lời ngắn gọn bằng tiếng Việt và ghi nguồn dạng [id] sau mỗi ý."
)


def build_prompt(question, hits):
    ctx = "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
    return f"NGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"


def call_llm(prompt):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    body = json.dumps(
        {
            "model": os.environ.get("RAG_MODEL", "claude-haiku-5-5"),
            "max_tokens": 400,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"]).strip()
    except Exception as e:  # mang loi/het han muc -> roi ve extractive
        print("[RAG] LLM loi, dung extractive:", e)
        return None


def extractive_answer(question, hits):
    """Khong LLM: chunk tot nhat neu ngan; neu dai thi chon cac cau co nhieu tu khop nhat. Luon kem [nguon]."""
    q = set(tokenize(question))
    best = hits[0]
    body = lambda h: re.sub(r"^Hỏi: .*? Đáp: ", "", h["text"])
    if best["id"].startswith("faq:") or len(best["text"]) <= 450:
        return f"{body(best)} [{best['id']}]", [best]
    sents = []
    for s_ in re.split(r"(?<=[.;])\s+", best["text"]):
        if len(tokenize(s_)) >= 4:
            sents.append((len(q & set(tokenize(s_))), s_))
    sents.sort(key=lambda x: -x[0])
    chosen = [x for x in sents[:2] if x[0] > 0] or sents[:1]
    return " ".join(s_ for _, s_ in chosen) + f" [{best['id']}]", [best]


def grounded(answer, hits):
    """Kiem tra hau kiem: moi con so trong cau tra loi phai co trong ngu canh."""
    ctx = " ".join(h["text"] for h in hits)
    body = re.sub(r"\[[^\]]*\]", " ", answer)  # bo the nguon [faq:Q14]
    return [n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in ctx]


# ------------------------------------------------------------------ he thong
class RagAssistant:
    def __init__(self, kb_dir=None):
        # self.kb = KnowledgeBase(kb_dir or PROJECT_ROOT / "data" / "course")
        self.kb = KnowledgeBase(kb_dir or PROJECT_ROOT / "data")
        self.linker = EntityLinker(self.kb)
        self.chunks = build_chunks(self.kb)
        self.index = BM25(self.chunks)
        self.engine = Engine(self.kb, self.linker)

    def expand(self, question):
        toks = tokenize(question)
        extra = []
        for e in self.linker.extract(question):
            if e.type in ("TOPIC", "RESOURCE_TOPIC"):
                for term in self.linker.search_terms.get(e.value, []):
                    extra += tokenize(term)
        return toks + extra

    def _faq_entities(self):
        if not hasattr(self, "_fe"):
            self._fe = {
                c["id"]: {
                    e.value
                    for e in self.linker.extract(c["text"].split(" Đáp:")[0])
                    if e.type != "PARTREF"
                }
                for c in self.chunks
                if c["id"].startswith("faq:")
            }
        return self._fe

    def _boost(self, question, scores, frame=None):
        """Ket hop retrieval tu vung voi entity cua Phan II: chunk dung thuc the duoc cong diem,
        FAQ noi ve thuc the khac bi tru diem."""
        ents = self.linker.extract(question)
        vals = {e.value for e in ents if e.type != "PARTREF"}
        want = {}
        for e in ents:
            if e.type == "WEEK":
                want["schedule:" + e.value] = 8.0
            elif e.type == "CHAPTER":
                want["topic:" + e.value] = 8.0
            elif e.type == "LO":
                want["lo:" + e.value] = 8.0
            elif e.type == "TOPIC":
                for cid in self.engine.chapters_for(e.value)[0]:
                    want["topic:" + cid] = 5.0
        if frame is not None:
            a = frame.args
            if str(a.get("part", "")).startswith("PART_"):
                want["assignment:" + a["part"]] = 12.0
            elif "assignment" in a:
                want["assignment:BTL01"] = 12.0
            if a.get("rule") and a["rule"] != "ALL":
                for k in self.kb.regs:
                    if k.lower() == a["rule"].lower():
                        want["regulation:" + k] = 8.0
            if "exam" in a:
                want["course_info:assessment"] = 3.0
        fe = self._faq_entities()
        not_res = frame is not None and frame.intent != "GET_RESOURCE"
        for i, c in enumerate(self.chunks):
            if c["id"] in want:
                scores[i] += want[c["id"]]
            elif not_res and c["id"].startswith("resource:"):
                scores[i] *= 0.4
            elif c["id"] in fe and vals and fe[c["id"]] != vals:
                scores[i] *= 0.4
        return scores

    def missing(self, frame):
        """Thuc the/truong duoc hoi co ton tai trong KB khong? (chong tra loi lac de khi cau hoi toi thong tin khong co)"""
        a, kb = frame.args, self.kb
        if "chapter" in a and a["chapter"] not in kb.chapters:
            return a["chapter"]
        if "week" in a and int(a["week"].split("_")[1]) not in kb.weeks:
            return a["week"]
        if "lo" in a and a["lo"] not in kb.los:
            return a["lo"]
        if frame.intent == "GET_INSTRUCTOR" and "instructor" not in kb.info:
            return "instructor"
        return None

    def retrieve(self, question, k=TOP_K, frame=None):
        scores = self._boost(question, self.index.score(self.expand(question)), frame)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        hits = [
            dict(self.chunks[i], score=round(scores[i], 3))
            for i in order
            if scores[i] > 0
        ]
        return [
            h for h in hits if hits and h["score"] >= 0.5 * hits[0]["score"]
        ]  # bo chunk qua yeu so voi top-1

    def ask(self, question):
        frame = build_frame(
            question, self.linker
        )  # tan dung NLU Phan II de cong pham vi
        hits = self.retrieve(question, frame=frame)
        info = {
            "question": question,
            "hits": hits,
            "prompt": "",
            "mode": "-",
            "gate": frame.intent,
        }
        miss = self.missing(frame)
        if miss:
            return dict(
                info,
                answer=NOT_FOUND,
                sources=[],
                mode=f"gate:not-in-kb({miss})",
                warn=[],
            )
        if frame.intent == "UNKNOWN":
            return dict(
                info, answer=OUT_OF_SCOPE, sources=[], mode="gate:out-of-scope", warn=[]
            )
        if frame.intent == "CONTEXT_DEPENDENT":
            return dict(
                info,
                answer=NEED_CONTEXT,
                sources=[],
                mode="gate:needs-context",
                warn=[],
            )
        if not hits or hits[0]["score"] < MIN_SCORE:
            return dict(
                info, answer=NOT_FOUND, sources=[], mode="gate:low-score", warn=[]
            )
        prompt = build_prompt(question, hits)
        ans = call_llm(prompt)
        if ans is not None:
            mode, used = "llm", hits
        else:
            (ans, used), mode = extractive_answer(question, hits), "extractive"
        warn = grounded(ans, hits)
        if warn:
            ans, mode = (
                NOT_FOUND,
                mode + "+blocked(ungrounded numbers: %s)" % ",".join(warn),
            )
            used = []
        return dict(
            info,
            prompt=prompt,
            answer=ans,
            sources=[u["id"] for u in used],
            mode=mode,
            warn=warn,
        )


def run(**kwargs):
    in_dir, out_dir = default_dirs()
    in_dir, out_dir = kwargs.get("input_dir", in_dir), kwargs.get("output_dir", out_dir)
    out = Path(out_dir) / "rag"
    out.mkdir(parents=True, exist_ok=True)
    qs = [
        l.strip()
        for l in (Path(in_dir) / "sentences.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if l.strip()
    ]
    rag = RagAssistant(kwargs.get("kb_dir"))
    R, A = [], []
    for i, q in enumerate(qs, 1):
        r = rag.ask(q)
        hits = (
            "\n".join(
                f"    #{n} {h['id']}  score={h['score']}\n       {h['text'][:160]}{'...' if len(h['text']) > 160 else ''}"
                for n, h in enumerate(r["hits"], 1)
            )
            or "    (khong co)"
        )
        R.append(f"[{i:03d}] Q: {q}\n  gate: {r['gate']}\n  retrieved:\n{hits}\n")
        A.append(
            f"[{i:03d}] Q: {q}\n  A: {r['answer']}\n  mode: {r['mode']}\n  sources: {', '.join(r['sources']) or '-'}\n"
        )
    (out / "retrieval.txt").write_text("\n".join(R), encoding="utf-8")
    (out / "answer.txt").write_text("\n".join(A), encoding="utf-8")
    (Path(out_dir) / "retrieval.txt").write_text("\n".join(R), encoding="utf-8")
    print(
        f"[II-RAG] {len(qs)} cau -> {out}/retrieval.txt, answer.txt ({'LLM' if os.environ.get('ANTHROPIC_API_KEY') else 'extractive, khong co API key'})"
    )
