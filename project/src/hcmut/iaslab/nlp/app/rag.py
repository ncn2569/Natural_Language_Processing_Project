"""Phan II - Lua chon 2: Retrieval-Augmented Generation (RAG).

User Question -> Retrieve (BM25 tu cai dat, mo rong truy van bang entity/tu dien cua Phan I)
-> Retrieved Context -> Prompt -> LLM (neu co API key) / trich xuat cau co can cu (neu khong) -> Answer + nguon.

Chong bia (hallucination):
  - cong ngoai pham vi (dung bo phan loai UNKNOWN cua Phan II-Classical) + nguong diem retrieval;
  - prompt bat buoc chi dung ngu canh va phai noi "khong tim thay" neu thieu;
  - kiem tra hau kiem: so lieu/ma trong cau tra loi phai xuat hien trong ngu canh.

LLM: GEMINI_API_KEY/GOOGLE_API_KEY trong environment hoac .env o repo root.
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock

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
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_REQUEST_INTERVAL = 10.0
MAX_RATE_LIMIT_RETRIES = 2
_request_lock = Lock()
_next_request_at = 0.0
ENV_FILE = PROJECT_ROOT.parent / ".env"


def get_api_key() -> str | None:
    """Load API settings without overriding process variables."""
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            match = re.fullmatch(
                r"\s*(?:export\s+)?"
                r"(GEMINI_API_KEY|GOOGLE_API_KEY|RAG_MODEL"
                r"|RAG_REQUEST_INTERVAL_SECONDS)\s*=\s*(.*?)\s*",
                line,
            )
            if match:
                name, value = match.groups()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(name, value)
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


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


def _request_interval() -> float:
    value = os.environ.get(
        "RAG_REQUEST_INTERVAL_SECONDS", str(DEFAULT_REQUEST_INTERVAL)
    )
    try:
        interval = float(value)
    except ValueError:
        return DEFAULT_REQUEST_INTERVAL
    if math.isfinite(interval) and interval >= 0:
        return interval
    return DEFAULT_REQUEST_INTERVAL


def _retry_after(value: str) -> float:
    if value.isascii() and value.isdigit():
        return float(value)
    if value:
        try:
            retry_at = parsedate_to_datetime(value).timestamp()
        except (ValueError, TypeError, OverflowError, OSError):
            return 0.0
        return max(0.0, retry_at - time.time())
    return 0.0


def _request_llm(request: urllib.request.Request) -> dict:
    """Serialize API calls and pace generation and judging together."""
    global _next_request_at

    interval = _request_interval()
    with _request_lock:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            wait = _next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            cooldown = interval
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code != 429:
                    raise
                headers = error.headers or {}
                retry_after = _retry_after(headers.get("Retry-After", ""))
                cooldown = max(interval, 10.0 * 2**attempt, retry_after)
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"[RAG] HTTP 429. Thu lai sau {cooldown:g} giay.")
                error.close()
            finally:
                _next_request_at = time.monotonic() + cooldown


def call_llm(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = SYSTEM,
) -> str | None:
    key = get_api_key()
    if not key:
        return None
    model = model or os.environ.get("RAG_MODEL") or DEFAULT_MODEL
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError(
            "RAG_MODEL phải là tên model Gemini, không phải URL hoặc đường dẫn."
        )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1024},
    }
    if system is not None:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        data=body,
        headers={
            "x-goog-api-key": key,
            "content-type": "application/json",
        },
    )
    try:
        response = _request_llm(req)
        candidates = response.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            print("[RAG] Gemini khong co dap an hoan chinh, dung extractive.")
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not answer:
            print("[RAG] Gemini tra noi dung rong, dung extractive.")
        return answer or None
    except urllib.error.HTTPError as error:
        print(f"[RAG] Gemini HTTP {error.code}, dung extractive.")
        return None
    except (OSError, ValueError, TypeError, AttributeError):
        # Không in exception hoặc body để tránh lộ key/nội dung request.
        print("[RAG] Gemini loi ket noi/phan hoi, dung extractive.")
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

    def ask(self, question, *, use_llm=True):
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
        ans = call_llm(prompt) if use_llm else None
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
        f"[II-RAG] {len(qs)} cau -> {out}/retrieval.txt, answer.txt ({'Gemini requested; xem mode tung cau' if get_api_key() else 'extractive, khong co API key'})"
    )
