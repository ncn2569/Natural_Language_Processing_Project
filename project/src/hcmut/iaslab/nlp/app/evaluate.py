"""Danh gia so bo (Phan IV) tren data/course/sample_queries.txt: do chinh xac intent va entity chinh."""
import re
from pathlib import Path

from .generate import PROJECT_ROOT, default_dirs
from .part2 import CourseAssistant


def load_queries(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(Q\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\S+)\s*$", line)
        if m:
            rows.append(m.groups())
    return rows


def run(**kwargs):
    _, out_dir = default_dirs()
    out_dir = kwargs.get("output_dir", out_dir)
    qpath = kwargs.get("queries", PROJECT_ROOT / "data" / "course" / "sample_queries.txt")
    ca = CourseAssistant(kwargs.get("kb_dir"))
    rows = load_queries(qpath)
    n_int = n_ent = n_ent_total = 0
    wrong, lines = [], []
    for qid, q, gi, ge in rows:
        frame, query, res = ca.ask(q)
        ok_i = frame.intent == gi
        ctx = gi == "CONTEXT_DEPENDENT"            # entity can ngu canh -> khong cham o che do 1 luot
        ok_e = ctx or frame.primary == ge
        n_int += ok_i
        if not ctx:
            n_ent_total += 1
            n_ent += ok_e
        if not (ok_i and ok_e):
            wrong.append((qid, q, gi, frame.intent, ge, frame.primary))
        lines.append(f"{qid} | {'OK ' if ok_i and ok_e else 'ERR'} | {q}\n    gold: {gi}/{ge}  pred: {frame.intent}/{frame.primary}\n"
                     f"    {frame.semantic()}\n    A: {res.answer}")
    n = len(rows)
    out = [f"DANH GIA SO BO - Classical NLP (che do 1 luot, khong ngu canh)",
           f"So cau kiem thu: {n}",
           f"Intent dung: {n_int}/{n} = {n_int / n:.1%}",
           f"Entity chinh dung (khong tinh cau can ngu canh): {n_ent}/{n_ent_total} = {n_ent / max(1, n_ent_total):.1%}",
           f"Cau sai: {len(wrong)}", ""]
    for w in wrong:
        out.append("SAI " + " | ".join(map(str, w)))
    out += ["", "CHI TIET", ""] + lines
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "evaluation.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:6]))


# ====================================================== danh gia RAG (Phan II - Lua chon 2 + Phan IV)
def _gold_chunks(rag, ent):
    """Nhan entity ky vong -> tap chunk duoc coi la dung (suy ra tu KB, khong viet tay noi dung)."""
    if re.fullmatch(r"CH\d\d", ent):
        return {"topic:" + ent}
    if ent.startswith("WEEK_"):
        return {"schedule:" + ent}
    if ent.startswith("LO"):
        return {"lo:" + ent}
    if ent.startswith("PART_") or ent == "BTL01":
        return {"assignment:" + ent}
    if ent == "CO3085":
        return {"course_info:general", "course_info:description", "faq:Q1", "faq:Q2"}
    if ent in ("MIDTERM", "FINAL"):
        return {"course_info:assessment", "faq:Q14", "faq:Q15"}
    if ent == "LLM_RAG":
        return {"regulation:LLM_RAG", "faq:Q12"}
    if ent == "AI_USAGE":
        return {"regulation:AI_usage"}
    if ent == "STATISTICAL_NLP":
        return {"resource:REFERENCE 1", "resource:TEXTBOOK 2"}
    if ent == "PYTHON_NLP":
        return {"resource:REFERENCE 2"}
    main, rest = rag.engine.chapters_for(ent)
    return {"topic:" + c for c in main + rest} | ({"faq:Q3", "faq:Q4", "faq:Q6", "faq:Q7", "faq:Q8"} if main else set())


def run_rag(**kwargs):
    from .rag import RagAssistant, grounded
    from .qa import NOT_FOUND, OUT_OF_SCOPE
    _, out_dir = default_dirs()
    out = Path(kwargs.get("output_dir", out_dir)) / "rag"
    out.mkdir(parents=True, exist_ok=True)
    rag = RagAssistant(kwargs.get("kb_dir"))
    rows = load_queries(kwargs.get("queries", PROJECT_ROOT / "data" / "course" / "sample_queries.txt"))
    hit1 = hit3 = n_ret = ok_abst = n_abst = ctx_ok = n_ctx = cited = n_ans = 0
    miss, lines = [], []
    for qid, q, gi, ge in rows:
        r = rag.ask(q)
        ids = [h["id"] for h in r["hits"]]
        if gi == "UNKNOWN":
            n_abst += 1
            ok = r["answer"] in (OUT_OF_SCOPE, NOT_FOUND)
            ok_abst += ok
        elif gi == "CONTEXT_DEPENDENT":
            n_ctx += 1
            ok = r["mode"] == "gate:needs-context"
            ctx_ok += ok
        else:
            gold = _gold_chunks(rag, ge)
            n_ret += 1
            ok1 = bool(ids[:1]) and ids[0] in gold
            ok3 = any(i in gold for i in ids[:3])
            hit1 += ok1
            hit3 += ok3
            ok = ok3
            n_ans += 1
            cited += bool(r["sources"])
            if not ok3:
                miss.append((qid, q, ge, ids[:3]))
        lines.append(f"{qid} | {'OK ' if ok else 'ERR'} | {q}\n    top: {ids[:3]}\n    mode: {r['mode']}\n    A: {r['answer']}")

    # vi du khong co thong tin trong KB + thu cong chong bia
    extra = ["Giảng viên của môn NLP là ai?", "Chương 13 nói về gì?", "Tuần 20 học gì?", "Học phí môn NLP là bao nhiêu?"]
    nf = []
    for q in extra:
        r = rag.ask(q)
        nf.append(f"  Q: {q}\n     mode={r['mode']} | top={[h['id'] for h in r['hits'][:2]]}\n     A: {r['answer'][:200]}")
    fake = "Midterm kéo dài 45 phút."
    ctx_hits = rag.retrieve("Midterm kéo dài bao lâu?", frame=None)
    block = grounded(fake, ctx_hits)

    head = ["DANH GIA RAG (che do trich xuat khong LLM; voi LLM thi cau chu thay doi nhung retrieval giu nguyen)",
            f"Tong cau kiem thu: {len(rows)} (co dap an trong KB: {n_ret}, ngoai pham vi: {n_abst}, can ngu canh: {n_ctx})",
            f"Retrieval hit@1: {hit1}/{n_ret} = {hit1 / n_ret:.1%}",
            f"Retrieval hit@3: {hit3}/{n_ret} = {hit3 / n_ret:.1%}",
            f"Tu choi dung cau ngoai pham vi: {ok_abst}/{n_abst}",
            f"Nhan dien cau can ngu canh: {ctx_ok}/{n_ctx}",
            f"Cau tra loi co nguon trich dan: {cited}/{n_ans}", ""]
    for m in miss:
        head.append("RETRIEVAL TRUOT " + " | ".join(map(str, m)))
    head += ["", "CAU KHONG CO TRONG KB (phai thong bao khong tim thay, khong bia):"] + nf
    head += ["", "KIEM TRA CHONG HALLUCINATION (thu nghiem):",
             f"  Cau tra loi gia dinh cua LLM: \"{fake}\"",
             f"  So lieu khong co trong ngu canh: {block or 'khong'} -> " + ("BI CHAN, thay bang thong bao khong tim thay" if block else "cho qua"),
             "  Han che: chi bat duoc loi so lieu; LLM van co the bia ten/khai niem khong co so -> can doi chieu nguon.",
             "", "CHI TIET", ""] + lines
    (out / "evaluation.txt").write_text("\n".join(head) + "\n", encoding="utf-8")
    print("\n".join(head[:9] + [h for h in head if h.startswith("RETRIEVAL TRUOT")]))
