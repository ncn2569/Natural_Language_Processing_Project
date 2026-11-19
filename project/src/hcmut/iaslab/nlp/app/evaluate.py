"""Phần IV: đánh giá Classical QA và Hội thoại theo nhóm."""
import re
from pathlib import Path
from typing import Any

PathInput = str | Path

# from .generate import PROJECT_ROOT, default_dirs
from .part2 import PROJECT_ROOT, CourseAssistant, default_dirs


def load_queries(path: PathInput) -> list[tuple[str, ...]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(Q\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\S+)\s*$", line)
        if m:
            rows.append(m.groups())
    return rows


def load_evaluation_queries(path: PathInput) -> list[dict[str, Any]]:
    """Load prepared evaluation cases in their original order."""
    import json

    query_path = Path(path)
    metadata_path = query_path.with_name("evaluation_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []
    for line in query_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "ID |")):
            continue
        qid, question, intent, entity, answer = (
            value.strip() for value in line.split("|", maxsplit=4)
        )
        rows.append(dict(
            id=qid,
            question=question,
            expected_intent=intent,
            expected_entity=entity,
            expected_answer=answer,
            **metadata[qid],
        ))
    return rows


def normalize_answer(answer: str) -> str:
    """So khớp mẫu Classical, không dùng intent/entity để suy ra câu trả lời đúng.

    Chỉ bỏ khác biệt chữ hoa, khoảng trắng, Unicode và chú thích lịch mẫu
    đã bổ sung trong gold. Không xóa phủ định, số liệu hoặc tên đối tượng.
    """
    import unicodedata
    answer = unicodedata.normalize("NFC", answer)
    answer = re.sub(r"\s+Đây là dữ liệu lịch mẫu(?:, không phải lịch hành chính chính thức)?\.\s*$",
                    "", answer, flags=re.I)
    return " ".join(answer.split()).casefold()


def answer_matches(actual: str, expected: str) -> bool:
    return normalize_answer(actual) == normalize_answer(expected)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, int]:
    n = len(rows)
    correct = sum(row["answer_ok"] for row in rows)
    return dict(total=n, intent_correct=sum(row["intent_ok"] for row in rows),
                entity_correct=sum(row["entity_ok"] for row in rows),
                answer_correct=correct, answer_incorrect=n - correct)


class ClassicalEvaluator:
    """Run the Classical pipeline and export grouped evaluation results."""

    def __init__(
        self,
        *,
        output_dir: PathInput | None = None,
        queries: PathInput | None = None,
        kb_dir: PathInput | None = None,
        grammar_path: PathInput | None = None,
    ) -> None:
        _, default_out = default_dirs()
        self.output_dir = Path(output_dir or default_out)
        self.queries = Path(queries or PROJECT_ROOT / "data/evaluation_queries.txt")
        self.kb_dir = kb_dir or PROJECT_ROOT / "data"
        self.grammar_path = Path(grammar_path or PROJECT_ROOT / "src/grammar/grammar.txt")


    def run(self) -> dict[str, Any]:
        """Phần IV Classical: Parser + QA; F giữ ngữ cảnh bằng DialogueManager.

        Tất cả số liệu và chi tiết báo cáo được tính từ kết quả chạy thực tế.
        Không sử dụng kết quả duyệt thủ công hoặc danh sách ID được cho điểm sẵn.
        """
        import json
        from src.grammar import CourseParser, load_cfg
        from .dialogue import DialogueManager

        out_dir = self.output_dir
        qpath = self.queries
        kb_dir = self.kb_dir
        grammar_path = self.grammar_path
        cases = load_evaluation_queries(qpath)
        if not cases:
            return {}
        parser = CourseParser(load_cfg(grammar_path))
        assistant = CourseAssistant(kb_dir)
        manager = DialogueManager(kb_dir=kb_dir, grammar_path=grammar_path)
        results, active = [], None
        for case in cases:
            qid, question = case["id"], case["question"]
            context = {}
            if case.get("dialogue_id"):
                dialogue_id = case["dialogue_id"]
                if active != dialogue_id:
                    manager.reset()
                    active = dialogue_id
                turn = manager.ask(question)
                frame, query, result = turn.frame, turn.query, turn.result
                context = dict(resolved_input=turn.resolved_input, context_used=turn.context_used)
            else:
                active = None
                tree = parser.parse_to_bracket(question)
                frame, query, result = assistant.ask(question, tree)
            results.append(dict(**case, intent=frame.intent, entity=frame.primary,
                                intent_ok=frame.intent == case["expected_intent"],
                                entity_ok=frame.primary == case["expected_entity"],
                                answer=result.answer,
                                answer_ok=answer_matches(result.answer, case["expected_answer"]),
                                semantic=frame.semantic(), query=query.text,
                                parse_status=frame.parse_status, sources=result.sources, **context))

        total = summarize_results(results)
        groups = {g: summarize_results([r for r in results if r["group"] == g])
                  for g in dict.fromkeys(r["group"] for r in results)}
        dialogues = {}
        for row in results:
            if row.get("dialogue_id"):
                name = row["dialogue_id"]
                dialogues[name] = dialogues.get(name, True) and row["answer_ok"]

        def summary_lines(stats: dict[str, int]) -> list[str]:
            n = stats["total"]
            return [f"Số truy vấn: {n}",
                    f"Intent đúng: {stats['intent_correct']}/{n} = {stats['intent_correct'] / n:.2%}",
                    f"Entity chính đúng: {stats['entity_correct']}/{n} = {stats['entity_correct'] / n:.2%}",
                    f"Số câu trả lời đúng: {stats['answer_correct']}/{n} = {stats['answer_correct'] / n:.2%}",
                    f"Số câu trả lời sai: {stats['answer_incorrect']}/{n}"]

        head = ["ĐÁNH GIÁ TỔNG QUAN - CLASSICAL QA", *summary_lines(total)]
        if dialogues:
            head.append(f"Hội thoại đúng toàn bộ: {sum(dialogues.values())}/{len(dialogues)}")

        # Tóm tắt theo nhóm, các ca trả lời sai, phân tích nguyên nhân lỗi và ưu/nhược điểm (Mục 6).
        group_overview = ["", "TỔNG HỢP ĐỘ CHÍNH XÁC THEO NHÓM"]
        for group, stats in groups.items():
            n = stats["total"]
            group_overview.append(
                f"- {group} ({n} câu): "
                f"Intent {stats['intent_correct']}/{n} ({stats['intent_correct'] / n:.2%}) | "
                f"Entity {stats['entity_correct']}/{n} ({stats['entity_correct'] / n:.2%}) | "
                f"Câu trả lời {stats['answer_correct']}/{n} ({stats['answer_correct'] / n:.2%})"
            )

        failed_rows = [r for r in results if not r["answer_ok"]]
        error_section = ["", f"MỘT SỐ TRƯỜNG HỢP HỆ THỐNG TRẢ LỜI SAI ({len(failed_rows)}/{total['total']})"]
        for row in failed_rows:
            error_section.extend([
                f"- [{row['id']}] ({row['group']}) {row['question']}",
                f"    Intent: kỳ vọng={row['expected_intent']} | thực tế={row['intent']} | "
                f"Entity: kỳ vọng={row['expected_entity']} | thực tế={row['entity']}",
                f"    Kỳ vọng: {row['expected_answer']}",
                f"    Thực tế: {row['answer']}",
            ])

        by_err_group = {
            g: [r for r in failed_rows if r["group"] == g]
            for g in groups
            if any(r["group"] == g for r in failed_rows)
        }
        analysis_section = [
            "",
            "PHÂN TÍCH NGUYÊN NHÂN LỖI",
        ]
        if "Thiếu thông tin" in by_err_group:
            ids = ", ".join(r["id"] for r in by_err_group["Thiếu thông tin"])
            analysis_section.append(
                f"1. Thiếu cơ chế kiểm tra thuộc tính ngoài schema KB ({len(by_err_group['Thiếu thông tin'])} câu: {ids}): "
                "khi thực thể hợp lệ (CO3085, MIDTERM, FINAL, BTL01) đi kèm thuộc tính không có trong KB "
                "(phòng học, học phí, ngày/giờ thi, link tải slide, chính sách nộp muộn...), hệ thống rơi vào "
                "nhánh trả lời mặc định thay vì từ chối NOT_FOUND."
            )
        if "Ngoài phạm vi" in by_err_group:
            ids = ", ".join(r["id"] for r in by_err_group["Ngoài phạm vi"])
            analysis_section.append(
                f"2. Câu ngoài miền nhưng trùng từ khóa bề mặt ({len(by_err_group['Ngoài phạm vi'])} câu: {ids}): "
                "luật từ khóa kích hoạt nhầm khi câu ngoài phạm vi chứa từ chung như 'tài liệu', 'quy định', 'môn học' "
                "nhưng tân ngữ thuộc lĩnh vực khác."
            )
        lex_rows = (
            by_err_group.get("Diễn đạt tương đương", [])
            + by_err_group.get("Hội thoại", [])
            + [
                r
                for r in by_err_group.get("Câu hỏi hợp lệ", [])
                if not (r["intent_ok"] and r["entity_ok"])
            ]
        )
        if lex_rows:
            ids = ", ".join(r["id"] for r in lex_rows)
            analysis_section.append(
                f"3. Thiếu từ đồng nghĩa / paraphrase ngoài từ điển ({len(lex_rows)} câu: {ids}): "
                "hệ thống dựa trên từ điển entities.txt và regex nên bỏ sót các diễn đạt chưa có trong từ điển "
                "(số thứ tự bằng chữ tiếng Việt, bản dịch tiếng Việt của thuật ngữ, biến thể số nhiều, chủ đề tài liệu mới)."
            )
        fmt_rows = [
            r
            for r in by_err_group.get("Câu hỏi hợp lệ", [])
            if r["intent_ok"] and r["entity_ok"]
        ]
        if fmt_rows:
            ids = ", ".join(r["id"] for r in fmt_rows)
            analysis_section.append(
                f"4. Đúng intent/entity nhưng khác trường thuộc tính hoặc khác chuỗi so khớp mẫu ({len(fmt_rows)} câu: {ids}): "
                "câu hỏi nhận diện đúng intent và thực thể nhưng ánh xạ vào nhánh tóm tắt mặc định hoặc khác chi tiết diễn đạt/meta-rule so với chuỗi gold khi chấm exact match."
            )

        pros_cons_section = [
            "",
            "NHẬN XÉT ƯU ĐIỂM VÀ HẠN CHẾ CỦA PHƯƠNG PHÁP (CLASSICAL NLP)",
            "- Ưu điểm:",
            "  + Minh bạch, giải thích và truy vết được toàn bộ pipeline (Parse Tree -> Semantic Frame -> Pseudo-SQL -> Nguồn KB).",
            "  + Chạy 100% offline, tốc độ nhanh, kết quả tái lập hoàn toàn (deterministic), không phát sinh ảo giác số liệu.",
            "  + Độ chính xác cao trên câu hỏi trong miền, diễn đạt tương đương và hội thoại cục bộ có tham chiếu/tỉnh lược.",
            "- Hạn chế:",
            "  + Phụ thuộc độ phủ của văn phạm CFG, từ điển thực thể và tập luật; khó tổng quát hóa với paraphrase hoàn toàn mới.",
            "  + Dễ rơi vào fallback mặc định khi hỏi thuộc tính ngoài schema KB của thực thể hợp lệ, hoặc khi câu ngoài miền trùng từ khóa bề mặt.",
            "  + Câu trả lời sinh theo khuôn mẫu cố định, chưa linh hoạt như mô hình sinh ngôn ngữ tự nhiên.",
        ]

        lines = [
            *head,
            *group_overview,
            *error_section,
            *analysis_section,
            *pros_cons_section,
            "",
            "CHI TIẾT THEO TỪNG NHÓM",
        ]
        for group, stats in groups.items():
            lines.extend(["", f"NHÓM {group}", *summary_lines(stats)])
            for row in results:
                if row["group"] != group:
                    continue
                def yesno(value: bool) -> str:
                    return "ĐÚNG" if value else "SAI"
                lines.extend(["", f"[{row['id']}] {row['question']}",
                              f"  Intent: {yesno(row['intent_ok'])} | kỳ vọng: {row['expected_intent']} | thực tế: {row['intent']}",
                              f"  Entity: {yesno(row['entity_ok'])} | kỳ vọng: {row['expected_entity']} | thực tế: {row['entity']}",
                              f"  Câu trả lời: {yesno(row['answer_ok'])}",
                              f"  Kỳ vọng: {row['expected_answer']}",
                              f"  Thực tế: {row['answer']}",
                              f"  Parse: {row['parse_status']}",
                              f"  Semantic: {row['semantic']}",
                              f"  Query: {row['query']}",
                              f"  Nguồn: {', '.join(row['sources']) or '-'}"])
                if "resolved_input" in row:
                    lines.extend([f"  Câu sau giải tham chiếu: {row['resolved_input']}",
                                  f"  Ngữ cảnh sử dụng: {row['context_used']}"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = dict(scoring="normalized_template_match", summary=total, groups=groups,
                      dialogues=dialogues, results=results)
        (out_dir / "evaluation-results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n".join(head))
        return report


def run(
    *,
    output_dir: PathInput | None = None,
    queries: PathInput | None = None,
    kb_dir: PathInput | None = None,
    grammar_path: PathInput | None = None,
) -> dict[str, Any]:
    return ClassicalEvaluator(
        output_dir=output_dir, queries=queries, kb_dir=kb_dir,
        grammar_path=grammar_path,
    ).run()


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Evaluate Course Assistant (Classical NLP)")
    cli.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    args = cli.parse_args()
    run(output_dir=args.output_dir)
