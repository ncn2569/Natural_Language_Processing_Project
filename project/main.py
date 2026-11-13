"""entry của toàn bộ assignment"""

from __future__ import annotations

from pathlib import Path

from src.grammar import CourseParser, SentenceGenerator, export_grammar, load_cfg
from src.hcmut.iaslab.nlp.app import dialogue, evaluate, part2, rag


def run_part1(base_dir: Path) -> None:
    """Nạp/xuất grammar (3.1), sinh câu (3.2), phân tích input (3.3).

    base_dir là thư mục project/, không phụ thuộc vị trí gọi lệnh Python.
    """
    grammar_src = base_dir / "src" / "grammar" / "grammar.txt"
    input_sentences = base_dir / "input" / "sentences.txt"
    output_dir = base_dir / "output"

    grammar_out = output_dir / "grammar.txt"
    samples_out = output_dir / "samples.txt"
    parse_results_out = output_dir / "parse-results.txt"

    # Kiểm tra input trước khi ghi bất kỳ output nào để lỗi thiếu file rõ ràng.
    for required_path in (grammar_src, input_sentences):
        if not required_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {required_path}")

    # 3.1: Cả generator và parser dùng cùng một grammar nguồn.
    grammar = load_cfg(grammar_src)
    export_grammar(grammar_src, grammar_out)

    # 3.2: Mặc định 5.000 câu; seed và các giới hạn nằm ở SentenceGenerator.
    generator = SentenceGenerator(grammar)
    samples = generator.generate_to_file(samples_out)

    # 3.3: Mỗi dòng input cho một cây hoặc (), giữ đúng thứ tự đầu vào.
    parser = CourseParser(grammar)
    parse_results = parser.parse_file(input_sentences, parse_results_out)
    tree_count = sum(result != "()" for result in parse_results)
    print(f"Đã xuất grammar: {grammar_out}")
    print(
        f"Đã sinh {len(samples)} câu sau {generator.last_attempts} lần thử: {samples_out}"
    )
    print(
        f"Đã phân tích {len(parse_results)} dòng: {tree_count} có cây, "
        f"{len(parse_results) - tree_count} không có cây: {parse_results_out}"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    # 1. Chạy Phần I: Grammar -> Samples -> Parser (parse-results.txt)
    run_part1(base_dir)
    # 2. Chạy Phần II (Classical NLP): Semantic -> Intent/Entity -> Query -> Answer
    part2.run(
        input_dir=str(base_dir / "input"),
        output_dir=str(base_dir / "output"),
        kb_dir=str(base_dir / "data"),
    )
    # 3. Chạy Phần II (RAG): Retrieval -> Answer
    rag.run(
        input_dir=str(base_dir / "input"),
        output_dir=str(base_dir / "output"),
        kb_dir=str(base_dir / "data"),
    )
    # 4. Chạy Phần III: Xử lý ngữ cảnh hội thoại cục bộ (dialogue.txt)
    dialogue.run(
        output_dir=str(base_dir / "output"),
        kb_dir=str(base_dir / "data"),
        dialogues_path=str(base_dir / "data" / "dialogues.txt"),
    )
    # 5. Chạy Phần IV: Đánh giá hệ thống
    evaluate.run(
        output_dir=str(base_dir / "output"),
        kb_dir=str(base_dir / "data"),
        queries=str(base_dir / "data" / "sample_queries.txt"),
    )
    evaluate.run_rag(
        output_dir=str(base_dir / "output"),
        kb_dir=str(base_dir / "data"),
        queries=str(base_dir / "data" / "sample_queries.txt"),
    )


if __name__ == "__main__":
    main()
