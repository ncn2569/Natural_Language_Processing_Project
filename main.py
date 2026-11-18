
"""Entry point của toàn bộ CourseAssistant assignment."""

from __future__ import annotations

from pathlib import Path

from src.parser.grammar_loader import load_cfg
from src.parser.course_parser import CourseParser
from src.parser.sentence_generator import SentenceGenerator
from src.pipeline import CourseAssistant
import evaluate


def run_part1(base_dir: Path) -> None:
    """Phần I: Grammar -> Sentence Generation -> Parsing."""

    grammar_src = base_dir / "src" / "parser" / "course_grammar.cfg"
    input_sentences = base_dir / "course" / "sentences.txt"
    output_dir = base_dir / "output"

    grammar_out = output_dir / "grammar.txt"
    samples_out = output_dir / "samples.txt"
    parse_results_out = output_dir / "parse-results.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    for required_path in (grammar_src, input_sentences):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file đầu vào: {required_path}"
            )

    # 3.1: Nạp và xuất grammar
    grammar = load_cfg(grammar_src)
    grammar_out.write_text(
        grammar_src.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    # 3.2: Sinh câu
    generator = SentenceGenerator(grammar)
    samples = generator.generate_to_file(samples_out)

    # 3.3: Phân tích câu
    parser = CourseParser(grammar)
    parse_results = parser.parse_file(
        input_sentences,
        parse_results_out
    )

    tree_count = sum(result != "()" for result in parse_results)

    print("\n========== PHẦN I: CFG ==========")
    print(f"Đã xuất grammar: {grammar_out}")
    print(
        f"Đã sinh {len(samples)} câu: {samples_out}"
    )
    print(
        f"Đã phân tích {len(parse_results)} câu: "
        f"{tree_count} câu có cây, "
        f"{len(parse_results) - tree_count} câu không có cây."
    )


def run_part2(base_dir: Path) -> None:
    """Phần II: Semantic Analysis -> TF-IDF Retrieval -> Gemini."""

    print("\n========== PHẦN II: NLP + RAG ==========")

    assistant = CourseAssistant()

    # Xây dựng chỉ mục TF-IDF
    count = assistant.ingest()
    print(f"Đã nạp knowledge base: {count} chunks.")

    # Chạy các câu hỏi mẫu
    query_file = base_dir / "course" / "sample_queries.txt"

    if not query_file.is_file():
        print("Không tìm thấy sample_queries.txt, bỏ qua chạy câu hỏi mẫu.")
        return

    questions = []

    for line in query_file.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        parts = line.split("|")

        # Định dạng: Q01 | câu hỏi | intent | entity
        if len(parts) >= 2 and parts[0].strip().startswith("Q"):
            questions.append(parts[1].strip())

    for question in questions:
        print(f"\nCâu hỏi: {question}")

        try:
            result = assistant.ask(question)
            print(f"Intent: {result['semantic']['intent']}")
            print(f"Answer: {result['answer']}")
            print(f"Status: {result['generation_status']}")

        except Exception as exc:
            print(f"Lỗi khi xử lý câu hỏi: {exc}")

    print("\nĐã lưu kết quả tại output/retrieval.txt và output/answer.txt")


def run_part4() -> None:
    """Phần IV: Đánh giá Intent và Retrieval."""

    print("\n========== PHẦN IV: EVALUATION ==========")
    evaluate.main()


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Phần I
    run_part1(base_dir)

    # Phần II
    run_part2(base_dir)

    # Phần III: Chưa triển khai module Dialogue độc lập.
    print("\n========== PHẦN III: DIALOGUE ==========")
    print(
        "Chưa triển khai bộ quản lý ngữ cảnh hội thoại nhiều lượt. "
        "Hiện có dữ liệu dialogues.txt trong knowledge base."
    )

    # Phần IV
    run_part4()

    print("\n========== HOÀN TẤT ASSIGNMENT ==========")


if __name__ == "__main__":
    main()
