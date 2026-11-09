"""Main entry point for the CO3085 Course Assistant NLP pipeline."""

from __future__ import annotations

from pathlib import Path

from src.grammar import CourseParser, SentenceGenerator, export_grammar, load_cfg


def run_part1(base_dir: Path) -> None:
    """Execute Part I: Grammar export (3.1), Sentence generation (3.2), and Parsing (3.3)."""
    grammar_src = base_dir / "src" / "grammar" / "grammar.txt"
    input_sentences = base_dir / "input" / "sentences.txt"
    output_dir = base_dir / "output"

    grammar_out = output_dir / "grammar.txt"
    samples_out = output_dir / "samples.txt"
    parse_results_out = output_dir / "parse-results.txt"

    # 3.1 Load nltk.CFG and export to output/grammar.txt
    grammar = load_cfg(grammar_src)
    export_grammar(grammar_src, grammar_out)

    # 3.2 Generate up to 10,000 valid domain sentences into output/samples.txt
    generator = SentenceGenerator(grammar)
    generator.generate_to_file(samples_out, max_samples=5000)

    # 3.3 Parse input/sentences.txt using nltk.ChartParser into output/parse-results.txt
    parser = CourseParser(grammar)
    parser.parse_file(input_sentences, parse_results_out)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    run_part1(base_dir)


if __name__ == "__main__":
    main()
