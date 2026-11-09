"""Hồi quy parser: các lỗi tách từ/chọn cây và quy ước file của Phần I."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import run_part1
from src.grammar import CourseParser, load_cfg

PROJECT_DIR = Path(__file__).resolve().parents[1]


class CourseParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = CourseParser(load_cfg(PROJECT_DIR / "src/grammar/grammar.txt"))

    def test_valid_segmentation_after_first_four_is_not_lost(self):
        sentence = "Tài liệu tham khảo môn học này và môn học đó không?"
        tree = self.parser.parse(sentence)
        self.assertIsNotNone(tree)
        self.assertIn("YesNoQuestion", [node.label() for node in tree.subtrees()])
        subject = next(node for node in tree.subtrees() if node.label() == "Subject")
        self.assertEqual(subject.leaves(), ["tài liệu"])

    def test_better_tree_can_come_from_a_later_segmentation(self):
        tree = self.parser.parse("Tài liệu tham khảo môn học.")
        subject = next(node for node in tree.subtrees() if node.label() == "Subject")
        self.assertEqual(subject.leaves(), ["tài liệu"])
        verb = next(node for node in tree.subtrees() if node.label() == "ObjectVerb")
        self.assertEqual(verb.leaves(), ["tham khảo"])

    def test_internal_punctuation_in_entity_names_is_preserved(self):
        for sentence, terminal in [
            ('“L.O.2.5 là gì?”', "l.o.2.5"),
            ("Có được dùng LLM/RAG không?", "llm/rag"),
        ]:
            with self.subTest(sentence=sentence):
                self.assertIn(terminal, self.parser.parse(sentence).leaves())

    def test_reference_modifier_stays_with_referenced_noun(self):
        tree = self.parser.parse("Tài liệu của phần này ở đâu?")
        self.assertTrue(any(
            node.label() == "Nominal" and node.leaves() == ["phần", "này"]
            for node in tree.subtrees()
        ))

    def test_known_words_in_wrong_order_are_rejected(self):
        self.assertEqual(self.parser.parse_to_bracket("NLP tín chỉ bao nhiêu có?"), "()")

    def test_unknown_words_and_empty_input_are_rejected(self):
        for sentence in ("Ai là tổng thống của Pháp?", "", "   "):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.parser.parse_to_bracket(sentence), "()")

    def test_file_preserves_line_order_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            output_path = Path(directory) / "output.txt"
            input_path.write_text("\nMôn NLP có bao nhiêu tín chỉ?\n\n", encoding="utf-8")
            results = self.parser.parse_file(input_path, output_path)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], "()")
            self.assertNotEqual(results[1], "()")
            self.assertEqual(results[2], "()")
            self.assertEqual(output_path.read_text().splitlines(), results)

    def test_missing_input_does_not_overwrite_old_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.txt"
            output_path.write_text("old output\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                self.parser.parse_file(Path(directory) / "missing.txt", output_path)
            self.assertEqual(output_path.read_text(), "old output\n")

    def test_library_failure_reports_line_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.txt"
            output_path = Path(directory) / "output.txt"
            input_path.write_text("câu thứ nhất\ncâu thứ hai\n", encoding="utf-8")
            output_path.write_text("old output\n", encoding="utf-8")
            # Mô phỏng lỗi giới hạn nội bộ thư viện, không biến nó thành ().
            with patch.object(
                self.parser, "parse_to_bracket",
                side_effect=["(tree)", ValueError("tree extraction limit")],
            ):
                with self.assertRaisesRegex(RuntimeError, "dòng 2"):
                    self.parser.parse_file(input_path, output_path)
            self.assertEqual(output_path.read_text(), "old output\n")

    def test_main_checks_input_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory)
            output_dir = base_dir / "output"
            output_dir.mkdir()
            grammar_output = output_dir / "grammar.txt"
            grammar_output.write_text("old grammar", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                run_part1(base_dir)
            self.assertEqual(grammar_output.read_text(), "old grammar")

    def test_existing_51_sentences_keep_expected_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            results = self.parser.parse_file(
                PROJECT_DIR / "input/sentences.txt", Path(directory) / "results.txt"
            )
        self.assertEqual(len(results), 51)
        self.assertTrue(all(result != "()" for result in results[:48]))
        self.assertEqual(results[48:], ["()", "()", "()"])


if __name__ == "__main__":
    unittest.main()
