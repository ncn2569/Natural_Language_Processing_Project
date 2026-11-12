"""Kiểm tra bộ sinh bằng grammar nhỏ và kiểm tra tích hợp với grammar môn học."""

from pathlib import Path
import tempfile
import unittest

import nltk

from src.grammar import CourseParser, SentenceGenerator, load_cfg

PROJECT_DIR = Path(__file__).resolve().parents[1]


class SentenceGeneratorTests(unittest.TestCase):
    def test_same_rule_can_be_used_in_subject_and_object(self):
        grammar = nltk.CFG.fromstring('''
            S -> NounPhrase VerbPhrase
            NounPhrase -> "tôi" | "nlp"
            VerbPhrase -> "học" NounPhrase
        ''')
        samples = SentenceGenerator(grammar, max_attempts=100).generate(4)
        self.assertEqual(set(samples), {
            "tôi học tôi", "tôi học nlp", "nlp học tôi", "nlp học nlp",
        })
        parser = CourseParser(grammar)
        self.assertTrue(all(parser.parse(sentence) is not None for sentence in samples))

    def test_seed_reproduces_order_even_on_same_generator_instance(self):
        grammar = load_cfg(PROJECT_DIR / "src/grammar/grammar.txt")
        generator = SentenceGenerator(grammar, seed=42)
        first = generator.generate(150)
        second = generator.generate(150)
        self.assertEqual(first, second)
        self.assertEqual(first, SentenceGenerator(grammar, seed=42).generate(150))
        self.assertEqual(len(set(first)), 150)

    def test_recursive_grammar_stops_after_attempt_budget(self):
        generator = SentenceGenerator(
            nltk.CFG.fromstring("S -> S"), max_depth=4, max_attempts=10
        )
        with self.assertWarnsRegex(UserWarning, "0/1"):
            self.assertEqual(generator.generate(1), [])
        self.assertEqual(generator.last_attempts, 10)

    def test_reports_actual_count_when_grammar_has_fewer_sentences(self):
        generator = SentenceGenerator(
            nltk.CFG.fromstring('S -> "nlp"'), max_attempts=10
        )
        with self.assertWarnsRegex(UserWarning, "1/2"):
            self.assertEqual(generator.generate(2), ["nlp"])

    def test_limit_counts_terminal_phrases_not_space_separated_words(self):
        generator = SentenceGenerator(
            nltk.CFG.fromstring('S -> "môn học"'), max_terminal_phrases=1
        )
        self.assertEqual(generator.generate(1), ["môn học"])

    def test_depth_and_phrase_limits_reject_incomplete_derivations(self):
        grammar = nltk.CFG.fromstring('S -> A A\nA -> "nlp"')
        for settings in ({"max_depth": 1}, {"max_terminal_phrases": 1}):
            with self.subTest(settings=settings):
                generator = SentenceGenerator(grammar, max_attempts=5, **settings)
                with self.assertWarns(UserWarning):
                    self.assertEqual(generator.generate(1), [])

    def test_zero_samples_writes_empty_file_and_limit_is_enforced(self):
        generator = SentenceGenerator(nltk.CFG.fromstring('S -> "nlp"'))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "samples.txt"
            self.assertEqual(generator.generate_to_file(output_path, 0), [])
            self.assertEqual(output_path.read_bytes(), b"")
        for invalid_count in (-1, 10_001):
            with self.subTest(count=invalid_count):
                with self.assertRaises(ValueError):
                    generator.generate(invalid_count)

    def test_generated_course_samples_parse_with_the_same_grammar(self):
        grammar = load_cfg(PROJECT_DIR / "src/grammar/grammar.txt")
        samples = SentenceGenerator(grammar).generate(200)
        parser = CourseParser(grammar)
        self.assertEqual(len(samples), 200)
        for sentence in samples:
            with self.subTest(sentence=sentence):
                tree = parser.parse(sentence)
                self.assertIsNotNone(tree)
                self.assertLessEqual(len(tree.leaves()), 12)


if __name__ == "__main__":
    unittest.main()
