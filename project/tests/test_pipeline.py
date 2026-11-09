"""Kiểm tra đủ ba bước bằng thư mục tạm, không ghi đè output của nhóm."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
import unittest

from main import run_part1
from src.grammar import CourseParser, load_cfg

PROJECT_DIR = Path(__file__).resolve().parents[1]


class PartOnePipelineTests(unittest.TestCase):
    def test_full_pipeline_outputs_and_all_5000_generated_sentences(self):
        with tempfile.TemporaryDirectory() as directory:
            base_dir = Path(directory)
            (base_dir / "src/grammar").mkdir(parents=True)
            (base_dir / "input").mkdir()
            shutil.copyfile(
                PROJECT_DIR / "src/grammar/grammar.txt", base_dir / "src/grammar/grammar.txt"
            )
            shutil.copyfile(
                PROJECT_DIR / "input/sentences.txt", base_dir / "input/sentences.txt"
            )
            with redirect_stdout(io.StringIO()):
                run_part1(base_dir)

            output_dir = base_dir / "output"
            self.assertEqual(
                (output_dir / "grammar.txt").read_bytes(),
                (base_dir / "src/grammar/grammar.txt").read_bytes(),
            )
            samples = (output_dir / "samples.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(samples), 5_000)
            self.assertEqual(len(set(samples)), 5_000)
            self.assertTrue(all(sentence.strip() for sentence in samples))

            parser = CourseParser(load_cfg(base_dir / "src/grammar/grammar.txt"))
            for line_number, sentence in enumerate(samples, start=1):
                with self.subTest(line=line_number, sentence=sentence):
                    self.assertIsNotNone(parser.parse(sentence))

            parse_results = (output_dir / "parse-results.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(parse_results), 51)
            self.assertTrue(all(result != "()" for result in parse_results[:48]))
            self.assertEqual(parse_results[48:], ["()", "()", "()"])


if __name__ == "__main__":
    unittest.main()
