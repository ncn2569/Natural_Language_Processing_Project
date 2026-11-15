"""Kiểm tra Phần III: xử lý ngữ cảnh hội thoại cục bộ (History List, Centering,
Definite Descriptions, Ordinal Reference, Ellipsis và Guardrails).
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.hcmut.iaslab.nlp.app.dialogue import DialogueManager, load_dialogues, run
from src.hcmut.iaslab.nlp.app.qa import NEED_CONTEXT, NOT_FOUND

PROJECT_DIR = Path(__file__).resolve().parents[1]


class DialogueContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dm = DialogueManager(
            kb_dir=PROJECT_DIR / "data",
            grammar_path=PROJECT_DIR / "src/grammar/grammar.txt",
        )

    def test_dialogue_01_answer_entity_and_definite_chapter(self) -> None:
        turns = self.dm.run_dialogue(
            ["Tuần 3 học gì?", "Tài liệu của chương này là gì?"]
        )
        self.assertEqual(turns[0].frame.args.get("week"), "WEEK_03")
        self.assertEqual(turns[1].resolutions[0].resolved_value, "CH03")
        self.assertEqual(turns[1].parse_status, "OK")
        self.assertEqual(turns[1].frame.intent, "GET_RESOURCE")
        self.assertEqual(turns[1].frame.args.get("chapter"), "CH03")
        self.assertTrue(turns[1].result.found)
        self.assertIn("TEXTBOOK 1", turns[1].result.answer)

    def test_dialogue_02_pronoun_anaphora_with_syntax_tree(self) -> None:
        turns = self.dm.run_dialogue(
            ["Chương 4 nói về gì?", "Nó có liên quan đến parsing không?"]
        )
        t2 = turns[1]
        self.assertEqual(t2.resolutions[0].resolved_value, "CH04")
        self.assertIn("syntax_tree+rules", t2.resolutions[0].source_note)
        self.assertEqual(t2.parse_status, "OK")
        self.assertEqual(t2.frame.predicate, "CHECK_CHAPTER_COVERS")
        self.assertEqual(t2.frame.args, {"chapter": "CH04", "topic": "PARSING"})
        self.assertTrue(t2.result.answer.startswith("Có."))

    def test_dialogue_03_definite_part_resolution_and_query(self) -> None:
        turns = self.dm.run_dialogue(
            ["Bài tập lớn có phần parser không?", "Phần này cần xuất file gì?"]
        )
        t2 = turns[1]
        self.assertEqual(t2.resolutions[0].resolved_value, "PART_I")
        self.assertEqual(t2.parse_status, "OK")
        self.assertEqual(t2.frame.intent, "GET_ASSIGNMENT")
        self.assertEqual(t2.frame.predicate, "GET_ASSIGNMENT_PART")
        self.assertEqual(t2.frame.args.get("part"), "PART_I")
        self.assertIn("assignments WHERE part='PART_I'", t2.query.text)

    def test_dialogue_04_ordinal_and_ellipsis(self) -> None:
        turns = self.dm.run_dialogue(
            ["Chương 11 có hỏi đáp không?", "Còn chương sau?"]
        )
        t2 = turns[1]
        resolved_vals = {r.resolved_value for r in t2.resolutions}
        self.assertIn("CH12", resolved_vals)
        self.assertEqual(t2.parse_status, "OK")
        self.assertTrue(t2.result.found)
        self.assertIn("Chương 12", t2.result.answer)
        self.assertIn("Discourse Context and Reference", t2.result.answer)

    def test_dialogue_05_lo_definite_description_to_chapter(self) -> None:
        turns = self.dm.run_dialogue(
            ["L.O.2.5 học về gì?", "Nội dung này nằm ở chương nào?"]
        )
        t2 = turns[1]
        self.assertEqual(t2.resolutions[0].resolved_value, "LO2.5")
        self.assertEqual(t2.parse_status, "OK")
        self.assertEqual(t2.frame.predicate, "GET_LO_CHAPTER")
        self.assertTrue(t2.result.found)
        self.assertIn("Chương 6", t2.result.answer)

    def test_assignment_example_week_5_part_fallback(self) -> None:
        turns = self.dm.run_dialogue(
            ["Tuần 5 học những gì?", "Tài liệu của phần này ở đâu?"]
        )
        t2 = turns[1]
        self.assertEqual(t2.resolutions[0].resolved_value, "CH04")
        self.assertEqual(t2.parse_status, "OK")
        self.assertEqual(t2.frame.intent, "GET_RESOURCE")
        self.assertTrue(t2.result.found)

    def test_centering_and_recency_over_multiple_turns(self) -> None:
        turns = self.dm.run_dialogue(
            [
                "Chương 4 nói về gì?",
                "Chương 11 nói về gì?",
                "Nó có liên quan đến hỏi đáp không?",
            ]
        )
        self.assertEqual(turns[2].resolutions[0].resolved_value, "CH11")
        self.assertTrue(turns[2].result.answer.startswith("Có."))

        three_turn = self.dm.run_dialogue(
            [
                "Tuần 3 học gì?",
                "Tài liệu của chương này là gì?",
                "Nó có liên quan đến PCFG không?",
            ]
        )
        self.assertEqual(three_turn[2].resolutions[0].resolved_value, "CH03")
        self.assertTrue(three_turn[2].result.answer.startswith("Có."))

    def test_ellipsis_other_entity_types(self) -> None:
        week_turns = self.dm.run_dialogue(["Tuần 3 học gì?", "Còn tuần sau?"])
        self.assertEqual(week_turns[1].frame.args.get("week"), "WEEK_04")
        self.assertIn("Tuần 4", week_turns[1].result.answer)

        exam_turns = self.dm.run_dialogue(["Midterm bao nhiêu phút?", "Còn final?"])
        self.assertEqual(exam_turns[1].frame.args.get("exam"), "FINAL")
        self.assertIn("90 phút", exam_turns[1].result.answer)

    def test_guardrails_missing_context_and_out_of_bounds_ordinal(self) -> None:
        for sentence in (
            "Nó có liên quan đến parsing không?",
            "Tài liệu của chương này là gì?",
            "Còn chương sau?",
        ):
            with self.subTest(sentence=sentence):
                turns = self.dm.run_dialogue([sentence])
                self.assertEqual(turns[0].result.answer, NEED_CONTEXT)
                self.assertFalse(turns[0].result.found)

        oos_then_ref = self.dm.run_dialogue(
            ["Ai là tổng thống của Pháp?", "Nó có liên quan đến parsing không?"]
        )
        self.assertEqual(oos_then_ref[1].result.answer, NEED_CONTEXT)

        oob_ordinal = self.dm.run_dialogue(["Chương 12 nói về gì?", "Còn chương sau?"])
        self.assertEqual(oob_ordinal[1].frame.args.get("chapter"), "CH13")
        self.assertEqual(oob_ordinal[1].result.answer, NOT_FOUND)
        self.assertFalse(oob_ordinal[1].result.found)

    def test_load_dialogues_and_run_writes_required_sections(self) -> None:
        dialogues = load_dialogues(PROJECT_DIR / "data/dialogues.txt")
        self.assertEqual(len(dialogues), 5)

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            with redirect_stdout(io.StringIO()):
                run(output_dir=str(out_dir), kb_dir=str(PROJECT_DIR / "data"))
            out_file = out_dir / "dialogue.txt"
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            for required_label in (
                "DIALOGUE 01",
                "DIALOGUE 05",
                "Hội thoại đầu vào",
                "Ngữ cảnh sử dụng",
                "Kết quả diễn giải",
                "Câu trả lời",
            ):
                self.assertIn(required_label, content)


if __name__ == "__main__":
    unittest.main()
