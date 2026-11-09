"""Phần I, mục 3.3: tách terminal tiếng Việt và tìm cây bằng NLTK ChartParser."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import nltk

LEADING_PUNCT_RE = re.compile(r'^[\s"“”\'‘’()\[\]]+')
TRAILING_PUNCT_RE = re.compile(r'[\s?!.,;:"“”\'‘’()\[\]]+$')

# Các luật linh hoạt giúp nhận thêm câu nhưng cũng tạo nhiều cây nhập nhằng.
# Khi có lựa chọn, ưu tiên cây dùng ít luật này hơn (không phải cấm các luật).
FALLBACK_RULES = {
    ("Modifiers", ("Modifier", "Modifiers")),  # Nhiều bổ ngữ cùng một danh ngữ.
    ("NounHead", ("Noun", "NounHead")),       # Ghép các danh từ liên tiếp.
    ("Clause", ("Predicate",)),               # Mệnh đề thiếu chủ ngữ.
    ("WhClause", ("WhPredicate",)),           # Câu hỏi thiếu chủ ngữ.
    ("VerbPhrase", ("VerbCore", "PrepositionalPhrase")),  # Gắn thêm cụm giới từ.
}


class CourseParser:
    """Chuẩn hóa → thử các cách tách → phân tích → chọn một cây đại diện.

    Grammar nguồn dùng terminal chữ thường. Một terminal có thể là cả cụm
    nhiều âm tiết, ví dụ "bài tập lớn", chứ không nhất thiết là một từ đơn.
    """

    def __init__(self, grammar: nltk.CFG) -> None:
        self.grammar = grammar
        self.chart_parser = nltk.ChartParser(grammar)
        self.lexicon = {
            symbol
            for production in grammar.productions()
            for symbol in production.rhs()
            if isinstance(symbol, str)
        }
        self.max_term_words = max(
            (len(terminal.split()) for terminal in self.lexicon), default=1
        )

    @staticmethod
    def preprocess(sentence: str) -> list[str]:
        """Đưa câu về chữ thường, bỏ dấu câu ở mép, rồi tách theo khoảng trắng.

        Đây mới là bước chuẩn hóa; các âm tiết còn được ghép thành terminal.
        Dấu '.' và '/' bên trong L.O.2.5, LLM/RAG phải được giữ nguyên.
        """
        cleaned = LEADING_PUNCT_RE.sub("", sentence.strip())
        cleaned = TRAILING_PUNCT_RE.sub("", cleaned).lower()
        words = []
        for raw_word in cleaned.split():
            word = LEADING_PUNCT_RE.sub("", raw_word)
            word = TRAILING_PUNCT_RE.sub("", word)
            if word:
                words.append(word)
        return words

    def iter_tokenizations(self, sentence: str) -> Iterator[list[str]]:
        """Sinh lần lượt mọi cách ghép âm tiết thành terminal có trong grammar.

        Ưu tiên cụm dài là thứ tự thử, không phải kết luận cách tách đó đúng.
        Không cắt ở bốn ứng viên: cách tách tạo được cây có thể xuất hiện sau.
        """
        words = self.preprocess(sentence)
        if not words:
            return

        def backtrack(word_index: int, current_tokens: list[str]) -> Iterator[list[str]]:
            if word_index == len(words):
                # Sao chép vì danh sách hiện tại sẽ tiếp tục thay đổi khi quay lui.
                yield current_tokens.copy()
                return

            max_phrase_length = min(self.max_term_words, len(words) - word_index)
            for phrase_length in range(max_phrase_length, 0, -1):
                phrase = " ".join(words[word_index : word_index + phrase_length])
                if phrase in self.lexicon:
                    # Thử một cụm, đi tiếp, rồi bỏ cụm đó để thử lựa chọn khác.
                    current_tokens.append(phrase)
                    yield from backtrack(word_index + phrase_length, current_tokens)
                    current_tokens.pop()

        yield from backtrack(0, [])

    @staticmethod
    def _tree_score(tree: nltk.Tree) -> tuple[int, int]:
        """Ưu tiên ít luật dự phòng, sau đó ít production; điểm nhỏ hơn tốt hơn.

        Đây là heuristic chọn cây đại diện, không phải xác suất đúng ngữ nghĩa.
        """
        productions = tree.productions()
        fallback_count = 0
        for production in productions:
            right_hand_side = tuple(
                symbol.symbol() if isinstance(symbol, nltk.Nonterminal) else symbol
                for symbol in production.rhs()
            )
            if (production.lhs().symbol(), right_hand_side) in FALLBACK_RULES:
                fallback_count += 1
        return fallback_count, len(productions)

    def parse(self, sentence: str) -> nltk.Tree | None:
        """Chọn cây có điểm nhỏ nhất giữa mọi cách tách; không có cây thì None.

        Nếu NLTK gặp lỗi hoặc vượt giới hạn nội bộ, lỗi được truyền lên để
        phân biệt với câu không thuộc grammar. Câu dài/nhiều cây có thể chậm.
        """
        best_tree = None
        best_score = None
        for tokens in self.iter_tokenizations(sentence):
            # Duyệt lần lượt, không giữ thêm một list chứa toàn bộ cây.
            for tree in self.chart_parser.parse(tokens):
                score = self._tree_score(tree)
                if best_score is None or score < best_score:
                    best_tree, best_score = tree, score
        # Khi hòa điểm, cây gặp trước được giữ lại để thứ tự chọn ổn định.
        return best_tree

    def parse_to_bracket(self, sentence: str) -> str:
        """Xuất một cây trên một dòng; '()' nghĩa là không có cây trong CFG."""
        tree = self.parse(sentence)
        return "()" if tree is None else tree.pformat(margin=1_000_000)

    def parse_file(self, input_path: str | Path, output_path: str | Path) -> list[str]:
        """Mỗi dòng input tương ứng một dòng output; dòng trống cho kết quả ().

        Đọc và phân tích xong mới ghi file. Thiếu input hoặc lỗi phân tích
        không được ghi đè output cũ bằng kết quả rỗng/dang dở.
        """
        sentences = Path(input_path).read_text(encoding="utf-8").splitlines()
        results = []
        for line_number, sentence in enumerate(sentences, start=1):
            try:
                results.append(self.parse_to_bracket(sentence))
            except (ValueError, RecursionError) as error:
                raise RuntimeError(
                    f"Lỗi phân tích dòng {line_number} trong {input_path}: {error}"
                ) from error

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(results) + ("\n" if results else ""), encoding="utf-8"
        )
        return results
