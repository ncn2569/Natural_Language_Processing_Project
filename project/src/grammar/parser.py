"""Syntactic parser built on `nltk.ChartParser` (Section 3.3)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import nltk

LEADING_PUNCT_RE = re.compile(r'^[\s"“”\'‘’()\[\]]+')
TRAILING_PUNCT_RE = re.compile(r'[\s?!.,;:"“”\'‘’()\[\]]+$')

#khử nhập nhằng 
# Fallback rules penalized when selecting the canonical tree among ambiguous parses
FALLBACK_RULES = {
    ("Modifiers", ("Modifier", "Modifiers")),
    ("NounHead", ("Noun", "NounHead")),
    ("Clause", ("Predicate",)),
    ("WhClause", ("WhPredicate",)),
    ("VerbPhrase", ("VerbCore", "PrepositionalPhrase")),
}


class CourseParser:
    """
    Wrapper around `nltk.ChartParser` with multi-word terminal tokenization
    for Vietnamese compound terms in the grammar.
    """

    def __init__(self, grammar: nltk.CFG) -> None:
        self.grammar = grammar
        # bộ ptich parser của nltk
        self.chart_parser = nltk.ChartParser(grammar)
        # Tập hợp tất cả các từ lá
        self.lexicon: Set[str] = {
            str(prod.rhs()[0]).lower()
            for prod in grammar.productions()
            if prod.is_lexical()
        }
        # giới hạn cửa sổ tìm từ ghép ở bước tách từ tiếp theo.
        self.max_term_words: int = max(
            (len(term.split()) for term in self.lexicon), default=1
        )

    @staticmethod
    def preprocess(sentence: str) -> List[str]:
        """ Cắt bỏ dấu câu ở đầu và cuối chuỗi (ví dụ: ?, ., !, ", (, ))"""
        cleaned = LEADING_PUNCT_RE.sub("", sentence.strip())
        cleaned = TRAILING_PUNCT_RE.sub("", cleaned).lower()
        if not cleaned:
            return []

        words: List[str] = []
        for raw_tok in cleaned.split():
            # Cắt dấu câu dính ở mép từ, nhưng GIỮ NGUYÊN dấu '.' và '/' bên trong từ
            tok = TRAILING_PUNCT_RE.sub("", LEADING_PUNCT_RE.sub("", raw_tok))
            if tok:
                words.append(tok)
        return words

    def tokenize_candidates(
        self, sentence: str, max_candidates: int = 4
    ) -> List[List[str]]:
        """
        Segment the input sentence into valid grammar terminals (longest-match first,
        with backtracking fallback if two compound terms overlap).
        """
        words = self.preprocess(sentence)
        if not words:
            return []

        n = len(words)
        candidates: List[List[str]] = []
        def backtrack(idx: int, current: List[str]) -> None:
            if len(candidates) >= max_candidates:
                return
            if idx == n:
                candidates.append(list(current))
                return
            # Thử ghép từ cụm dài nhất (max_k) lùi dần về 1 từ đơn
            max_k = min(self.max_term_words, n - idx)
            for k in range(max_k, 0, -1):
                phrase = " ".join(words[idx : idx + k])
                if phrase in self.lexicon:
                    current.append(phrase)
                    backtrack(idx + k, current)
                    current.pop()

        backtrack(0, [])
        return candidates

    @staticmethod
    def _tree_score(tree: nltk.Tree) -> Tuple[int, int]:
        """Đếm số luật dự phòng (FALLBACK_RULES) xuất hiện trong cây. Cây nào dùng ít luật dự phòng hơn sẽ được ưu tiên."""
        fallback_count = 0
        prods = tree.productions()
        for prod in prods:
            lhs = prod.lhs().symbol()
            rhs = tuple(
                s.symbol() if isinstance(s, nltk.Nonterminal) else str(s)
                for s in prod.rhs()
            )
            if (lhs, rhs) in FALLBACK_RULES:
                fallback_count += 1
        return (fallback_count, len(prods))

    def parse(self, sentence: str) -> Optional[nltk.Tree]:
        """Return the best `nltk.Tree` for `sentence`, or `None` if ungrammatical."""
        for tokens in self.tokenize_candidates(sentence):
            trees = list(self.chart_parser.parse(tokens))
            if trees:
                return min(trees, key=self._tree_score)
        return None

    def parse_to_bracket(self, sentence: str) -> str:
        """Format parse tree on a single line, or `()` if out-of-grammar."""
        tree = self.parse(sentence)
        if tree is None:
            return "()"
        return tree.pformat(margin=1000000)

    def parse_file(self, input_path: str | Path, output_path: str | Path) -> List[str]:
        """Parse all sentences in `input/sentences.txt` -> `output/parse-results.txt`."""
        in_file = Path(input_path)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not in_file.exists():
            out_file.write_text("", encoding="utf-8")
            return []

        lines = in_file.read_text(encoding="utf-8").splitlines()
        results = [self.parse_to_bracket(line) for line in lines]
        out_file.write_text(
            "\n".join(results) + ("\n" if results else ""), encoding="utf-8"
        )
        return results
