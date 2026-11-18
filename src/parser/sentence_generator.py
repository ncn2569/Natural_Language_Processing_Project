
"""Phần I: Sinh câu từ Context-Free Grammar."""

from __future__ import annotations

import random
from pathlib import Path

from nltk.parse.generate import generate
import nltk


class SentenceGenerator:
    """Sinh câu dựa trên CFG sử dụng NLTK."""

    def __init__(
        self,
        grammar: nltk.CFG,
        seed: int = 42,
        max_depth: int = 10,
    ) -> None:
        self.grammar = grammar
        self.seed = seed
        self.max_depth = max_depth
        self.last_attempts = 0

    def generate_to_file(
        self,
        output_path: str | Path,
        n: int = 5000,
    ) -> list[str]:
        """Sinh tối đa n câu và ghi vào file."""

        sentences = list(
            generate(
                self.grammar,
                depth=self.max_depth,
                n=n,
            )
        )

        self.last_attempts = len(sentences)

        # Xáo trộn thứ tự câu bằng seed cố định.
        rng = random.Random(self.seed)
        rng.shuffle(sentences)

        results = [" ".join(tokens) for tokens in sentences]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(results) + ("\n" if results else ""),
            encoding="utf-8",
        )

        return results
