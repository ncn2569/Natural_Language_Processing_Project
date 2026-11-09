"""Phần I, mục 3.2: sinh câu bằng cách mở rộng các luật CFG có giới hạn."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import nltk
from nltk.grammar import Nonterminal

DEFAULT_SAMPLE_COUNT = 5_000
MAX_SAMPLE_COUNT = 10_000


class SentenceGenerator:
    """Chọn luật → mở rộng vế phải → nhận câu hoàn chỉnh và chưa trùng.

    Giới hạn độ sâu, số terminal và số lần thử giúp thuật toán kết thúc ngay
    cả với grammar đệ quy. Câu sinh thuộc CFG nhưng có thể chưa tự nhiên về nghĩa.
    """

    def __init__(
        self,
        grammar: nltk.CFG,
        seed: int = 42,
        max_depth: int = 14,
        max_terminal_phrases: int = 12,
        max_attempts: int = 100_000,
    ) -> None:
        if min(max_depth, max_terminal_phrases, max_attempts) < 1:
            raise ValueError("Giới hạn độ sâu, số terminal và số lần thử phải dương.")
        self.grammar = grammar
        self.seed = seed
        self.max_depth = max_depth
        self.max_terminal_phrases = max_terminal_phrases
        self.max_attempts = max_attempts
        self.last_attempts = 0

        self.rules_by_symbol: dict[Nonterminal, list[nltk.Production]] = {}
        for production in grammar.productions():
            self.rules_by_symbol.setdefault(production.lhs(), []).append(production)

        # Đi qua các lớp bọc đơn như Sentences -> Sentence để luân phiên
        # các nhánh thật của grammar, thay vì viết tay bảy mẫu câu độc lập.
        branch_symbol = grammar.start()
        visited_symbols = set()
        while branch_symbol not in visited_symbols:
            visited_symbols.add(branch_symbol)
            branches = self.rules_by_symbol.get(branch_symbol, [])
            if len(branches) != 1 or len(branches[0].rhs()) != 1:
                break
            child = branches[0].rhs()[0]
            if not isinstance(child, Nonterminal):
                break
            branch_symbol = child
        self.branches = self.rules_by_symbol.get(branch_symbol, [])

    def _expand_symbol(
        self,
        symbol: str | Nonterminal,
        depth: int,
        terminals: list[str],
        rng: random.Random,
    ) -> bool:
        """Thêm terminal hoặc mở rộng nonterminal; False thì hủy cả lần thử.

        Vế trái của nhánh được chọn ở độ sâu 0; con của nó ở độ sâu 1.
        Nonterminal chỉ được mở rộng khi depth < max_depth. Terminal ở độ
        sâu max_depth vẫn được nhận vì không cần mở rộng thêm.
        """
        if isinstance(symbol, str):
            # Một terminal có thể chứa nhiều âm tiết, như "bài tập lớn".
            if len(terminals) >= self.max_terminal_phrases:
                return False
            terminals.append(symbol)
            return True

        rules = self.rules_by_symbol.get(symbol, [])
        if depth >= self.max_depth or not rules:
            return False

        # Cùng một luật được phép dùng cho cả chủ ngữ và tân ngữ.
        # Giới hạn độ sâu thay cho việc cấm dùng lại luật trong cả câu.
        chosen_rule = rng.choice(rules)
        for child in chosen_rule.rhs():
            if not self._expand_symbol(child, depth + 1, terminals, rng):
                return False
        return True

    def generate(self, max_samples: int = DEFAULT_SAMPLE_COUNT) -> list[str]:
        """Sinh tối đa số câu yêu cầu; cảnh báo nếu hết lượt thử trước khi đủ.

        Cùng grammar, cấu hình, seed và môi trường cho cùng danh sách theo thứ tự.
        Có thể yêu cầu 0 câu; số lượng vượt giới hạn của đề sẽ bị từ chối.
        """
        if not 0 <= max_samples <= MAX_SAMPLE_COUNT:
            raise ValueError(f"Số câu phải nằm trong khoảng 0 đến {MAX_SAMPLE_COUNT}.")

        # Random cục bộ không làm thay đổi trạng thái ngẫu nhiên của phần khác.
        rng = random.Random(self.seed)
        samples = []
        seen_sentences = set()
        self.last_attempts = 0

        while (
            len(samples) < max_samples
            and self.last_attempts < self.max_attempts
            and self.branches
        ):
            branch = self.branches[self.last_attempts % len(self.branches)]
            self.last_attempts += 1
            terminals: list[str] = []
            complete = True
            for symbol in branch.rhs():
                if not self._expand_symbol(symbol, 1, terminals, rng):
                    complete = False
                    break
            if not complete:
                continue  # Bỏ câu dang dở; lần thử sau dùng danh sách terminal mới.

            sentence = " ".join(terminals).strip()
            if sentence and sentence not in seen_sentences:
                seen_sentences.add(sentence)
                samples.append(sentence)

        # List giữ thứ tự xuất; set chỉ giúp kiểm tra trùng, không dùng để ghi file.
        if len(samples) < max_samples:
            warnings.warn(
                f"Chỉ sinh được {len(samples)}/{max_samples} câu sau "
                f"{self.last_attempts} lần thử; giới hạn độ sâu/số terminal hoặc "
                "số câu khác nhau của grammar có thể chưa cho phép đạt mục tiêu.",
                UserWarning,
                stacklevel=2,
            )
        return samples

    def generate_to_file(
        self, output_path: str | Path, max_samples: int = DEFAULT_SAMPLE_COUNT
    ) -> list[str]:
        """Ghi mỗi câu một dòng; trả danh sách thực tế để báo cáo số lượng."""
        samples = self.generate(max_samples)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(samples) + ("\n" if samples else ""), encoding="utf-8"
        )
        return samples
