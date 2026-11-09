"""Sentence generator built on `nltk.CFG` (Section 3.2)."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk
from nltk.grammar import Nonterminal

Template = Tuple[str, ...]

# Core sentence patterns from `Sentences -> Sentence -> Clause | WhClause | YesNoQuestion | Request`
#Khởi động sinh câu từ 7 cấu trúc câu chính này để đảm bảo câu sinh ra cân bằng giữa các loại câu hỏi và yêu cầu.
CORE_PATTERNS: List[Tuple[str, ...]] = [
    ("Subject", "WhPredicate"),  #Câu hỏi Wh: Tuần 5 học gì?
    ("WhNounPhrase", "Predicate"),  # Chương nào nói về CFG?
    ("Subject", "Predicate", "QuestionParticle"), # Câu Yes/No: Bài tập lớn có dùng AI không?
    ("RequestVerb", "WhClause"), # Câu yêu cầu: Cho biết môn NLP ở đâu
    ("RequestVerb", "NounPhrase"), # Tra cứu bài tập lớn
    ("ImperativeMarker", "VerbPhrase"),  # Hãy học...
    ("Subject", "Predicate"), # Câu trần thuật
]
#Bỏ qua các luật tự đệ quy lặp từ (ví dụ NounHead -> Noun NounHead dễ sinh ra "môn môn môn"
SKIPPED_FALLBACK_RULES = {
    ("NounHead", ("Noun", "NounHead")),
    ("Modifiers", ("Modifier", "Modifiers")),
    ("Clause", ("Predicate",)),
    ("WhClause", ("WhPredicate",)),
}


class SentenceGenerator:
    """
    Generates valid sentences from an `nltk.CFG`:
    - Uses bounded BFS over `nltk.CFG.productions()` to avoid `nltk.parse.generate`'s
      exponential recursion limit on cyclic rules.
    - Instantiates pre-terminal templates in round-robin order up to `max_samples`.
    """

    def __init__(
        self,
        grammar: nltk.CFG,
        max_template_len: int = 8,
        max_templates_per_branch: int = 400,
    ) -> None:
        self.grammar = grammar
        self.max_template_len = max_template_len
        self.max_templates_per_branch = max_templates_per_branch

        # Group NLTK productions by LHS symbol name
        self.by_lhs: Dict[str, List[nltk.Production]] = {}
        for prod in grammar.productions():
            self.by_lhs.setdefault(prod.lhs().symbol(), []).append(prod)

        # Pre-terminals are non-terminals whose productions are all lexical
        self.preterminals: Set[str] = {
            lhs
            for lhs, prods in self.by_lhs.items()
            if prods and all(p.is_lexical() for p in prods)
        }

        self.lexical_map: Dict[str, List[str]] = {
            lhs: [str(p.rhs()[0]) for p in prods if p.is_lexical()]
            for lhs, prods in self.by_lhs.items()
            if lhs in self.preterminals
        }

    def _collect_templates(self, initial_pattern: Tuple[str, ...]) -> List[Template]:
        """Expand syntactic non-terminals via bounded BFS down to pre-terminal templates."""
        templates: List[Template] = []
        seen: Set[Template] = set()
        queue: deque[Tuple[Tuple[str, ...], Set[nltk.Production]]] = deque(
            [(initial_pattern, set())]
        )

        while queue and len(templates) < self.max_templates_per_branch:
            current, used_prods = queue.popleft()
            if len(current) > self.max_template_len:
                continue

            # Tìm ký hiệu Non-terminal đầu tiên chưa phải là Pre-terminal
            expand_idx = next(
                (i for i, sym in enumerate(current) if sym not in self.preterminals),
                -1,
            )
             # Nếu tất cả đã là Pre-terminal (ví dụ: ["NamePrefix", "CourseName", "WhPlace"])
            if expand_idx == -1:
                if current not in seen:
                    seen.add(current)
                    templates.append(current)
                continue

            target = current[expand_idx]
            for prod in self.by_lhs.get(target, []):
                # Không áp dụng cùng một luật quá 1 lần trên 1 nhánh (tránh lặp vô hạn)
                if prod in used_prods:
                    continue
                # ... Thay thế và đẩy nhánh mới vào hàng đợi queue ...
                rhs_names = tuple(
                    s.symbol() if isinstance(s, Nonterminal) else str(s)
                    for s in prod.rhs()
                )
                if (target, rhs_names) in SKIPPED_FALLBACK_RULES:
                    continue

                next_syms = current[:expand_idx] + rhs_names + current[expand_idx + 1 :]
                if len(next_syms) <= self.max_template_len:
                    queue.append((next_syms, used_prods | {prod}))

        templates.sort(
            key=lambda t: (
                0
                if any(
                    s
                    in {
                        "CourseName",
                        "TopicName",
                        "IndexedNoun",
                        "ExamName",
                        "OutcomeName",
                        "ToolName",
                        "AssignmentName",
                    }
                    for s in t
                )
                else 1,
                0
                if any(
                    s
                    in {
                        "ObjectVerb",
                        "PrepositionalVerb",
                        "Copula",
                        "RequestVerb",
                        "WhQuantity",
                        "WhPlace",
                    }
                    for s in t
                )
                else 1,
                len(t),
            )
        )
        return templates

    def _instantiate_template(
        self, template: Template, max_per_template: int
    ) -> List[str]:
        """ Lấy danh sách từ khả dĩ cho từng vị trí"""
        slot_options = [self.lexical_map.get(sym, [sym]) for sym in template]
        # Tính tổng số tổ hợp có thể tạo ra
        radix_total = 1
        for opts in slot_options:
            radix_total *= len(opts)
        # Chọn bước nhảy coprime với tổng số tổ hợp
        limit = min(radix_total, max_per_template)
        stride = 1
        if radix_total > limit:
            stride = max(1, radix_total // limit)
            while math.gcd(stride, radix_total) != 1:
                stride += 1

        results: List[str] = []
        seen_local: Set[str] = set()
        k = len(slot_options)
        idx = 0

        for _ in range(min(radix_total, limit * 2)):
            # Giải mã chỉ số `idx` thành từ ở từng vị trí (mixed-radix indexing)
            if len(results) >= limit:
                break
            rem = idx
            words: List[str] = []
            for s_idx in range(k - 1, -1, -1):
                opts = slot_options[s_idx]
                base = len(opts)
                choice_idx = rem % base
                # Tránh sinh 2 từ giống nhau qua liên từ "và"/"hoặc" (ví dụ: "nlp và nlp")
                if (
                    s_idx >= 2
                    and template[s_idx - 1] == "Conjunction"
                    and template[s_idx] == template[s_idx - 2]
                    and base > 1
                ):
                    choice_idx = (choice_idx + 1) % base
                words.append(opts[choice_idx])
                rem //= base
            words.reverse()
            # Bước nhảy idx tiếp theo
            sent = " ".join(words)
            if sent not in seen_local:
                seen_local.add(sent)
                results.append(sent)
            idx = (idx + stride) % radix_total

        return results

    def generate(self, max_samples: int = 10000) -> List[str]:
        """ Thu thập templates từ các nhánh CORE_PATTERNS"""
        branch_lists = [self._collect_templates(pat) for pat in CORE_PATTERNS]
        # Xen kẽ các templates từ từng nhánh (interleave)
        interleaved: List[Template] = []
        max_len = max((len(b) for b in branch_lists), default=0)
        for i in range(max_len):
            for b in branch_lists:
                if i < len(b):
                    interleaved.append(b[i])

        if not interleaved:
            return []

        quota = max(15, (max_samples // len(interleaved)) + 10)
        # Sinh trước các pool câu cho từng template
        pools = [self._instantiate_template(t, quota) for t in interleaved]

        samples: List[str] = []
        seen: Set[str] = set()
        round_idx = 0
        max_pool = max((len(p) for p in pools), default=0)

        # Lấy xoay vòng (round-robin): Mỗi template lấy 1 câu, rồi lặp lại vòng tiếp theo
        while len(samples) < max_samples and round_idx < max_pool:
            for pool in pools:
                if round_idx < len(pool):
                    sent = pool[round_idx]
                    if sent not in seen:
                        seen.add(sent)
                        samples.append(sent)
                        if len(samples) >= max_samples:
                            break
            round_idx += 1

        return samples

    def generate_to_file(
        self, output_path: str | Path, max_samples: int = 10000
    ) -> List[str]:
        """Generate sentences and write them to `output/samples.txt`."""
        samples = self.generate(max_samples=max_samples)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(samples) + "\n", encoding="utf-8")
        return samples
