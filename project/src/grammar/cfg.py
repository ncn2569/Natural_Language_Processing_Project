"""Phần I, mục 3.1: nạp grammar nguồn và xuất bản sao dùng để nộp bài."""

from __future__ import annotations

import shutil
from pathlib import Path

import nltk


def load_cfg(grammar_path: str | Path) -> nltk.CFG:
    """Đọc file UTF-8 thành CFG; thiếu file hoặc sai cú pháp phải báo lỗi.

    Terminal là chuỗi trong dấu nháy; nonterminal là nhãn như NounPhrase.
    Grammar của dự án dùng terminal chữ thường để khớp bước chuẩn hóa câu.
    """
    raw_text = Path(grammar_path).read_text(encoding="utf-8")
    try:
        return nltk.CFG.fromstring(raw_text)
    except ValueError as error:
        raise ValueError(f"Grammar không hợp lệ tại {grammar_path}: {error}") from error


def export_grammar(grammar_src: str | Path, grammar_out: str | Path) -> None:
    """Xuất bản sao; chỉ sửa grammar nguồn, không sửa bản trong output/."""
    output = Path(grammar_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grammar_src, output)
