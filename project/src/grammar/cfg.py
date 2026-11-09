"""Load and export NLTK Context-Free Grammar (Section 3.1)."""

from __future__ import annotations

import shutil
from pathlib import Path

import nltk


def load_cfg(grammar_path: str | Path) -> nltk.CFG:
    """Load an `nltk.CFG` directly from the grammar file."""
    raw_text = Path(grammar_path).read_text(encoding="utf-8")
    return nltk.CFG.fromstring(raw_text)


def export_grammar(grammar_src: str | Path, grammar_out: str | Path) -> None:
    """Copy the grammar definition to `output/grammar.txt` (Section 3.1)."""
    out_path = Path(grammar_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grammar_src, out_path)
