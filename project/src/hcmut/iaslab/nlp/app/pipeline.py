"""Diem vao cho Docker/hcommon.cli: Phan I (3.1, 3.2) -> Phan II (Classical + RAG) -> danh gia."""
from . import evaluate, generate, part2, rag


def run(**kwargs):
    generate.run(**kwargs)
    part2.run(**kwargs)
    evaluate.run(**kwargs)
    rag.run(**kwargs)
    evaluate.run_rag(**kwargs)
