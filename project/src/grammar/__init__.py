"""Các thành phần Phần I: đọc CFG, sinh câu và phân tích cú pháp."""

from .cfg import load_cfg, export_grammar
from .generator import SentenceGenerator
from .parser import CourseParser

__all__ = [
    "load_cfg",
    "export_grammar",
    "SentenceGenerator",
    "CourseParser",
]
