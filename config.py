"""Central configuration for CourseAssistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

COURSE_DIR = ROOT_DIR / "course"
OUTPUT_DIR = ROOT_DIR / "output"
DATABASE_DIR = ROOT_DIR / "database"
GRAMMAR_PATH = ROOT_DIR / "src" / "parser" / "course_grammar.cfg"
INDEX_PATH = DATABASE_DIR / "tfidf_index.joblib"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.12"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

if TOP_K < 1:
    raise ValueError("TOP_K phải lớn hơn hoặc bằng 1.")
if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
    raise ValueError("SIMILARITY_THRESHOLD phải nằm trong khoảng [0, 1].")
