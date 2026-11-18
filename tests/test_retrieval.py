from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COURSE_DIR
from src.retrieval.document_loader import load_documents, split_documents
from src.retrieval.vector_store import CourseVectorStore


def make_store(tmp_path):
    docs = load_documents(COURSE_DIR)
    chunks = split_documents(docs)
    store = CourseVectorStore(tmp_path / "test_index.joblib")
    store.rebuild(chunks)
    return store


def test_course_credit_retrieval(tmp_path):
    store = make_store(tmp_path)
    results = store.search("Môn NLP có bao nhiêu tín chỉ?", top_k=5)
    assert results
    assert any("credits: 3" in item["text"].lower() or "3 tín chỉ" in item["text"].lower()
               for item in results)


def test_topic_retrieval(tmp_path):
    store = make_store(tmp_path)
    results = store.search("Chương nào nói về CFG?", top_k=5)
    assert results
    assert any("CHAPTER 3" in item["text"] or "Context-Free Grammars" in item["text"]
               or "CFG" in item["text"] for item in results)


def test_sample_data_is_marked():
    docs = load_documents(COURSE_DIR)
    schedule = next(doc for doc in docs if doc.source == "schedule.txt")
    assignments = next(doc for doc in docs if doc.source == "assignments.txt")
    regulations = next(doc for doc in docs if doc.source == "regulations.txt")
    assert schedule.data_type == "sample_data"
    assert assignments.data_type == "sample_data"
    assert regulations.data_type == "sample_data"
