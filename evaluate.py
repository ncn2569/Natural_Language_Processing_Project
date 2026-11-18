"""Evaluate intent classification and retrieval against course/sample_queries.txt.

This script does not call Gemini; it evaluates the local NLP/RAG stages.
"""
from pathlib import Path
from config import COURSE_DIR, OUTPUT_DIR
from src.retrieval.document_loader import load_documents, split_documents
from src.retrieval.vector_store import CourseVectorStore
from src.semantic.semantic_analyzer import analyze_semantics
from config import INDEX_PATH, TOP_K


def read_cases(path: Path):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 4 and parts[0].startswith("Q"):
            cases.append((parts[0], parts[1], parts[2], parts[3]))
    return cases


def main():
    store = CourseVectorStore(INDEX_PATH)
    if not store.is_ready():
        docs = load_documents(COURSE_DIR)
        store.rebuild(split_documents(docs))

    cases = read_cases(COURSE_DIR / "sample_queries.txt")
    evaluated = [case for case in cases if case[2] != "CONTEXT_DEPENDENT"]
    correct = 0
    lines = ["COURSE ASSISTANT - LOCAL EVALUATION", f"TOTAL_CASES: {len(cases)}"]

    for case_id, question, expected_intent, expected_entity in cases:
        semantic = analyze_semantics(question, None)
        results = store.search(semantic.search_query, TOP_K)
        actual_intent = semantic.intent
        is_supported = expected_intent != "CONTEXT_DEPENDENT"
        matched = actual_intent == expected_intent if is_supported else None
        if is_supported and matched:
            correct += 1
        top = results[0] if results else None
        top_similarity = f"{top['similarity']:.4f}" if top else "0.0000"
        lines.append(
            f"\n{case_id} | {question}\n"
            f"expected_intent={expected_intent} actual_intent={actual_intent} "
            f"intent_match={matched}\n"
            f"expected_entity={expected_entity} actual_entities={semantic.entities}\n"
            f"top1_source={top['source'] if top else 'NONE'} "
            f"top1_similarity={top_similarity}"
        )

    lines.insert(2, f"INTENT_ACCURACY_SUPPORTED_ONLY: {correct}/{len(evaluated)}")
    lines.append("\nNOTE: CONTEXT_DEPENDENT cases are excluded from intent accuracy; this version does not implement dialogue memory.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Đã đánh giá local pipeline. Kết quả: output/evaluation.txt")
    print(f"Intent accuracy trên các case không phụ thuộc hội thoại: {correct}/{len(evaluated)}")


if __name__ == "__main__":
    main()
