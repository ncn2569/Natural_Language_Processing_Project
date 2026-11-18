
"""End-to-end pipeline: CFG Parser -> Semantic Analysis -> TF-IDF RAG -> Gemini."""

from pathlib import Path

from config import (
    GRAMMAR_PATH,
    COURSE_DIR,
    INDEX_PATH,
    OUTPUT_DIR,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

from src.parser.grammar_loader import load_cfg
from src.parser.course_parser import CourseParser

from src.semantic.semantic_analyzer import analyze_semantics

from src.retrieval.document_loader import (
    load_documents,
    split_documents,
)

from src.retrieval.vector_store import CourseVectorStore


NO_ANSWER = "Không tìm thấy thông tin trong dữ liệu."


# Ưu tiên nguồn tài liệu dựa trên loại câu hỏi.
PREFERRED_SOURCES = {
    "GET_DEFINITION": [
        "topics.txt",
        "course_info.txt",
        "entities.txt",
    ],
    "GET_COURSE_INFO": [
        "course_info.txt",
        "faq.txt",
    ],
    "GET_RESOURCE": [
        "resources.txt",
    ],
    "GET_TOPIC": [
        "topics.txt",
        "course_info.txt",
    ],
    "GET_SCHEDULE": [
        "schedule.txt",
    ],
    "GET_ASSIGNMENT": [
        "assignments.txt",
    ],
    "GET_RULE": [
        "regulations.txt",
        "faq.txt",
    ],
    "GET_DEADLINE": [
        "assignments.txt",
        "schedule.txt",
    ],
    "GET_LO": [
        "course_info.txt",
        "faq.txt",
    ],
}


class CourseAssistant:

    def __init__(self):
        # 1. Load CFG grammar.
        grammar = load_cfg(GRAMMAR_PATH)
        self.parser = CourseParser(grammar)

        # 2. Initialize local TF-IDF vector store.
        self.store = CourseVectorStore(INDEX_PATH)

        # 3. Lazy initialization of Gemini.
        self._llm = None

    # --------------------------------------------------
    # INGEST
    # --------------------------------------------------

    def ingest(self) -> int:
        """Load documents, split them and build TF-IDF index."""

        documents = load_documents(COURSE_DIR)
        chunks = split_documents(documents)

        self.store.rebuild(chunks)

        return len(chunks)

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    def _get_llm(self):
        """Initialize Gemini only when it is needed."""

        if self._llm is None:
            from src.generation.llm import GeminiGenerator
            self._llm = GeminiGenerator()

        return self._llm

    # --------------------------------------------------
    # CONTEXT
    # --------------------------------------------------

    @staticmethod
    def _format_context(items: list[dict]) -> str:
        """Convert retrieved chunks into LLM context."""

        context_parts = []

        for index, item in enumerate(items, start=1):

            source = item.get("source", "unknown")
            content = item.get("text", "")

            context_parts.append(
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"Content:\n{content}"
            )

        return "\n\n".join(context_parts)

    # --------------------------------------------------
    # INTENT-BASED RANKING
    # --------------------------------------------------

    @staticmethod
    def _rank_for_intent(
        items: list[dict],
        intent: str,
    ) -> list[dict]:
        """Adjust retrieval ranking according to question intent."""

        preferred = PREFERRED_SOURCES.get(intent, [])

        ranked = []

        for item in items:

            source = item.get("source", "").lower()

            base_score = item.get(
                "ranking_score",
                item.get("similarity", 0.0),
            )

            if source in preferred:
                priority = 2.5 - preferred.index(source) * 0.3
            else:
                priority = 1.0

            item["intent_score"] = base_score * priority

            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                item.get("intent_score", 0.0),
                item.get("similarity", 0.0),
            ),
            reverse=True,
        )

        return ranked

    # --------------------------------------------------
    # ASK
    # --------------------------------------------------

    def ask(self, question: str) -> dict:
        """Process a question and generate an answer."""

        # 1. CFG parsing.
        parse_tree = self.parser.parse(question)

        # 2. Semantic analysis.
        semantic = analyze_semantics(
            question,
            parse_tree,
        )

        intent = semantic.intent

        # 3. TF-IDF retrieval.
        candidate_k = max(TOP_K * 4, 20)

        retrieved = self.store.search(
            semantic.search_query,
            candidate_k,
        )

        # 4. Remove supporting examples from main candidates.
        candidates = [
            item
            for item in retrieved
            if item.get("data_type") != "supporting_data"
        ]

        # 5. Rank according to intent.
        ranked = self._rank_for_intent(
            candidates,
            intent,
        )

        preferred_sources = [
            source.lower()
            for source in PREFERRED_SOURCES.get(intent, [])
        ]

        # 6. Filter relevant documents.
        accepted = [
            item
            for item in ranked
            if (
                item.get("similarity", 0.0)
                >= SIMILARITY_THRESHOLD
                or (
                    item.get("source", "").lower()
                    in preferred_sources
                    and item.get("similarity", 0.0) >= 0.02
                )
            )
        ][:TOP_K]

        # 7. Generate answer.
        answer = NO_ANSWER
        generation_status = "NO_ANSWER"

        if intent == "UNKNOWN":

            answer = (
                "Xin lỗi, tôi chưa nhận diện được "
                "loại câu hỏi này."
            )
            generation_status = "UNKNOWN_INTENT"

        elif not accepted:

            answer = NO_ANSWER
            generation_status = "NO_ANSWER"

        else:

            context = self._format_context(accepted)

            try:
                answer = self._get_llm().answer(
                    question,
                    context,
                )

                if answer:
                    generation_status = "GEMINI_SUCCESS"
                else:
                    answer = NO_ANSWER
                    generation_status = "NO_ANSWER"

            except Exception as error:

                answer = (
                    "Không thể tạo câu trả lời từ Gemini: "
                    f"{error}"
                )
                generation_status = "GEMINI_ERROR"

        # --------------------------------------------------
        # SAVE RETRIEVAL OUTPUT
        # --------------------------------------------------

        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        retrieval_lines = [
            f"QUESTION: {question}",
            f"INTENT: {intent}",
            f"SEARCH_QUERY: {semantic.search_query}",
            "",
            "RETRIEVAL RESULTS",
            "=" * 60,
        ]

        accepted_ids = {
            id(item)
            for item in accepted
        }

        for index, item in enumerate(ranked, start=1):

            source = item.get("source", "unknown")
            chunk_id = item.get("chunk", "unknown")

            similarity = item.get("similarity", 0.0)
            ranking_score = item.get("ranking_score", 0.0)
            intent_score = item.get("intent_score", 0.0)

            is_accepted = id(item) in accepted_ids

            retrieval_lines.append(
                f"{index}. {source} | "
                f"chunk={chunk_id} | "
                f"similarity={similarity:.4f} | "
                f"ranking_score={ranking_score:.4f} | "
                f"intent_score={intent_score:.4f} | "
                f"accepted={is_accepted}"
            )

            retrieval_lines.append(
                str(item.get("text", ""))
            )

            retrieval_lines.append("-" * 60)

        (output_path / "retrieval.txt").write_text(
            "\n".join(retrieval_lines),
            encoding="utf-8",
        )

        # --------------------------------------------------
        # SAVE ANSWER OUTPUT
        # --------------------------------------------------

        answer_text = (
            f"QUESTION: {question}\n\n"
            f"INTENT: {intent}\n"
            f"ENTITIES: {semantic.entities}\n\n"
            f"PARSER_STATUS: {semantic.parser_status}\n"
            f"GENERATION_STATUS: {generation_status}\n"
            f"PARSE_TREE:\n{semantic.parse_tree}\n\n"
            f"ANSWER:\n{answer}\n"
        )

        (output_path / "answer.txt").write_text(
            answer_text,
            encoding="utf-8",
        )

        # --------------------------------------------------
        # RETURN RESULT
        # --------------------------------------------------

        return {
            "question": question,
            "semantic": semantic.to_dict(),
            "retrieved": retrieved,
            "answer": answer,
            "generation_status": generation_status,
        }
