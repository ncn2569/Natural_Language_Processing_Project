
"""Load and split the course knowledge base into retrieval chunks."""

from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class CourseDocument:
    source: str
    text: str
    chunk_id: int = 0
    data_type: str = "course_content"


def _classify_source(filename: str) -> str:
    """
    Classify documents according to their roles in the knowledge base.

    course_content: Main course information.
    faq_data: Frequently asked questions.
    sample_data: Illustrative or non-official information.
    supporting_data: Supporting examples and reference data.
    """

    name = filename.lower()

    # Main course knowledge
    if name in {
        "course_info.txt",
        "topics.txt",
        "resources.txt",
    }:
        return "course_content"

    # FAQ information
    if name == "faq.txt":
        return "faq_data"

    # Sample or illustrative information
    if name in {
        "schedule.txt",
        "assignments.txt",
        "regulations.txt",
    }:
        return "sample_data"

    # Supporting information
    if name in {
        "00_readme.txt",
        "sample_queries.txt",
        "sentences.txt",
        "dialogues.txt",
        "entities.txt",
    }:
        return "supporting_data"

    return "course_content"


def load_documents(course_dir: str | Path) -> list[CourseDocument]:
    """
    Load all supported documents from the knowledge base.

    Args:
        course_dir: Knowledge base directory.

    Returns:
        List of loaded course documents.
    """

    root = Path(course_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục knowledge base: {root}"
        )

    documents = []

    for path in sorted(root.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        if not content:
            continue

        documents.append(
            CourseDocument(
                source=path.name,
                text=content,
                data_type=_classify_source(path.name),
            )
        )

    if not documents:
        raise ValueError(
            f"Không tìm thấy tài liệu hợp lệ trong: {root}"
        )

    return documents


def split_documents(
    documents: list[CourseDocument],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[CourseDocument]:
    """
    Split documents into smaller overlapping chunks.

    The splitting process prefers paragraph, line and word boundaries
    to preserve the meaning of the original content.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Cần chunk_size > overlap >= 0."
        )

    chunks = []

    for document in documents:

        text = document.text
        start = 0
        chunk_id = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            if end < len(text):

                candidates = [
                    text.rfind("\n\n", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end),
                ]

                boundary = max(candidates)

                if boundary > start + chunk_size // 2:
                    end = boundary

            piece = text[start:end].strip()

            if piece:

                chunks.append(
                    CourseDocument(
                        source=document.source,
                        text=piece,
                        chunk_id=chunk_id,
                        data_type=document.data_type,
                    )
                )

                chunk_id += 1

            if end >= len(text):
                break

            start = max(start + 1, end - overlap)

    return chunks
