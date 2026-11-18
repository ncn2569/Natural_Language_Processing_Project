
"""
Local TF-IDF retrieval using character n-grams and cosine similarity.

The retrieval system combines:
1. TF-IDF similarity.
2. Source priority.
3. Document type priority.

It runs locally without downloading neural embedding models.
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CourseVectorStore:

    # Source priority weights.
    # These weights influence ranking, not the original similarity score.
    SOURCE_WEIGHTS = {
        "course_info.txt": 1.35,
        "topics.txt": 1.15,
        "resources.txt": 1.05,
        "faq.txt": 1.15,
    }

    DATA_TYPE_WEIGHTS = {
        "course_content": 1.0,
        "faq_data": 0.95,
        "supporting_data": 0.75,
        "sample_data": 0.65,
    }

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)

        self.vectorizer = None
        self.matrix = None
        self.chunks = []

        self.load()

    def _get_source_weight(self, item: dict) -> float:
        """
        Return source priority weight.

        Explicit source weights take priority over data type weights.
        """

        source = item["source"].lower()

        if source in self.SOURCE_WEIGHTS:
            return self.SOURCE_WEIGHTS[source]

        data_type = item.get(
            "data_type",
            "course_content"
        )

        return self.DATA_TYPE_WEIGHTS.get(
            data_type,
            1.0
        )

    def rebuild(self, chunks):
        """
        Build TF-IDF index from document chunks.
        """

        if not chunks:
            raise ValueError(
                "Knowledge base không có chunk để lập chỉ mục."
            )

        self.chunks = [
            {
                "source": chunk.source,
                "text": chunk.text,
                "chunk": chunk.chunk_id,
                "data_type": getattr(
                    chunk,
                    "data_type",
                    "course_content"
                ),
            }
            for chunk in chunks
        ]

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            strip_accents=None,
            sublinear_tf=True,
            min_df=1,
        )

        texts = [
            item["text"]
            for item in self.chunks
        ]

        self.matrix = self.vectorizer.fit_transform(texts)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "chunks": self.chunks,
            },
            self.index_path
        )

        print(
            f"Đã xây dựng TF-IDF index: "
            f"{len(self.chunks)} chunks."
        )

    def load(self):
        """
        Load previously saved TF-IDF index.
        """

        if not self.index_path.exists():
            return

        data = joblib.load(self.index_path)

        self.vectorizer = data.get("vectorizer")
        self.matrix = data.get("matrix")
        self.chunks = data.get("chunks", [])

    def is_ready(self) -> bool:
        """
        Check whether the vector store is ready.
        """

        return (
            self.vectorizer is not None
            and self.matrix is not None
            and bool(self.chunks)
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Retrieve relevant chunks using TF-IDF similarity
        and source-aware ranking.

        similarity:
            Original cosine similarity.

        ranking_score:
            Similarity adjusted by source priority.
        """

        if not self.is_ready():
            return []

        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).ravel()

        results = []

        for index, raw_score in enumerate(scores):

            item = self.chunks[index].copy()

            similarity = float(raw_score)

            source_weight = self._get_source_weight(
                item
            )

            ranking_score = similarity * source_weight

            item["similarity"] = similarity

            item["source_weight"] = source_weight

            item["ranking_score"] = ranking_score

            results.append(item)

        # Sort by adjusted ranking score.
        # Original similarity is preserved for threshold filtering.
        results.sort(
            key=lambda item: (
                item["ranking_score"],
                item["similarity"]
            ),
            reverse=True
        )

        return results[:max(1, top_k)]
