"""Gemini-backed answer generation for the RAG pipeline."""
import time
from google import genai
from google.genai import errors
from config import GEMINI_API_KEY, GEMINI_MODEL
from src.generation.prompt import build_prompt


class GeminiGenerator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Thiếu GEMINI_API_KEY. Hãy tạo file .env từ .env.example.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def answer(self, question: str, context: str) -> str:
        if not context or not context.strip():
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu của môn học."

        prompt = build_prompt(question, context)
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                return text.strip() if text else "Gemini không trả về nội dung câu trả lời."
            except errors.ServerError as exc:
                if attempt == 2:
                    raise RuntimeError(f"Gemini gặp lỗi máy chủ/quá tải sau 3 lần thử: {exc}") from exc
                delay = 2 ** (attempt + 1)
                print(f"Gemini tạm thời lỗi máy chủ; thử lại sau {delay} giây...")
                time.sleep(delay)
            except Exception as exc:
                raise RuntimeError(f"Gemini API lỗi ({type(exc).__name__}): {exc}") from exc
        raise RuntimeError("Không thể tạo câu trả lời bằng Gemini.")
