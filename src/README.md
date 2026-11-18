# CourseAssistant – Hướng dẫn cài đặt và sử dụng

## 1. Yêu cầu môi trường

- Python 3.11 (khuyến nghị) hoặc Python 3.13.
- Google Gemini API Key.
- Windows CMD hoặc Terminal.

## 2. Cài đặt dự án

### Bước 1: Tạo môi trường ảo

```cmd
py -3.11 -m venv .venv
```

Kích hoạt môi trường ảo:

```cmd
.venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Cấu hình Gemini API

Tại file `.env`

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.8-flash
```

Thay `YOUR_GEMINI_API_KEY` bằng API Key của bạn.

Lưu ý:
- Model ID phải được Gemini API hỗ trợ và tài khoản của bạn có quyền sử dụng.
- Nếu gặp lỗi 404, hãy kiểm tra lại tên model.
- Nếu gặp lỗi 503, có thể máy chủ Gemini đang quá tải hoặc tạm thời không khả dụng.
- Không chia sẻ API Key và không đưa file `.env` lên GitHub.

## 4. Chạy chương trình

### Bước 1: Nạp dữ liệu và xây dựng chỉ mục TF-IDF

```cmd
python main.py --ingest
```

Chương trình sẽ:
- Đọc các tài liệu trong thư mục `course/`.
- Chia tài liệu thành các đoạn nhỏ (chunks).
- Xây dựng chỉ mục TF-IDF.
- Lưu chỉ mục tại `database/tfidf_index.joblib`.

Chỉ cần thực hiện lại bước này khi có thay đổi trong dữ liệu của thư mục `course/`.

### Bước 2: Đặt câu hỏi

```cmd
python main.py --ask "NLP là gì?"
```

Một số câu hỏi mẫu:

```cmd
python main.py --ask "Môn NLP có bao nhiêu tín chỉ?"
python main.py --ask "Môn học có bao nhiêu ECTS?"
python main.py --ask "CFG là gì?"
python main.py --ask "Môn học có những tài liệu tham khảo nào?"
python main.py --ask "Chương 3 nói về nội dung gì?"
```

### Bước 3: Kiểm tra kết quả

Sau khi chạy, chương trình hiển thị:
- Semantic Representation: Ý định và thực thể được nhận diện.
- Retrieval: Các đoạn tài liệu được truy xuất và điểm tương đồng.
- Answer: Câu trả lời được Gemini tạo ra.
- Generation Status: Trạng thái sinh câu trả lời.

Kết quả cũng được lưu tại:

```text
output/
├── retrieval.txt
└── answer.txt
```

## 5. Kiểm thử hệ thống

Chạy các lệnh sau để kiểm tra:

```cmd
pytest -q
python evaluate.py
python test_gemini.py
```

Ý nghĩa:
- `pytest -q`: Chạy các kiểm thử tự động.
- `evaluate.py`: Thực hiện đánh giá hệ thống nếu file này có trong dự án.
- `test_gemini.py`: Kiểm tra khả năng kết nối và sinh câu trả lời của Gemini API.

## 6. Cấu trúc thư mục chính

```text
CourseAssistant/
├── .env
├── .env.example
├── requirements.txt
├── config.py
├── main.py
├── README.md
├── test_gemini.py
├── evaluate.py
├── course/
├── database/
├── output/
├── src/
│   ├── parser/
│   ├── semantic/
│   ├── retrieval/
│   ├── generation/
│   └── pipeline.py
└── tests/
```

**Lưu ý:** Một số file như `evaluate.py`, `.env.example` hoặc thư mục `tests/` chỉ cần có nếu đã được tạo trong repository.