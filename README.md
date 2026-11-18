# CourseAssistant – Phần II: Mở rộng LLM/RAG

## 1. Mục tiêu và nguyên tắc triển khai

Project thực hiện lựa chọn 2 của Phần II. Phần I được giữ làm đầu vào của pipeline: CFG grammar, `load_cfg()` và `CourseParser.parse()` tiếp tục được sử dụng để phân tích câu hỏi và ghi parse tree. Phần II bổ sung semantic representation, retrieval, context construction và sinh câu trả lời bằng Gemini.

**Pipeline:**

`User Question -> Part-I CFG Parser -> Semantic Analysis -> TF-IDF Retrieval -> Retrieved Context -> Gemini LLM -> Answer`

Retrieval sử dụng TF-IDF character n-grams và cosine similarity. Đây là phương pháp local, phù hợp với knowledge base nhỏ và không cần tải embedding model. Trong báo cáo cần gọi đúng là TF-IDF retrieval, không mô tả đây là neural embedding.

## 2. Cài đặt (Windows CMD)

Khuyến nghị Python 3.11; có thể dùng Python 3.13 nếu các thư viện tương thích.

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Cấu hình Gemini

Sao chép `.env.example` thành `.env`, sau đó điền API key của bạn:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.8-flash
```

Tên model phải là model ID mà API key của bạn được phép sử dụng. Nếu gặp lỗi 404, kiểm tra model ID trong Google AI Studio. Nếu gặp 503, đó là lỗi máy chủ/quá tải tạm thời; retrieval vẫn được ghi ra file để có thể kiểm tra độc lập. Không chia sẻ `.env` hoặc API key.

## 4. Nạp knowledge base

```cmd
python main.py --ingest
```

Các file `.txt` và `.md` trong `course/` được tải, chia thành chunks và lập chỉ mục tại `database/tfidf_index.joblib`.

## 5. Đặt câu hỏi

```cmd
python main.py --ask "Môn NLP có bao nhiêu tín chỉ?"
```

Nếu chưa có index, hệ thống tự động nạp dữ liệu. Kết quả được lưu tại:

- `output/retrieval.txt`: parse status/tree của Phần I, semantic representation, top-k kết quả, similarity, nguồn và trạng thái chấp nhận.
- `output/answer.txt`: câu hỏi, intent/entities, context đã truy xuất, nguồn và câu trả lời.

## 6. Kiểm thử

```cmd
pytest -q
python evaluate.py
python test_gemini.py
```

`pytest` kiểm tra retrieval local. `test_gemini.py` kiểm tra riêng API Gemini, không phụ thuộc parser hoặc vector store.

## 7. Phạm vi và độ tin cậy dữ liệu

Theo `course/00_README.txt`, nội dung kiến thức môn học được xây dựng dựa trên đề cương CO3085 HK261. Các file `schedule.txt`, `assignments.txt`, `regulations.txt` chứa dữ liệu mẫu, không phải thông tin hành chính chính thức. Hệ thống gắn nhãn `sample_data` vào các chunk này và prompt yêu cầu Gemini phân biệt chúng.

## 8. Tham số cần đánh giá

- `TOP_K`: số chunk lấy từ retrieval.
- `SIMILARITY_THRESHOLD`: ngưỡng cosine similarity để chấp nhận context.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: kích thước và phần chồng lấp chunk.

Ngưỡng hiện tại là giá trị khởi tạo, không phải kết quả tối ưu đã được chứng minh. Cần chạy bộ câu hỏi trong `course/sample_queries.txt`, xem `output/retrieval.txt`, ghi nhận trường hợp truy xuất sai/thiếu và điều chỉnh tham số. Các câu hỏi tham chiếu như “Nó có liên quan đến parsing không?” hoặc “Còn chương sau?” cần discourse context nếu muốn xử lý chính xác; bản triển khai này chưa tự suy ra lịch sử hội thoại.

## 9. Cấu trúc chính

```text
CourseAssistant/
├── config.py
├── main.py
├── test_gemini.py
├── course/                 # Knowledge base mẫu
├── database/               # TF-IDF index được sinh khi ingest
├── output/                 # retrieval.txt và answer.txt
├── src/
│   ├── parser/              # Phần I: giữ nguyên grammar/parser
│   ├── semantic/            # Intent/entity/search query
│   ├── retrieval/           # Load, chunk, TF-IDF, cosine similarity
│   ├── generation/          # Prompt và Gemini
│   └── pipeline.py
└── tests/
```
