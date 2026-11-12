# Course Assistant — Trợ lý Học tập Thông minh (CO3085)

> **Báo cáo Kỹ thuật & Tài liệu Hướng dẫn Hệ thống Course Assistant**  
> Môn học: Xử lý Ngôn ngữ Tự nhiên (CO3085) — Khoa KH&KT Máy tính, ĐH Bách Khoa TP.HCM.

---

## Thông tin Nhóm & Đề tài

* **Tên đề tài:** Hệ thống Trợ lý Hỗ trợ Học tập (Course Assistant) cho môn Xử lý Ngôn ngữ Tự nhiên.
* **Lựa chọn thực hiện:** Nhóm chọn **Lựa chọn 2: Mở rộng bằng LLM/RAG**, được xây dựng kế thừa chặt chẽ và thể hiện đầy đủ kết quả từ nền tảng **Classical NLP** (Parser cú pháp Phần I $\rightarrow$ Biểu diễn Vị từ Ngữ nghĩa $\rightarrow$ Trích xuất Thực thể & Ý định $\rightarrow$ Cổng Guardrail chống ảo giác cho RAG).

---

## Bảng Tóm tắt Trạng thái Triển khai

| Hạng mục / Yêu cầu đề bài | File mã nguồn chính | File Output tương ứng | Trạng thái |
|---|---|---|:---:|
| **3.1 Tập luật & Văn phạm (CFG)** | `src/grammar/cfg.py`, `grammar.txt` | `output/grammar.txt` | **Hoàn thành** |
| **3.2 Giải thuật sinh câu mẫu** | `src/grammar/generator.py` | `output/samples.txt` *(5.000 câu)* | **Hoàn thành** |
| **3.3 Bộ phân tích cú pháp (Parser)** | `src/grammar/parser.py` | `output/parse-results.txt` | **Hoàn thành** *(20/20 test pass)* |
| **4.1.1 Biểu diễn ngữ nghĩa (Semantic)** | `src/hcmut/.../app/nlu.py` | `output/semantic.txt` | **Hoàn thành** *(Dạng vị từ)* |
| **4.1.2 Nhận diện Intent & Entity** | `src/hcmut/.../app/nlu.py` | `output/intent-entity.txt` | **Hoàn thành** *(Nối cây cú pháp)* |
| **4.1.3 Truy vấn Knowledge Base** | `src/hcmut/.../app/qa.py` | `output/query.txt` *(Pseudo-SQL)* | **Hoàn thành** |
| **4.1.4 Truy xuất & Sinh câu trả lời** | `src/hcmut/.../app/qa.py`, `kb.py` | `output/answer.txt` | **Hoàn thành** |
| **4.2 Mở rộng LLM/RAG** | `src/hcmut/.../app/rag.py` | `output/retrieval.txt`, `output/rag/*` | **Hoàn thành** *(BM25 + Chống ảo giác)* |
| **5. Ngữ cảnh hội thoại (Phần III)** | `src/hcmut/.../app/nlu.py` | `output/intent-entity.txt` | Nhận diện câu `CONTEXT_DEPENDENT` |
| **6. Đánh giá hệ thống (Phần IV)** | `src/hcmut/.../app/evaluate.py` | `output/evaluation.txt` | **Hoàn thành** *(Đạt 100% Intent/Entity)* |

---

## 1. Cài đặt & Hướng dẫn Chạy chương trình

Hệ thống được thiết kế hoàn toàn khép kín, **chỉ sử dụng thư viện chuẩn của Python và NLTK**, không phụ thuộc các framework RAG cồng kềnh, đảm bảo có thể chạy 100% offline mà không cần API key.

### Cách 1: Chạy bằng Docker (Khuyên dùng khi chấm bài)
Build context là thư mục `project/`. Mọi kết quả output sẽ được ánh xạ trực tiếp về máy host qua Volume:
```bash
cd project

# 1. Build Docker image
docker build -t course-assistant .

# 2. Chạy container kiểm tra và xuất toàn bộ kết quả vào thư mục output/
docker run --rm -v "$(pwd)/output:/app/output" course-assistant
```

### Cách 2: Chạy trực tiếp trên máy (Local)
Yêu cầu: Python $\ge$ 3.10 (Đã kiểm thử chuẩn trên Python 3.13 và 3.14).
```bash
cd project

# Cài đặt thư viện (chỉ gồm nltk==3.10.4)
pip install -r requirements.txt

# Chạy toàn bộ pipeline tích hợp (Phần I -> Phần II Classical -> RAG -> Đánh giá)
python main.py

# Chạy bộ 20 unit tests kiểm thử cú pháp và giải thuật
python -m unittest discover -s tests -v
```

---

## 2. Kiến trúc Hệ thống (System Architecture)

Toàn bộ hệ thống Course Assistant vận hành theo một pipeline thống nhất, kết nối chặt chẽ từ Cú pháp $\rightarrow$ Ngữ nghĩa $\rightarrow$ Tri thức $\rightarrow$ RAG:

```
[Câu hỏi đầu vào: input/sentences.txt]
                 │
                 ▼
     ┌───────────────────────┐
     │  1. Chart Parser      │ ◄── [Văn phạm CFG: src/grammar/grammar.txt]
     │     (Phần I - 3.3)    │ ──► [output/parse-results.txt]
     └───────────┬───────────┘
                 │ (Cây cú pháp dạng ngoặc đơn hoặc ())
                 ▼
     ┌───────────────────────┐
     │  2. NLU & Semantics   │ ◄── [Từ điển & Thực thể: data/entities.txt]
     │     (Phần II - 4.1)   │ ──► [output/semantic.txt] (Vị từ logic)
     │                       │ ──► [output/intent-entity.txt] (Intent, Entity)
     └───────────┬───────────┘
                 │
        ┌────────┴─────────────────────────────────┐
        │                                          │
        ▼                                          ▼
┌───────────────────────────────┐  ┌───────────────────────────────────────┐
│ 3A. Classical QA Pipeline     │  │ 3B. LLM / RAG Pipeline (Mục 4.2)      │
│  - Sinh pseudo-SQL            │  │  - Chunking Knowledge Base            │
│  - Query Knowledge Base       │  │  - BM25 Retrieval + Entity Reranking  │
│  - Ghép câu trả lời chuẩn xác │  │  - Guardrails kiểm soát chống ảo giác │
│    [output/query.txt]         │  │  - Chế độ Extractive / Claude API     │
│    [output/answer.txt]        │  │    [output/retrieval.txt]             │
└───────────────────────────────┘  └───────────────────────────────────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  4. Module Đánh giá       │ ◄── [data/sample_queries.txt]
                    │     (Phần IV - Mục 6)     │ ──► [output/evaluation.txt]
                    └───────────────────────────┘
```

---

## 3. Mô tả Chi tiết các Thành phần Kỹ thuật

### 3.1. Văn phạm CFG & Bộ sinh câu (Phần I — Mục 3.1, 3.2)
* **Văn phạm (`src/grammar/grammar.txt`):** 
  Xây dựng theo trường phái **Syntactic Grammar** (Chapter 2, 3, 5 trong giáo trình) gồm 5 phân vùng rõ rệt: Cấu trúc câu (`Clause`, `WhClause`, `YesNoQuestion`, `Request`), Danh ngữ (`NounPhrase`), Động ngữ (`VerbPhrase`), Từ chức năng và Từ vựng chuyên môn môn học. Bao phủ đầy đủ 7 nhóm câu hỏi môn học: thông tin chung, chủ đề, lịch học, bài tập lớn, deadline, tài liệu và quy định.
* **Bộ sinh câu (`src/grammar/generator.py`):** 
  Duyệt cây CFG bằng thuật toán đệ quy có kiểm soát ngân sách: giới hạn độ sâu tối đa (`max_depth=14`), số cụm terminal (`max_terminal_phrases=12`) và số lần thử (`max_attempts=100.000`). Cơ chế gieo seed cố định (`seed=42`) giúp tái lập chính xác 5.000 câu tự nhiên không trùng lặp (`output/samples.txt`).

### 3.2. Bộ phân tích Cú pháp (Parser — Phần I — Mục 3.3)
* **Thuật toán (`src/grammar/parser.py`):** 
  Sử dụng `nltk.ChartParser` kết hợp giải thuật **quay lui (backtracking) tách cụm từ tiếng Việt** theo độ dài giảm dần, cho phép nhận diện chính xác các terminal đa âm tiết (`"bài tập lớn"`, `"xử lý ngôn ngữ tự nhiên"`, `"công cụ ai"`).
* **Giải quyết nhập nhằng (Ambiguity Resolution - Chapter 7):** 
  Hàm heuristic `_tree_score()` chấm điểm và ưu tiên các cây phân tích ít dùng luật dự phòng (như ghép danh từ liên tiếp hay mệnh đề khuyết chủ ngữ), xuất cây chuẩn xác hoặc trả về `()` nếu câu không thuộc miền văn phạm.

### 3.3. Biểu diễn Ngữ nghĩa & Nhận diện Intent/Entity (Phần II — Mục 4.1.1, 4.1.2)
* **Cầu nối Cú pháp $\rightarrow$ Ngữ nghĩa (`src/hcmut/.../app/nlu.py`):**
  Hệ thống **không bỏ rơi cây cú pháp**, mà dùng hàm `tree_intent()` và `collect_tree_features()` để đọc trực tiếp các nút trên cây:
  - Nhánh `WhQuantity` + `UnitNoun (tín chỉ)` $\rightarrow$ `GET_COURSE_INFO`.
  - Nhánh `WhTime` + `Noun (deadline)` $\rightarrow$ `GET_DEADLINE`.
  - Nút `YesNoQuestion` xác định trực tiếp câu hỏi Đúng/Sai (`yes_no=True`).
  Kết quả từ cây cú pháp được kết hợp cùng bộ từ điển thực thể và luật từ vựng để tạo nên nguồn xác thực `source: rules+tree(agree)`.
* **Biểu diễn vị từ logic (Semantic Frame):**
  Ánh xạ câu thành các vị từ hình thức có cấu trúc (Chapter 8, 10 trong giáo trình), ví dụ:
  - *"Môn NLP có bao nhiêu tín chỉ?"* $\rightarrow$ `GET_CREDIT(course=CO3085)`
  - *"Tuần 5 học những nội dung gì?"* $\rightarrow$ `GET_WEEK_CONTENT(week=WEEK_05)`
  - *"Chương 11 có dạy question answering không?"* $\rightarrow$ `CHECK_CHAPTER_COVERS(chapter=CH11, topic=QUESTION_ANSWERING)`

### 3.4. Knowledge Base & QA Engine Classical (Phần II — Mục 4.1.3, 4.1.4)
* **Knowledge Base (`src/hcmut/.../app/kb.py`):**
  Nạp và bóc tách dữ liệu từ các file text thuần trong `data/` (`course_info.txt`, `topics.txt`, `schedule.txt`, `resources.txt`, `regulations.txt`, `faq.txt`, `assignments.txt`), tuyệt đối không hardcode kiến thức trong mã nguồn.
* **Procedural Semantics (`src/hcmut/.../app/qa.py`):**
  Chuyển đổi Semantic Frame thành các truy vấn giả lập pseudo-SQL (`output/query.txt`), ví dụ:
  `SELECT credits FROM course_info WHERE course_id='CO3085'`
  Engine thực thi câu truy vấn trực tiếp trên KB và ghép câu trả lời tự nhiên (`output/answer.txt`). Nếu câu hỏi nằm ngoài phạm vi hoặc không có dữ liệu, hệ thống đưa ra thông báo từ chối chuẩn: *"Xin lỗi, tôi không tìm thấy thông tin này trong dữ liệu của môn học."*

### 3.5. Mở rộng Hệ thống bằng LLM/RAG (Phần II — Mục 4.2)
* **Chunking (`src/hcmut/.../app/rag.py`):**
  Toàn bộ Knowledge Base được phân mảnh thành các chunk ngữ cảnh độc lập kèm ID định danh nguồn trích dẫn: `topic:CHxx`, `schedule:WEEK_xx`, `regulation:xxx`, `faq:Qxx`.
* **Thuật toán BM25 & Entity Reranking:**
  Cài đặt thuật toán xếp hạng BM25 thuần túy bằng Python (không cần thư viện bên ngoài). Kết hợp mở rộng truy vấn (query expansion) và cộng trọng số cho các chunk khớp với thực thể đã được bóc tách từ NLU Phần I.
* **Cơ chế Chống Bịa đặt (Hallucination Guardrails):**
  1. *Cổng phạm vi (Scope Gating):* Tận dụng bộ phân loại Intent của Phần II để chặn các câu hỏi `UNKNOWN` hoặc `CONTEXT_DEPENDENT` trước khi gọi mô hình.
  2. *Kiểm tra thực thể:* Chặn sớm các thực thể không tồn tại trong dữ liệu (ví dụ: Chương 13, Tuần 20, L.O.9.9).
  3. *Ngưỡng điểm BM25:* Lọc bỏ các tài liệu có điểm truy xuất dưới ngưỡng `MIN_SCORE`.
  4. *Hậu kiểm số liệu (Post-generation Grounding):* Mọi con số/mã xuất hiện trong câu trả lời bắt buộc phải có mặt trong ngữ cảnh trích xuất; nếu phát hiện số lạ sẽ tự động thu hồi và báo không tìm thấy.
* **Chế độ phát sinh linh hoạt:**
  - Nếu có biến môi trường `ANTHROPIC_API_KEY`: Gọi mô hình Claude API (`claude-haiku-5-5`) để sinh câu trả lời mượt mà kèm nguồn `[id]`.
  - Nếu chạy offline hoặc không có API key: Tự động kích hoạt chế độ **Extractive Offline** (trích xuất câu mang bằng chứng cao nhất từ top chunk kèm nguồn trích dẫn), đảm bảo an toàn tuyệt đối khi chấm tự động trên Docker.

---

## 4. Đánh giá Thực nghiệm (Phần IV — Mục 6)

Hệ thống được đánh giá tự động thông qua module [`evaluate.py`]trên tập kiểm thử chuẩn gồm 34 câu hỏi (`data/sample_queries.txt`) bao phủ đủ các dạng câu hợp lệ, câu ngoài phạm vi và câu cần ngữ cảnh:

```text
======================================================================
DANH GIA SO BO - Classical NLP (che do 1 luot, khong ngu canh)
- So cau kiem thu: 34
- Do chinh xac Intent: 34/34 = 100.0%
- Do chinh xac Entity: 31/31 = 100.0% (khong tinh cau can ngu canh)
- So cau sai: 0

DANH GIA RAG (Che do Extractive & Retrieval Evaluation)
- Tong cau kiem thu: 34 (KB: 28, Ngoai pham vi: 3, Can ngu canh: 3)
- Retrieval Hit@1: 23/28 = 82.1%
- Retrieval Hit@3: 24/28 = 85.7%
- Tu choi dung cau ngoai pham vi: 3/3 = 100.0%
- Nhan dien cau can ngu canh: 3/3 = 100.0%
- Ty le cau tra loi co trich dan nguon: 28/28 = 100.0%
- Kiem tra chong Hallucination: Phat hien va chan thanh cong so lieu bia
======================================================================
```

---
