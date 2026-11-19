# Course Assistant — Trợ lý Học tập Thông minh (CO3085)

> **Báo cáo Kỹ thuật & Tài liệu Hướng dẫn Hệ thống Course Assistant**  
> Môn học: Xử lý Ngôn ngữ Tự nhiên (CO3085) — Khoa KH&KT Máy tính, ĐH Bách Khoa TP.HCM.

---

## Thông tin Nhóm & Đề tài

* **Tên đề tài:** Hệ thống Trợ lý Hỗ trợ Học tập (Course Assistant) cho môn Xử lý Ngôn ngữ Tự nhiên.
* **Lựa chọn thực hiện:** Nhóm chọn **Lựa chọn 1: Classical/Statistical NLP** (Mục 4.1), triển khai đầy đủ pipeline NLP cổ điển: Văn phạm CFG & Parser cú pháp (Phần I) $\rightarrow$ Biểu diễn Vị từ Ngữ nghĩa & Trích xuất Intent/Entity $\rightarrow$ Truy vấn Knowledge Base & Sinh câu trả lời (Phần II) $\rightarrow$ Xử lý ngữ cảnh hội thoại cục bộ (Phần III) $\rightarrow$ Đánh giá hệ thống trên bộ 151 câu hỏi (Phần IV).

---

## Bảng Tóm tắt Trạng thái Triển khai

| Hạng mục / Yêu cầu đề bài | File mã nguồn chính | File Output tương ứng | Trạng thái |
|---|---|---|:---:|
| **3.1 Tập luật & Văn phạm (CFG)** | `src/grammar/cfg.py`, `grammar.txt` | `output/grammar.txt` | **Hoàn thành** |
| **3.2 Giải thuật sinh câu mẫu** | `src/grammar/generator.py` | `output/samples.txt` *(5.000 câu)* | **Hoàn thành** |
| **3.3 Bộ phân tích cú pháp (Parser)** | `src/grammar/parser.py` | `output/parse-results.txt` | **Hoàn thành** *(35/35 test pass)* |
| **4.1.1 Biểu diễn ngữ nghĩa (Semantic)** | `src/hcmut/.../app/nlu.py` | `output/semantic.txt` | **Hoàn thành** *(Dạng vị từ)* |
| **4.1.2 Nhận diện Intent & Entity** | `src/hcmut/.../app/nlu.py` | `output/intent-entity.txt` | **Hoàn thành** *(Nối cây cú pháp)* |
| **4.1.3 Truy vấn Knowledge Base** | `src/hcmut/.../app/qa.py` | `output/query.txt` *(Pseudo-SQL)* | **Hoàn thành** |
| **4.1.4 Truy xuất & Sinh câu trả lời** | `src/hcmut/.../app/qa.py`, `kb.py` | `output/answer.txt` | **Hoàn thành** |
| **5. Ngữ cảnh hội thoại (Phần III)** | `src/hcmut/.../app/dialogue.py` | `output/dialogue.txt` | **Hoàn thành** *(8 hội thoại mẫu)* |
| **6. Đánh giá hệ thống (Phần IV)** | `src/hcmut/.../app/evaluate.py` | `output/evaluation.txt` | **Hoàn thành** *(151 câu hỏi)* |

---

## 1. Cài đặt & Hướng dẫn Chạy chương trình

Hệ thống được thiết kế hoàn toàn khép kín, **chỉ sử dụng thư viện chuẩn của Python và NLTK**, đảm bảo chạy 100% offline, tái lập kết quả ổn định và không cần API key.

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

# Chạy toàn bộ pipeline tích hợp (Phần I -> Phần II Classical -> Phần III Dialogue -> Phần IV Đánh giá)
python main.py

# Chạy bộ 35 unit tests kiểm thử cú pháp, KB và hội thoại
python -m unittest discover -s tests -v
```

---

## 2. Kiến trúc Hệ thống (System Architecture)

Toàn bộ hệ thống Course Assistant vận hành theo một pipeline NLP cổ điển thống nhất, kết nối chặt chẽ từ Cú pháp $\rightarrow$ Ngữ nghĩa $\rightarrow$ Tri thức $\rightarrow$ Hội thoại $\rightarrow$ Đánh giá:

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
                 ▼
     ┌───────────────────────────────┐
     │  3. Classical QA Pipeline     │ ◄── [Knowledge Base: data/*.txt]
     │  - Sinh pseudo-SQL            │ ──► [output/query.txt]
     │  - Query Knowledge Base       │ ──► [output/answer.txt]
     │  - Guardrail ngoài phạm vi    │
     └───────────┬───────────────────┘
                 │
                 ▼
     ┌───────────────────────────────┐
     │  4. Local Discourse Context   │ ◄── [Hội thoại mẫu: data/dialogues.txt]
     │     (Phần III - Mục 5)        │ ──► [output/dialogue.txt]
     └───────────┬───────────────────┘
                 │
                 ▼
     ┌───────────────────────────────┐
     │  5. Module Đánh giá           │ ◄── [data/evaluation_queries.txt (151 câu)]
     │     (Phần IV - Mục 6)         │ ──► [output/evaluation.txt]
     └───────────────────────────────┘
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

### 3.5. Xử lý Ngữ cảnh Hội thoại Cục bộ (Phần III — Mục 5)
* **Dialogue Manager (`src/hcmut/.../app/dialogue.py`):**
  Duy trì trạng thái hội thoại (`DiscourseState`) với History List và Centering qua từng lượt hỏi đáp, giải quyết các hiện tượng tham chiếu đại từ (*"nó"*), mô tả xác định (*"phần này"*, *"chương này"*), tham chiếu thứ tự (*"chương sau"*, *"tuần tiếp theo"*) và tỉnh lược (ellipsis). Kết quả được ghi đầy đủ vào `output/dialogue.txt`.

---

## 4. Đánh giá Thực nghiệm (Phần IV — Mục 6)

Hệ thống được đánh giá tự động thông qua module `evaluate.py` trên tập kiểm thử mở rộng gồm 151 truy vấn (`data/evaluation_queries.txt`), bao phủ câu hỏi hợp lệ, diễn đạt tương đương, ngoài phạm vi, thiếu thông tin trong KB, tiền đề sai và 6 hội thoại nhiều lượt:

```text
ĐÁNH GIÁ TỔNG QUAN - CLASSICAL QA
Số truy vấn: 151
Intent đúng: 135/151 = 89.40%
Entity chính đúng: 130/151 = 86.09%
Số câu trả lời đúng: 114/151 = 75.50%
Số câu trả lời sai: 37/151
Hội thoại đúng toàn bộ: 4/6
```

---
