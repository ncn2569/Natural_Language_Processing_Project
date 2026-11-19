# Course Assistant — Trợ lý Hỗ trợ Học tập Thông minh (CO3085)

> **Bài tập lớn môn Xử lý Ngôn ngữ Tự nhiên (CO3085)**  
> Khoa Khoa học và Kỹ thuật Máy tính — Trường Đại học Bách Khoa, ĐHQG-HCM.

---

## 1. Thành viên nhóm

| STT | Họ và tên | MSSV | Nội dung phụ trách |
| :---: | --- | :---: | --- |
| 1 | *(Điền họ tên)* | *(Điền MSSV)* | Phần I: Văn phạm CFG, sinh câu & Parser |
| 2 | *(Điền họ tên)* | *(Điền MSSV)* | Phần II: Semantic Representation, Intent/Entity, KB & QA (Lựa chọn 1) |
| 3 | *(Điền họ tên)* | *(Điền MSSV)* | Phần III: Xử lý ngữ cảnh hội thoại cục bộ (Dialogue) |
| 4 | *(Điền họ tên)* | *(Điền MSSV)* | Phần IV: Bộ kiểm thử, đánh giá hệ thống & báo cáo |

---

## 2. Lựa chọn thực hiện

* **Lựa chọn Phần II:** **Lựa chọn 1 — Classical/Statistical NLP (Mục 4.1)**. Hệ thống không sử dụng LLM/RAG, vận hành 100% dựa trên văn phạm phi ngữ cảnh (CFG), biểu diễn vị từ ngữ nghĩa, liên kết thực thể từ từ điển, truy vấn Knowledge Base có cấu trúc (pseudo-SQL) và sinh câu trả lời theo luật.
* **Phần nâng cao:** Triển khai **Phần III — Xử lý ngữ cảnh hội thoại cục bộ** (`output/dialogue.txt`).
* **Tổng hợp kết quả đầu ra:**

| Hạng mục đề bài | File mã nguồn chính | File Output |
| --- | --- | --- |
| **3.1 Văn phạm CFG** | `src/grammar/grammar.txt`, `src/grammar/cfg.py` | `output/grammar.txt` |
| **3.2 Sinh câu từ văn phạm** | `src/grammar/generator.py` | `output/samples.txt` *(5.000 câu)* |
| **3.3 Bộ phân tích cú pháp** | `src/grammar/parser.py` | `output/parse-results.txt` |
| **4.1.1 Biểu diễn ngữ nghĩa** | `src/hcmut/iaslab/nlp/app/nlu.py` | `output/semantic.txt` |
| **4.1.2 Intent & Entity** | `src/hcmut/iaslab/nlp/app/nlu.py` | `output/intent-entity.txt` |
| **4.1.3 Truy vấn Knowledge Base** | `src/hcmut/iaslab/nlp/app/qa.py` | `output/query.txt` |
| **4.1.4 Truy xuất & Sinh câu trả lời** | `src/hcmut/iaslab/nlp/app/qa.py`, `kb.py`, `part2.py` | `output/answer.txt` |
| **5. Ngữ cảnh hội thoại (Phần III)** | `src/hcmut/iaslab/nlp/app/dialogue.py` | `output/dialogue.txt` |
| **6. Đánh giá hệ thống (Phần IV)** | `src/hcmut/iaslab/nlp/app/evaluate.py` | `output/evaluation.txt`, `output/evaluation-results.json` |

---

## 3. Hướng dẫn cài đặt

Hệ thống chỉ sử dụng thư viện chuẩn của Python và `nltk==3.10.4` (không tải corpus ngoài, chạy 100% offline, không cần API key).

* **Yêu cầu môi trường:** Python $\ge$ 3.10 hoặc Docker.
* **Cài đặt thư viện (Local):**

```bash
cd project
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy chương trình

### Cách 1: Chạy trực tiếp trên máy (Local)

```bash
cd project
python main.py
```

Lệnh trên chạy trọn vẹn 4 phần theo thứ tự: Phần I (`grammar.txt`, `samples.txt`, `parse-results.txt`) $\rightarrow$ Phần II Classical (`semantic.txt`, `intent-entity.txt`, `query.txt`, `answer.txt`) $\rightarrow$ Phần III Dialogue (`dialogue.txt`) $\rightarrow$ Phần IV Evaluation (`evaluation.txt`, `evaluation-results.json`).

### Cách 2: Chạy bằng Docker

Build context là thư mục `project/`:

```bash
cd project

# Build Docker image
docker build -t course-assistant .

# Chạy container và ánh xạ thư mục output/ về máy host
docker run --rm -v "$(pwd)/output:/app/output" course-assistant
```

---

## 5. Kiến trúc hệ thống

```text
[input/sentences.txt]
         │
         ▼
┌─────────────────────────────┐
│ 1. CFG Parser (Phần I)      │ ◄── src/grammar/grammar.txt
│    src/grammar/parser.py    │ ──► output/grammar.txt, samples.txt, parse-results.txt
└──────────────┬──────────────┘
               │ (Cây cú pháp dạng ngoặc đơn hoặc ())
               ▼
┌─────────────────────────────┐
│ 2. Semantic & NLU (Phần II) │ ◄── data/entities.txt
│    src/hcmut/.../app/nlu.py │ ──► output/semantic.txt, intent-entity.txt
└──────────────┬──────────────┘
               │ (Semantic Frame: predicate, intent, entities, yes/no)
               ▼
┌─────────────────────────────┐
│ 3. KB Query & QA (Phần II)  │ ◄── data/{course_info,topics,schedule,assignments,resources,regulations,faq}.txt
│    kb.py, qa.py, part2.py   │ ──► output/query.txt, answer.txt
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. Dialogue Context (P.III) │ ◄── data/dialogues.txt
│    dialogue.py              │ ──► output/dialogue.txt
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. Evaluation (Phần IV)     │ ◄── data/evaluation_queries.txt (132 truy vấn)
│    evaluate.py              │ ──► output/evaluation.txt, evaluation-results.json
└─────────────────────────────┘
```

### Cấu trúc thư mục `project/`

```text
project/
├── Dockerfile
├── README.md
├── main.py
├── requirements.txt
├── data/                   # Knowledge Base, từ điển thực thể, hội thoại mẫu & bộ đánh giá
├── input/                  # sentences.txt (51 câu đầu vào cho Phần I & Phần II)
├── models/                 # Thư mục dự phòng model/config theo quy cách nộp bài
├── output/                 # Toàn bộ kết quả đầu ra Phần I–IV
└── src/
    ├── grammar/            # Phần I: grammar.txt, cfg.py, generator.py, parser.py
    └── hcmut/iaslab/nlp/app/
        ├── kb.py           # Nạp & bóc tách Knowledge Base từ data/*.txt
        ├── nlu.py          # Semantic representation + nhận diện Intent/Entity từ cây cú pháp
        ├── qa.py           # Sinh truy vấn pseudo-SQL và thực thi trên KB
        ├── part2.py        # Pipeline Phần II Classical NLP
        ├── dialogue.py     # Phần III: Xử lý ngữ cảnh hội thoại cục bộ
        └── evaluate.py     # Phần IV: Đánh giá theo nhóm & xuất báo cáo
```

---

## 6. Mô tả chi tiết các thành phần NLP

### 6.1. Văn phạm CFG & Sinh câu (`src/grammar/grammar.txt`, `cfg.py`, `generator.py`)
* **Văn phạm (CFG):** Xây dựng theo cấu trúc cú pháp tiếng Việt trong miền Course Assistant gồm 5 nhóm luật: cấu trúc câu (`Clause`, `WhClause`, `YesNoQuestion`, `Request`), danh ngữ (`NounPhrase`, `Nominal`, `Modifiers`), động ngữ (`VerbPhrase`, `WhVerbPhrase`), từ chức năng và từ vựng chuyên môn (môn học, chương, tuần, chuẩn đầu ra, kỳ thi, tài liệu, bài tập lớn, quy định).
* **Giải thuật sinh câu:** Duyệt sinh ngẫu nhiên có kiểm soát từ ký hiệu bắt đầu `Sentences` với giới hạn độ sâu (`max_depth=14`), số cụm terminal (`max_terminal_phrases=12`), ngân sách thử (`max_attempts=100.000`) và hạt giống cố định (`seed=42`) để tạo chính xác **5.000 câu hợp lệ không trùng lặp** vào `output/samples.txt`.

### 6.2. Bộ phân tích cú pháp (`src/grammar/parser.py`)
* Sử dụng `nltk.ChartParser` kết hợp **tách cụm từ đa âm tiết tiếng Việt theo giải thuật quay lui (backtracking)** ưu tiên cụm dài trước (như `"bài tập lớn"`, `"tài liệu tham khảo"`, `"công cụ ai"`), đồng thời giữ nguyên dấu chấm/gạch chéo trong mã thực thể (`L.O.2.5`, `LLM/RAG`).
* **Giải nhập nhằng cú pháp:** Hàm `_tree_score()` chấm điểm phạt các luật dự phòng (`Noun NounHead`, mệnh đề khuyết chủ ngữ) để chọn cây cú pháp ưu tiên nhất. Câu không hợp lệ hoặc ngoài văn phạm trả về `()`. Trên 51 dòng `input/sentences.txt`: **48 câu có cây hợp lệ, 3 câu ngoài văn phạm trả `()`**.

### 6.3. Biểu diễn ngữ nghĩa (`src/hcmut/iaslab/nlp/app/nlu.py`)
* Kết nối trực tiếp đầu ra cây cú pháp Phần I qua `parse_bracket()`, `collect_tree_features()` và `tree_intent()` để trích xuất đặc trưng cấu trúc (loại câu hỏi `WhQuantity`, `WhTime`, `YesNoQuestion`, danh từ trung tâm...).
* Ánh xạ sang khung vị từ logic có tham số (`SemanticFrame`) xuất vào `output/semantic.txt`, ví dụ:
  * *"Môn NLP có bao nhiêu tín chỉ?"* $\rightarrow$ `GET_CREDIT(course=CO3085)`
  * *"Tuần 5 học những nội dung gì?"* $\rightarrow$ `GET_WEEK_CONTENT(week=WEEK_05)`
  * *"Deadline bài tập lớn là khi nào?"* $\rightarrow$ `GET_DEADLINE(assignment=BTL01)`
  * *"Chương 11 có liên quan đến hỏi đáp không?"* $\rightarrow$ `CHECK_CHAPTER_COVERS(chapter=CH11, topic=QUESTION_ANSWERING)`

### 6.4. Nhận diện Intent & Entity (`src/hcmut/iaslab/nlp/app/nlu.py`)
* **Entity Linking (`EntityLinker`):** Nạp từ điển thực thể và từ đồng nghĩa từ `data/entities.txt` kết hợp biểu thức chính quy nhận diện `COURSE`, `CHAPTER`, `WEEK`, `LO`, `ASSIGN`, `PART`, `EXAM`, `TOPIC`, `RESOURCE_TOPIC`, `RULE`.
* **Intent Classification:** Kết hợp tín hiệu từ cây cú pháp (`tree_intent`) và luật từ khóa/thực thể để phân loại các intent: `GET_COURSE_INFO`, `GET_TOPIC`, `GET_SCHEDULE`, `GET_LO`, `GET_ASSIGNMENT`, `GET_DEADLINE`, `GET_RESOURCE`, `GET_RULE`, `GET_INSTRUCTOR`, `CONTEXT_DEPENDENT`, `UNKNOWN`. Xuất chi tiết vào `output/intent-entity.txt`.

### 6.5. Knowledge Base & Truy vấn (`kb.py`, `qa.py`, `part2.py`)
* **Knowledge Base (`kb.py`):** Đọc và cấu trúc hóa dữ liệu trực tiếp từ các file thuần trong `data/` (`course_info.txt`, `topics.txt`, `schedule.txt`, `assignments.txt`, `resources.txt`, `regulations.txt`, `faq.txt`), không hardcode kiến thức môn học trong mã nguồn.
* **Truy vấn & Sinh câu trả lời (`qa.py`):** Chuyển `SemanticFrame` thành câu truy vấn pseudo-SQL (`output/query.txt`), ví dụ `SELECT credits FROM course_info WHERE course_id='CO3085'`, thực thi truy xuất trên KB và sinh câu trả lời tiếng Việt kèm nguồn trích dẫn (`output/answer.txt`).
* **Xử lý thiếu thông tin / ngoài phạm vi:** Câu `UNKNOWN` trả thông báo ngoài phạm vi; thực thể không có trong KB (chương/tuần/LO không tồn tại, thiếu thông tin giảng viên) trả `"Xin lỗi, tôi không tìm thấy thông tin này trong dữ liệu của môn học."`.

### 6.6. Ngữ cảnh hội thoại cục bộ — Phần III (`dialogue.py`)
* `DialogueManager` duy trì `DiscourseState` (History List + Centering + `last_frame`) qua từng lượt hội thoại, xử lý 4 nhóm hiện tượng:
  1. Tham chiếu đại từ (*"nó"*).
  2. Mô tả xác định (*"chương này"*, *"tuần đó"*, *"phần này"*, *"kỳ thi này"*).
  3. Tham chiếu thứ tự (*"chương sau"*, *"tuần tiếp theo"*).
  4. Tỉnh lược (*"Còn final?"*, *"Còn chương sau?"*).
* Xuất đủ 4 trường theo đề bài (hội thoại đầu vào, ngữ cảnh sử dụng, kết quả diễn giải, câu trả lời) vào `output/dialogue.txt`.

---

## 7. Đánh giá hệ thống (Phần IV)

Module `src/hcmut/iaslab/nlp/app/evaluate.py` đánh giá tự động toàn bộ pipeline (Parser + Classical QA + DialogueManager) trên tập kiểm thử `data/evaluation_queries.txt` gồm **132 truy vấn** (99 câu độc lập + 6 hội thoại / 33 lượt), bao phủ đủ 5 nhóm yêu cầu của Mục 6 đề bài:

| Nhóm kiểm thử | Số truy vấn | Intent đúng | Entity chính đúng | Câu trả lời đúng |
| --- | :---: | :---: | :---: | :---: |
| Câu hỏi hợp lệ | 46 | 46/46 (100.00%) | 45/46 (97.83%) | 42/46 (91.30%) |
| Diễn đạt tương đương | 26 | 26/26 (100.00%) | 23/26 (88.46%) | 23/26 (88.46%) |
| Ngoài phạm vi | 13 | 10/13 (76.92%) | 10/13 (76.92%) | 10/13 (76.92%) |
| Thiếu thông tin trong KB | 14 | 11/14 (78.57%) | 12/14 (85.71%) | 5/14 (35.71%) |
| Hội thoại có ngữ cảnh (6 hội thoại) | 33 | 33/33 (100.00%) | 33/33 (100.00%) | 32/33 (96.97%) |
| **Tổng cộng** | **132** | **126/132 (95.45%)** | **123/132 (93.18%)** | **112/132 (84.85%)** |

* **Hội thoại đúng toàn bộ:** `5/6` hội thoại.
* Kết quả chi tiết từng câu được xuất tại `output/evaluation.txt` và `output/evaluation-results.json`.

---

## 8. Các hạn chế của hệ thống

Qua kết quả thực nghiệm (20/132 câu chưa khớp kỳ vọng), hệ thống còn các hạn chế đặc trưng của hướng Classical NLP dựa trên luật:

1. **Câu hỏi ngoài miền nhưng chứa từ khóa trùng với miền môn học (3/13 ca nhóm Ngoài phạm vi):** Khi không có tên môn học khác mà chỉ trùng từ khóa như *"tài liệu"*, *"quy định"*, *"môn học"* (ví dụ *"Tài liệu nào giúp tôi đầu tư chứng khoán?"*, *"Cho tôi quy định hoàn thuế thu nhập cá nhân."*), bộ phân loại luật vẫn gán vào `GET_RESOURCE`/`GET_RULE` thay vì `UNKNOWN`.
2. **Thuộc tính không có trong KB của thực thể hợp lệ (9/14 ca nhóm Thiếu thông tin):** Khi hỏi các thuộc tính không tồn tại trong dữ liệu mẫu (*phòng học, học phí, ngày/giờ thi, link tải slide, chính sách trừ điểm nộp muộn, điểm qua môn*), câu vẫn khớp thực thể `CO3085`/`MIDTERM`/`FINAL` và rơi vào nhánh trả lời mặc định (tóm tắt môn học, hình thức thi, danh sách quy định) thay vì báo không tìm thấy.
3. **Phụ thuộc độ phủ từ điển đồng nghĩa và paraphrase:** Các cách diễn đạt chưa có trong `data/entities.txt` hoặc từ điển mở rộng (như số thứ tự bằng chữ *"tuần thứ ba"*, cụm dịch tiếng Việt *"văn phạm phi ngữ cảnh"*, dạng số nhiều *"frames"*, hoặc chủ đề tài liệu *"machine translation"*) làm bước liên kết thực thể bỏ sót hoặc trả về mức tổng quát.
4. **Độ phủ văn phạm CFG và câu hỏi ghép nhiều ý:** Văn phạm CFG tập trung vào các cấu trúc hỏi đáp trọng tâm trong miền môn học; các câu dài, phức tạp hoặc nằm ngoài từ vựng CFG trả về `()` ở Parser và dựa vào luật từ vựng ở NLU.
5. **Giới hạn so khớp chuỗi mẫu khi đánh giá tự động:** Một số câu trả lời đúng về mặt dữ liệu KB (ví dụ lịch học tổng quan tuần 15 giữ nguyên tên tiếng Anh `Integration and Evaluation` từ `schedule.txt`, hoặc danh sách quy định ẩn mục `OUT_OF_SCOPE`) vẫn bị đánh sai bởi phép so khớp mẫu chính xác.
