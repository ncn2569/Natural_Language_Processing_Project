# Course Assistant - NLP (CO3085)

## Cài đặt & chạy
Yêu cầu: Docker.

```bash
# Điền STUDENT_ID trong util.sh (dạng MSSV1-MSSV2-MSSV3-MSSV4), rồi:
./util.sh test      # build + chạy trong Docker, kết quả ở ./<STUDENT_ID>/output
./util.sh submit    # tạo file <STUDENT_ID>.zip
```

Chạy local (không Docker):
```bash
cd python
pip install -r requirements.txt
python -m hcmut.iaslab.nlp.app.generate
```
Khi chạy trong Docker, chương trình đọc `/nlp/input` và ghi `/nlp/output`.

## Kiến trúc (hiện tại: Phần I - 3.1, 3.2)
```
data/grammar.txt  ->  generate.py  ->  output/grammar.txt, output/samples.txt
```

## Grammar
CFG viết tay, mỗi dòng `LHS -> alt1 | alt2`, `ε` là chuỗi rỗng. Gồm 7 nhóm câu hỏi:
GET_COURSE_INFO, GET_TOPIC, GET_SCHEDULE, GET_ASSIGNMENT, GET_DEADLINE, GET_RESOURCE, GET_RULE.

## Bộ sinh câu (3.2)
Đếm số câu mỗi non-terminal; nếu tổng ≤ 10.000 thì liệt kê hết, ngược lại lấy mẫu ngẫu nhiên
không trùng (seed cố định). Câu viết hoa đầu, dấu `?` dính từ cuối.

## Phần II - Classical NLP (Lựa chọn 1)
Pipeline: `Input -> Parsing (Phần I) -> Semantic -> Intent/Entity -> Query -> KB -> Answer`
(`app/part2.py`, `nlu.py`, `qa.py`, `kb.py`).

- **Knowledge base** (`data/course/*.txt`): course_info, topics, schedule, resources, regulations, faq, entities,
  assignments. `kb.py` đọc từng file thành cấu trúc (chương, tuần, LO, tài liệu, quy định...). Mọi câu trả lời
  chỉ lấy từ đây. `assignments.txt` hiện là BẢN NHÁP dựng từ đề bài (xem ghi chú trong file) - thay bằng file chính thức.
- **Entity**: luật regex (Chương N, Tuần N, L.O.x.y, LLM/RAG, AI, Midterm/Final, Bài tập lớn, Phần I-IV) + từ điển
  `entities.txt` (chủ đề và từ đồng nghĩa); khớp dài nhất, không chồng lấp.
- **Intent**: luật có thứ tự ưu tiên + keyword cue (UNKNOWN -> CONTEXT_DEPENDENT -> LO -> DEADLINE -> INSTRUCTOR ->
  RESOURCE -> RULE -> ASSIGNMENT -> COURSE_INFO -> SCHEDULE -> TOPIC). Không dùng LLM.
- **Cây cú pháp (Phần I)**: đọc `output/parse-results.txt` (bracket, `()` = bị từ chối). Hiện nhãn gốc (Q_TOPIC...) chỉ để
  đối chiếu với kết quả luật (map ở `TREE_LABEL_TO_INTENT` trong `nlu.py`); luật từ vựng là nguồn quyết định, và nếu
  parser từ chối câu thì vẫn dùng luật (ghi `parse: REJECTED` trong `intent-entity.txt`). Cần sửa nếu muốn semantic
  phụ thuộc hoàn toàn vào cây.
- **Semantic representation**: dạng vị từ, ví dụ `GET_CREDIT(course=CO3085)`, `GET_TOPIC_CHAPTER(topic=CFG)`,
  `CHECK_CHAPTER_COVERS(chapter=CH11, topic=QUESTION_ANSWERING)`.
- **Query**: pseudo-SQL (`query.txt`) + thực thi trên KB. Tìm chương theo chủ đề: keyword > tên mục > summary.
- **Không tìm thấy / ngoài phạm vi**: trả lời "Xin lỗi, ... không tìm thấy thông tin này trong dữ liệu của môn học."
- Câu phụ thuộc ngữ cảnh ("Nó...", "Còn chương sau?") được nhận diện là CONTEXT_DEPENDENT; giải quyết ở Phần III.

Đánh giá sơ bộ: `output/evaluation.txt` (34 câu mẫu, so intent và entity chính với nhãn trong `sample_queries.txt`).

## Phần II - LLM/RAG (Lựa chọn 2, phần mở rộng - `app/rag.py`)
Pipeline: `Câu hỏi -> cổng phạm vi -> Retrieve (BM25 + rerank theo entity) -> Context -> Prompt -> LLM -> Answer + nguồn`.
Chạy riêng: `python -m hcmut.iaslab.nlp.app.rag`; kết quả ở `output/rag/` (`retrieval.txt`, `answer.txt`, `evaluation.txt`).

- **Chunking**: mỗi chương, mỗi tuần, mỗi L.O., mỗi tài liệu, mỗi quy định, mỗi FAQ, mỗi phần bài tập lớn là một chunk có id
  (vd `topic:CH03`, `schedule:WEEK_03`, `regulation:LLM_RAG`) -> dùng làm nguồn trích dẫn.
- **Retrieval**: BM25 tự cài đặt (không thư viện ngoài) + mở rộng truy vấn bằng từ điển chủ đề (`hỏi đáp` -> `question answering`)
  + cộng điểm chunk khớp đúng thực thể (Chương N, Tuần N, L.O., phần bài tập lớn, quy định) lấy từ bộ entity của Phần II-Classical;
  FAQ nói về thực thể khác bị trừ điểm. Top-k = 4, bỏ chunk yếu hơn 50% điểm top-1.
- **Prompt**: `SYSTEM` bắt buộc chỉ dùng NGỮ CẢNH, thiếu thì nói đúng câu "không tìm thấy", ghi nguồn `[id]`; ngữ cảnh là các chunk
  dạng `[id] nội dung` (xem `build_prompt`).
- **LLM**: nếu có biến môi trường `ANTHROPIC_API_KEY` thì gọi Claude (model `RAG_MODEL`, mặc định `claude-haiku-5-5`; chưa kiểm tra
  với API thật trong môi trường phát triển). Không có key hoặc lỗi mạng -> chế độ **extractive**: trả lại chunk/câu khớp nhất kèm `[nguồn]`.
  Khi nộp bài nhóm cần ghi rõ chế độ nào đã dùng để tạo `output/rag/*`.
- **Chống bịa**: (1) cổng ngoài phạm vi / cần ngữ cảnh (dùng intent của Phần II); (2) thực thể được hỏi không có trong KB
  (Chương 13, Tuần 20, L.O.9.9, giảng viên) -> "không tìm thấy"; (3) ngưỡng điểm retrieval; (4) hậu kiểm: mọi con số trong câu trả lời
  phải có trong ngữ cảnh, nếu không thì chặn.
- **Hạn chế đã biết**: "Học phí môn NLP là bao nhiêu?" vẫn bị trả lời bằng nội dung không liên quan (thông tin không có trong KB nhưng
  từ khóa "NLP" khớp chunk khác); hậu kiểm chỉ bắt được sai số liệu, không bắt được bịa tên/khái niệm; đánh giá retrieval dùng nhãn
  suy ra từ `sample_queries.txt` nên có phần lạc quan.

## Parser / Phần III / Phần IV / Hạn chế
_(bổ sung khi làm các phần còn lại)_ Hạn chế hiện tại: luật tay nên diễn đạt lạ có thể trượt; câu "bài tập" chung chung
bị coi là ngoài phạm vi; chưa có dữ liệu giảng viên và deadline cụ thể trong KB.
