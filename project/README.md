# Trợ lý Hỗ trợ Học tập Thông minh – Course Assistant (CO3085)

Tài liệu này mô tả chi tiết những thay đổi đã thực hiện cho **Phần I (Viết văn phạm, Giải thuật sinh câu và Bộ phân tích cú pháp)** trên nhánh `feat/parser-and-algo`, bao gồm: hướng dẫn chạy, ý nghĩa từng file code `.py`, các quyết định thiết kế (Design Decisions) và lý do (Reasoning) đằng sau từng quyết định.

---

## 1. Hướng dẫn cài đặt và chạy chương trình

### Cài đặt thư viện
Từ thư mục gốc của repository (`NLP/`):
```bash
pip install -r requirements.txt
```
*(Thư viện cốt lõi được thêm cho Phần I là `nltk>=3.6.0`).*

### Chạy toàn bộ pipeline Phần I
```bash
python3 project/main.py
```
Khi chạy lệnh trên, hệ thống thực hiện tuần tự 3 mục của Phần I:
1. **Mục 3.1**: Đọc văn phạm từ `project/src/grammar/grammar.txt` và xuất ra `project/output/grammar.txt`.
2. **Mục 3.2**: Sinh tối đa `10.000` câu hợp lệ thuộc miền môn học và ghi ra `project/output/samples.txt`.
3. **Mục 3.3**: Đọc danh sách các câu kiểm thử từ `project/input/sentences.txt`, phân tích cú pháp từng câu và ghi cây cú pháp (hoặc `()` nếu câu không hợp lệ / ngoài miền) ra `project/output/parse-results.txt`.

---

## 2. Cấu trúc thư mục & Ý nghĩa các file trong Phần I

```text
project/
├── main.py                      # Điểm vào chính điều phối toàn bộ pipeline Phần I (3.1 -> 3.2 -> 3.3)
├── src/
│   └── grammar/
│       ├── __init__.py          # Export các hàm và class chính của package grammar
│       ├── grammar.txt          # Tập luật văn phạm phi ngữ cảnh (CFG) gốc của miền Course Assistant
│       ├── cfg.py               # (Mục 3.1) Nạp grammar bằng nltk.CFG và xuất ra output/grammar.txt
│       ├── generator.py         # (Mục 3.2) Giải thuật sinh 10.000 câu hợp lệ bằng BFS 2 pha
│       └── parser.py            # (Mục 3.3) Bộ phân tích cú pháp dựa trên nltk.ChartParser + tách từ ghép
├── input/
│   └── sentences.txt            # 51 câu kiểm thử (48 câu hợp lệ từ đề bài/sample_queries/FAQ/dialogues + 3 câu ngoài miền)
└── output/
    ├── grammar.txt              # Output mục 3.1: Tập luật văn phạm
    ├── samples.txt              # Output mục 3.2: 10.000 câu sinh tự động từ văn phạm
    └── parse-results.txt        # Output mục 3.3: Cây cú pháp trên 1 dòng cho từng câu trong input/sentences.txt
```

---

## 3. Giải thích chi tiết từng file code `.py`

### 3.1. `project/main.py` — Điều phối Pipeline
- **`run_part1(base_dir: Path)`**:
  - Xác định đường dẫn tương đối tới `src/grammar/grammar.txt`, `input/sentences.txt` và thư mục `output/`.
  - Gọi `load_cfg` & `export_grammar` để hoàn thành mục 3.1.
  - Khởi tạo `SentenceGenerator(grammar)` và gọi `generate_to_file(..., max_samples=10000)` để hoàn thành mục 3.2.
  - Khởi tạo `CourseParser(grammar)` và gọi `parse_file(...)` để hoàn thành mục 3.3.

### 3.2. `project/src/grammar/cfg.py` — Nạp và xuất Grammar (Mục 3.1)
- **`load_cfg(grammar_path) -> nltk.CFG`**:
  - Đọc nội dung văn bản từ `src/grammar/grammar.txt` và truyền vào `nltk.CFG.fromstring(raw_text)`.
  - `nltk.CFG.fromstring` tự động bỏ qua các dòng trống và dòng comment bắt đầu bằng `#`, tách vế trái (`LHS`), vế phải (`RHS`) qua dấu `->` và các nhánh lựa chọn qua dấu `|`.
  - Các từ trong dấu ngoặc kép `"..."` được hiểu là **Terminal** (từ vựng lá), các ký hiệu không có ngoặc kép được hiểu là **Non-terminal** (`nltk.Nonterminal`).
- **`export_grammar(grammar_src, grammar_out)`**:
  - Tạo thư mục `output/` nếu chưa tồn tại và xuất nội dung văn phạm ra `output/grammar.txt`.

### 3.3. `project/src/grammar/parser.py` — Bộ phân tích cú pháp `CourseParser` (Mục 3.3)
Class `CourseParser` bọc quanh `nltk.ChartParser` và giải quyết bài toán từ ghép tiếng Việt cùng nhập nhằng cú pháp:
- **`__init__(self, grammar: nltk.CFG)`**:
  - Khởi tạo `self.chart_parser = nltk.ChartParser(grammar)`.
  - Trích xuất tập từ vựng `self.lexicon` từ tất cả các luật từ vựng (`prod.is_lexical()`) trong grammar (ví dụ: `"môn"`, `"môn học"`, `"bài tập lớn"`, `"l.o.2.5"`, `"llm/rag"`).
  - Tính `self.max_term_words`: số âm tiết tối đa của một terminal trong từ điển để giới hạn cửa sổ ghép từ.
- **`preprocess(sentence: str) -> List[str]`**:
  - Chuyển câu về chữ thường (`lower()`), loại bỏ dấu câu ở đầu/cuối câu (`?`, `.`, `!`, `,`, `;`, `:`, dấu ngoặc kép) nhưng **giữ nguyên** dấu `.` và `/` nằm bên trong từ (để không làm hỏng các thực thể như `"l.o.2.5"` hay `"llm/rag"`).
- **`tokenize_candidates(sentence: str, max_candidates: int = 4) -> List[List[str]]`**:
  - Tách câu đã chuẩn hóa thành danh sách các terminal có trong `self.lexicon` bằng thuật toán **khớp cụm từ dài nhất trước (Longest-Match First) kết hợp quay lui (Backtracking)**.
  - Nếu câu chứa bất kỳ từ nào không nằm trong `self.lexicon` (ví dụ câu ngoài miền có từ `"mưa"`, `"tổng thống"`, `"pháp"`), hàm dừng ngay và trả về danh sách rỗng `[]`.
- **`_tree_score(tree: nltk.Tree) -> Tuple[int, int]`**:
  - Chấm điểm cây cú pháp theo bộ `(fallback_count, len(prods))` để chọn cây chuẩn nhất khi câu có nhiều cây cú pháp hợp lệ.
- **`parse(sentence: str) -> Optional[nltk.Tree]`**:
  - Duyệt qua các cách tách từ từ `tokenize_candidates`, gọi `self.chart_parser.parse(tokens)` và trả về cây `nltk.Tree` có `_tree_score` nhỏ nhất (tốt nhất).
- **`parse_to_bracket(sentence: str) -> str` & `parse_file(input_path, output_path)`**:
  - Nếu câu hợp lệ, dùng `tree.pformat(margin=1000000)` để xuất cây cú pháp dưới dạng ngoặc đơn trên **1 dòng duy nhất**.
  - Nếu câu không hợp lệ hoặc ngoài phạm vi, xuất `()`.

### 3.4. `project/src/grammar/generator.py` — Bộ sinh câu `SentenceGenerator` (Mục 3.2)
Class `SentenceGenerator` sinh tối đa `10.000` câu hợp lệ từ đối tượng `nltk.CFG`:
- **`__init__(self, grammar: nltk.CFG, ...)`**:
  - Gom nhóm các luật `grammar.productions()` theo ký hiệu vế trái (`self.by_lhs`).
  - Xác định tập **Pre-terminals** (`self.preterminals`): các ký hiệu Non-terminal chỉ sinh trực tiếp ra từ vựng Terminal (như `CourseName`, `TopicName`, `Noun`, `ObjectVerb`, `WhQuantity`...), và ánh xạ từng Pre-terminal sang danh sách từ vựng (`self.lexical_map`).
- **`_collect_templates(initial_pattern) -> List[Template]` (Pha 1 - Sinh khung câu bằng BFS)**:
  - Khởi chạy duyệt chiều rộng (BFS) từ 7 mẫu câu cốt lõi (`CORE_PATTERNS`: câu hỏi Wh có chủ ngữ, câu hỏi Wh chủ ngữ, câu hỏi Yes/No, câu yêu cầu Request, câu mệnh lệnh, câu trần thuật).
  - Ở mỗi bước, mở rộng ký hiệu Non-terminal đầu tiên chưa phải là Pre-terminal.
  - Mỗi nhánh lưu tập `used_prods` để không lặp lại cùng một luật đệ quy quá 1 lần trên cùng một nhánh. Khi toàn bộ ký hiệu trong chuỗi đều là Pre-terminals (ví dụ `("NamePrefix", "CourseName", "ObjectVerb", "WhQuantity", "UnitNoun")`), thu được một khung câu hoàn chỉnh.
- **`_instantiate_template(template, max_per_template) -> List[str]` (Pha 2 - Điền từ vựng)**:
  - Điền từ vựng từ `self.lexical_map` vào các vị trí trong khung câu bằng hệ đếm hỗn hợp (mixed-radix indexing) với bước nhảy `stride` nguyên tố cùng nhau (`math.gcd(stride, radix_total) == 1`).
  - Chặn hiện tượng lặp hai từ giống hệt nhau qua liên từ `"và"`/`"hoặc"` (`Conjunction`).
- **`generate(max_samples=10000)` & `generate_to_file(...)`**:
  - Lấy mẫu xoay vòng (round-robin) qua tất cả các khung câu của cả 7 nhóm mẫu câu cho đến khi đủ `10.000` câu không trùng lặp và ghi ra `output/samples.txt`.

---

## 4. Các quyết định thiết kế (Design Decisions) & Lý do (Reasoning)

### Quyết định 1: Sử dụng thư viện `nltk` (`nltk.CFG`, `nltk.ChartParser`, `nltk.Tree`)
- **Quyết định**: Dùng cấu trúc dữ liệu và bộ parser chuẩn của `nltk` thay vì tự viết bộ đọc file văn phạm và quy hoạch động từ đầu.
- **Lý do (Reasoning)**:
  1. **Ngắn gọn, dễ hiểu, dễ bảo trì**: Giảm hơn 60% số dòng code, giúp mọi thành viên trong nhóm đọc hiểu luồng xử lý ngay lập tức và dễ dàng giải thích khi vấn đáp.
  2. **Tương thích trực tiếp với Phần II & Phần III**: Hàm `CourseParser.parse()` trả về đối tượng `nltk.Tree` chuẩn. Khi làm Phần II (Semantic Representation / Intent & Entity Recognition), nhóm có thể gọi trực tiếp các hàm có sẵn của `nltk.Tree` như `tree.subtrees()`, `tree.label()`, `tree.pos()`, `tree.leaves()` mà không cần viết thêm hàm duyệt cây thủ công.

### Quyết định 2: Tự cài đặt thuật toán BFS 2 pha trên `nltk.CFG` thay vì gọi `nltk.parse.generate.generate()`
- **Quyết định**: Không gọi hàm `nltk.parse.generate.generate(grammar, n=10000)` mặc định của NLTK mà viết `SentenceGenerator` duyệt BFS 2 pha trên `grammar.productions()`.
- **Lý do (Reasoning)**:
  1. **Tránh lỗi sập đệ quy của NLTK**: Văn phạm của nhóm có các luật đệ quy tự nhiên như `Clause -> TimePhrase Clause`, `NounPhrase -> SimpleNounPhrase Conjunction NounPhrase`, `VerbCore -> Modal VerbPhrase`. Do hàm `generate()` của NLTK duyệt theo chiều sâu (DFS), khi chạy thực tế trên văn phạm này sẽ bị kẹt trong nhánh đệ quy và dừng đột ngột với lỗi:
     `ValueError: Refusing to generate further: a recursive grammar can derive an exponential number of sentences, and generation exceeded the limit of 1000000 derivation-expansion steps (CWE-400)`.
  2. **Đảm bảo độ phủ đa dạng trên 10.000 câu**: Nếu sinh tuần tự hết từ vựng của từng khung câu một, chỉ riêng 1 khung câu đầu tiên đã có hàng chục ngàn tổ hợp từ vựng, dẫn tới toàn bộ file `output/samples.txt` chỉ có duy nhất 1 cấu trúc câu. Việc tách thành Pha 1 (BFS tìm khung câu) + Pha 2 (xoay vòng round-robin và bước nhảy `gcd(stride, radix_total) == 1`) giúp 10.000 câu biến thiên đồng thời cả Chủ ngữ, Động từ, Tân ngữ và trải đều trên mọi loại câu (`WhClause`, `YesNoQuestion`, `Request`, `Clause`).

### Quyết định 3: Tách từ ghép khớp từ điển có quay lui (`tokenize_candidates`) trước khi đưa vào `nltk.ChartParser`
- **Quyết định**: Ghép các âm tiết thành terminal dựa trên `self.lexicon` theo hướng ưu tiên cụm dài nhất (longest-match) có quay lui (backtracking).
- **Lý do (Reasoning)**:
  1. `nltk.ChartParser.parse(tokens)` yêu cầu danh sách `tokens` đầu vào phải khớp chính xác với các chuỗi terminal trong grammar. Trong tiếng Việt, nếu chỉ dùng `.split()` theo khoảng trắng, `"bài tập lớn"` sẽ bị tách thành `["bài", "tập", "lớn"]` và không khớp với terminal `"bài tập lớn"`.
  2. Trong văn phạm có nhiều từ đơn và từ ghép chồng lấp nhau (`"môn"` vs `"môn học"`, `"bài tập"` vs `"bài tập lớn"`, `"tài liệu"` vs `"tài liệu tham khảo"`, `"có"` vs `"có thể"`, `"cho"` + `"biết"` vs `"cho biết"`). Cơ chế Longest-Match kèm Backtracking đảm bảo nếu cách ghép dài nhất không tạo thành câu hợp lệ, parser vẫn tự động thử cách tách ngắn hơn mà không bao giờ bị lỗi tách từ sai.

### Quyết định 4: Bảo toàn dấu `.` và `/` bên trong từ khi tiền xử lý (`preprocess`)
- **Quyết định**: Chỉ xóa dấu câu ở đầu và cuối từ, giữ nguyên dấu `.` và `/` nằm giữa từ.
- **Lý do (Reasoning)**:
  - Miền dữ liệu môn học có các mã chuẩn đầu ra và công cụ như `"L.O.2.5"`, `"L.O.4.2"`, `"LLM/RAG"`. Nếu xóa toàn bộ dấu câu trên toàn chuỗi, `"L.O.2.5"` sẽ bị biến dạng thành `"lo25"` và không khớp với từ điển trong `grammar.txt`.

### Quyết định 5: Hàm chấm điểm cây `_tree_score` ưu tiên gắn kết bổ ngữ thấp (Low Modifier Attachment)
- **Quyết định**: Phạt nhẹ các luật dự phòng (`FALLBACK_RULES`: `Modifiers -> Modifier Modifiers`, `NounHead -> Noun NounHead`, mệnh đề khuyết chủ ngữ `Clause -> Predicate`, `WhClause -> WhPredicate`) khi chọn cây cú pháp từ danh sách cây của `nltk.ChartParser`.
- **Lý do (Reasoning)**:
  - Trong các câu hội thoại có ngữ cảnh cục bộ như *"Tài liệu của phần này ở đâu?"* hoặc *"Tài liệu của chương này là gì?"*, từ chỉ định `"này"` về mặt ngữ pháp có thể gắn vào `"phần"` (`"của [phần này]"`) hoặc gắn vào `"tài liệu"` (`"[tài liệu của phần] này"` thông qua `Modifiers -> Modifier Modifiers`).
  - Nhờ phạt nhẹ `Modifiers -> Modifier Modifiers`, parser luôn chọn cây gắn thấp `(Nominal (NounHead (Noun phần)) (Modifiers (Modifier (Demonstrative này))))`, gom `"phần này"` / `"chương này"` thành một nhánh con duy nhất. Điều này giúp **Phần III (Xử lý ngữ cảnh hội thoại cục bộ)** dễ dàng trích xuất đúng cụm danh từ tham chiếu.
  - Đồng thời, `"Môn NLP"` luôn được phân tích qua luật chuyên biệt `NamePrefix + NameExpression` và `"Tuần 5"` trong *"Tuần 5 học những nội dung gì?"* luôn được nhận diện chuẩn xác là `Subject` của câu.
