# Course Assistant — Phần I (CO3085)

Nhóm chọn **lựa chọn 2: mở rộng bằng LLM/RAG**. Hiện mã nguồn triển khai Phần I: grammar, sinh câu và phân tích cú pháp. Tài liệu này giúp thành viên khác chạy được chương trình, đọc hiểu code và giải thích các quyết định khi báo cáo.

**Thông tin thành viên:** nhóm bổ sung họ tên, MSSV và phân công thực tế trước khi nộp bài.

| Thành phần | Trạng thái |
|---|---|
| Grammar, sinh câu, parser | Đã triển khai và có kiểm thử |
| Semantic representation, intent/entity | Chưa triển khai |
| Retrieval và LLM/RAG | Chưa triển khai |
| Ngữ cảnh hội thoại | Chưa triển khai; parser đã nhận được một số câu có từ tham chiếu |

Các file `retrieval.txt`, `answer.txt`, `evaluation.txt` đang rỗng là chỗ dành cho phần tiếp theo. Có file chưa có nghĩa là đã hoàn thành chức năng.

## 1. Cài đặt và chạy nhanh

Các lệnh dưới đây chạy từ thư mục repository **`NLP/`**, nơi chứa thư mục `project/`.

Môi trường đã kiểm thử: Linux, **Python 3.14.4** trong venv và **Python 3.14.10** trong Docker, cùng **NLTK 3.10.4**. NLTK bản này yêu cầu Python từ 3.10; các phiên bản Python khác chưa được kiểm thử trong lần triển khai này.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python project/main.py
python -m unittest discover -s project/tests -t project -v
```

Danh sách dependency chính ở `project/requirements.txt`; file tại gốc chỉ chuyển tiếp đến đó. Phần I chỉ cần NLTK ngoài thư viện chuẩn. Không cần chạy `nltk.download()`, vì chương trình dùng grammar của nhóm và không dùng corpus hay tokenizer huấn luyện sẵn.

Nếu chỉ có thư mục `project/` sau khi giải nén bài nộp, đứng trong thư mục đó và chạy:

```bash
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

Đầu ra mặc định: 5.000 câu sinh; 51 dòng được phân tích, gồm 48 dòng có cây và 3 dòng `()`. Đây là số liệu kiểm tra cú pháp, không phải độ chính xác trả lời câu hỏi.

## 2. Đối chiếu yêu cầu và luồng xử lý

| Mục của đề | Mã nguồn | Input | Output |
|---|---|---|---|
| 3.1: tập luật/grammar | `src/grammar/cfg.py` | `src/grammar/grammar.txt` | `output/grammar.txt` |
| 3.2: giải thuật sinh câu | `src/grammar/generator.py` | CFG đã nạp | `output/samples.txt` |
| 3.3: parser | `src/grammar/parser.py` | CFG và `input/sentences.txt` | `output/parse-results.txt` |

Các đường dẫn trong bảng tính từ `project/`. Đề giới hạn tối đa 10.000 câu sinh; chương trình mặc định sinh 5.000, không cần cố đạt 10.000.

```text
grammar nguồn → load_cfg() → CFG ─┬→ export_grammar() → grammar.txt
                                 ├→ SentenceGenerator → samples.txt
                                 └→ CourseParser + sentences.txt → parse-results.txt
```

`main.py` chỉ điều phối ba bước. Nó xác định đường dẫn theo vị trí file, nên lệnh `python project/main.py` không phụ thuộc working directory của các module bên trong.

## 3. Đọc file nào trước?

```text
project/
├── main.py
├── requirements.txt
├── Dockerfile
├── src/grammar/
│   ├── cfg.py              # Nạp và xuất grammar.
│   ├── grammar.txt         # Nguồn luật duy nhất.
│   ├── generator.py        # Mở rộng luật có giới hạn để sinh câu.
│   ├── parser.py           # Ghép terminal, tìm cây và chọn cây đại diện.
│   └── __init__.py         # Các tên được import từ package.
├── input/sentences.txt
├── output/
├── data/                   # Knowledge base mẫu cho các phần tiếp theo.
└── tests/
    ├── test_parser.py
    ├── test_generator.py
    └── test_pipeline.py
```

Thứ tự đọc đề xuất: **README → main.py → grammar.txt → cfg.py → generator.py → parser.py → tests**. Trong mỗi module, đọc class/docstring và các hàm public trước, sau đó mới đọc hàm phụ.

`data/` chứa bộ dữ liệu môn học mẫu, bao gồm một số thông tin lịch/deadline giả lập; xem `data/00_README.txt`. Phần I hiện chưa truy vấn các file này để trả lời câu hỏi.

## 4. Các khái niệm cần hiểu

| Khái niệm | Ý nghĩa | Ví dụ |
|---|---|---|
| CFG | Văn phạm phi ngữ cảnh: tập luật mở rộng các ký hiệu | `Subject -> NounPhrase` |
| Nonterminal | Nhãn cấu trúc còn có thể mở rộng | `NounPhrase`, `CourseName` |
| Terminal | Chuỗi xuất hiện ở lá của cây; nằm trong dấu nháy ở grammar | `"nlp"`, `"bài tập lớn"` |
| Production | Một luật gồm vế trái và vế phải | `CourseName -> "nlp"` |
| Tokenization | Ghép các âm tiết đầu vào thành terminal mà grammar biết | `bài tập lớn` thành một token |
| Backtracking | Thử một lựa chọn, quay lại để thử lựa chọn khác | Thử cụm dài rồi thử cụm ngắn |
| Heuristic | Tiêu chí thực dụng để chọn trong nhiều kết quả | Ưu tiên cây dùng ít luật dự phòng |
| Seed | Giá trị khởi tạo bộ lấy mẫu để chạy lại được cùng chuỗi lựa chọn | `seed=42` |

Một âm tiết tách bằng khoảng trắng chưa chắc là một termi
| `max_samples` | 5.000 | Đủ nhiều để thử grammar; chấp nhận 0 đến 10.000 |
| `seed` | 42 | Chạy lại có thể tái lập trong cùng môi trường |
| `max_depth` | 14 | Chặn mở rộng đệ quy quá sâu |
| `max_terminal_phrases` | 12 | Chặn câu quá dài theo số terminal |
| `max_attempts` | 100.000 | Không lặp mãi knal. Nếu chỉ dùng `.split()`, `"bài tập lớn"` sẽ thành ba phần và không khớp terminal `"bài tập lớn"` trong grammar.

| `max_samples` | 5.000 | Đủ nhiều để thử grammar; chấp nhận 0 đến 10.000 |
| `seed` | 42 | Chạy lại có thể tái lập trong cùng môi trường |
| `max_depth` | 14 | Chặn mở rộng đệ quy quá sâu |
| `max_terminal_phrases` | 12 | Chặn câu quá dài theo số terminal |
| `max_attempts` | 100.000 | Không lặp mãi k
## 5. Grammar được tổ chức thế nào?

File nguồn có năm nhóm được chú thích:

1. Cấu trúc câu: trần thuật, câu hỏi Wh, Yes/No và yêu cầu.
2. Danh ngữ: tên môn, chủ đề, chỉ số tuần/chương, lượng từ và bổ ngữ.
3. Động ngữ: động từ và các thành phần theo sau.
4. Từ chức năng: giới từ, từ để hỏi, từ chỉ định, dấu hiệu yêu cầu.
5. Từ vựng của miền môn học: môn, chủ đề, bài tập, tài liệu và công cụ.

Ví dụ `NounHead -> NamePrefix NameExpression` giúp tạo cấu trúc cho `môn NLP`. `IndexedNoun Index` giúp tạo `tuần 5`. `ObjectVerb NounPhrase` biểu diễn động từ có tân ngữ, như `học CFG`.

Grammar có luật linh hoạt như `NounHead -> Noun NounHead` để nhận `deadline bài tập lớn`. Luật này cũng có thể tạo nhiều cách phân tích hoặc các chuỗi thiếu tự nhiên. Parser dùng heuristic để chọn một cây, không thể tự suy ra ý nghĩa đúng trong mọi trường hợp.

| `max_samples` | 5.000 | Đủ nhiều để thử grammar; chấp nhận 0 đến 10.000 |
| `seed` | 42 | Chạy lại có thể tái lập trong cùng môi trường |
| `max_depth` | 14 | Chặn mở rộng đệ quy quá sâu |
| `max_terminal_phrases` | 12 | Chặn câu quá dài theo số terminal |
| `max_attempts` | 100.000 | Không lặp mãi k
**Bốn nhóm cấu trúc câu khác với các nhóm intent** như hỏi lịch, hỏi deadline, hỏi tài liệu. Phần I nhận dạng cấu trúc; Phần II mới cần ánh xạ cấu trúc sang yêu cầu và thực thể.

Chỉ chỉnh `src/grammar/grammar.txt`. `output/grammar.txt` là bản sao xuất ra để kiểm tra/nộp bài.

## 6. Generator: cách sinh một câu

Generator lấy production trực tiếp từ CFG. Nó đi qua lớp bọc đơn `Sentences -> Sentence`, rồi lần lượt thử các nhánh thật của `Sentence`: `Clause`, `WhClause`, `YesNoQuestion`, `Request`.

Với mỗi lần thử:

1. Chọn nhánh tiếp theo.
2. Gặp terminal thì thêm cụm từ vào câu.
3. Gặp nonterminal thì chọn một production bằng bộ lấy mẫu có seed và mở rộng vế phải.
4. Vượt giới hạn hoặc gặp ký hiệu không có luật thì hủy cả lần thử.
5. Hoàn thành thì ghép các terminal; chỉ nhận câu chưa trùng.

```text
lặp đến đủ số câu hoặc hết lượt thử:
    chọn nhánh theo thứ tự luân phiên
    thử mở rộng các ký hiệu theo grammar
    nếu hoàn chỉnh và câu chưa có:
        thêm vào danh sách
```

Cùng một production được dùng lại ở nhiều vị trí. Ví dụ, chủ ngữ và tân ngữ của `Tôi học NLP` đều có thể cần luật `NounPhrase -> SimpleNounPhrase`.

### Cấu hình

Mặc định nằm ở `generator.py`; có thể truyền tham số khi tạo `SentenceGenerator` hoặc gọi `generate_to_file`.

| Tham số | Mặc định | Lý do |
|---|---:|---|
| `max_samples` | 5.000 | Đủ nhiều để thử grammar; chấp nhận 0 đến 10.000 |
| `seed` | 42 | Chạy lại có thể tái lập trong cùng môi trường |
| `max_depth` | 14 | Chặn mở rộng đệ quy quá sâu |
| `max_terminal_phrases` | 12 | Chặn câu quá dài theo số terminal |
| `max_attempts` | 100.000 | Không lặp mãi khi khó tìm thêm câu khác nhau |

Vế trái của nhánh sau khi bỏ lớp bọc đơn có độ sâu 0; các con có độ sâu 1. Nonterminal chỉ mở rộng khi độ sâu nhỏ hơn `max_depth`. Terminal ở độ sâu 14 vẫn được nhận. **12 terminal không có nghĩa là 12 từ/âm tiết**: `"xử lý ngôn ngữ tự nhiên"` cũng chỉ tính một terminal.

`random.Random(seed)` là bộ lấy mẫu cục bộ. Mỗi lần gọi `generate()` tạo lại nó từ seed; giữ thứ tự grammar/cấu hình/phiên bản môi trường sẽ giữ thứ tự output. Đổi thứ tự luật hoặc từ vựng có thể đổi kết quả.

Danh sách giữ thứ tự ghi file; set chỉ dùng phát hiện trùng. Hết số lần thử mà chưa đủ thì trả số câu thực tế và phát cảnh báo, không thêm câu ngoài grammar cho đủ số lượng.

Luân phiên số lượt thử giữa các nhánh **không bảo đảm số câu được nhận của mỗi nhánh bằng nhau**, cũng không bảo đảm cân bằng các intent.

## 7. Parser: từ câu đầu vào đến cây

```text
câu gốc
→ preprocess(): chuẩn hóa chữ và dấu câu
→ iter_tokenizations(): thử các cách ghép terminal
→ ChartParser: tìm cây theo CFG
→ _tree_score(): so điểm giữa mọi cách tách
→ trả một nltk.Tree hoặc None
```

### Ghép terminal và quay lui

`iter_tokenizations()` ưu tiên cụm dài nhưng vẫn thử cụm ngắn. Ví dụ, `tài liệu tham khảo` có thể là một terminal, hoặc `tài liệu` và động từ `tham khảo`.

Hàm dùng `yield` để trả từng cách tách khi cần. Bên trong, thao tác thêm token → đệ quy → bỏ token có nghĩa là thử một lựa chọn rồi quay lại thử lựa chọn tiếp theo. Bản hiện tại không giới hạn cứng ở bốn cách tách.

### ChartParser và chọn cây

ChartParser lưu các kết quả phân tích trung gian trong một chart để có thể tái sử dụng, thay vì phân tích lại từ đầu mọi cụm con. Grammar có thể cho nhiều cây, nên nhóm bổ sung tiêu chí:

```text
điểm cây = (số lần dùng luật dự phòng, tổng số production)
```

Python so tuple từ trái sang phải: ưu tiên ít luật dự phòng trước; nếu bằng nhau mới xét ít production. Các luật dự phòng và lý do có comment ngay tại `FALLBACK_RULES`. Khi hai cây cùng điểm, giữ cây gặp trước.

Parser so cây từ **mọi cách tách đã tìm được**, không dừng ở cách tách đầu tiên có cây. Đây là chọn cây theo heuristic, không bảo đảm cây đúng ngữ nghĩa. Duyệt cây lần lượt cũng không loại bỏ chi phí/giới hạn trích xuất cây bên trong NLTK.

### Quy ước file và lỗi

- Mỗi dòng input tương ứng một dòng output; cây được viết trên một dòng.
- Câu không có cây trong grammar, kể cả dòng trống, cho kết quả `()`.
- `()` không chứng minh câu sai tiếng Việt hoặc nằm ngoài miền môn học: có thể grammar chưa hỗ trợ cách diễn đạt đó.
- Thiếu file, grammar hỏng hoặc lỗi giới hạn thư viện được báo riêng. Lỗi phân tích file có thông tin dòng; không bị đổi thành `()`.
- `parse_file()` chỉ ghi sau khi phân tích xong toàn bộ input, nên lỗi đọc/phân tích không ghi đè file parse output cũ bằng kết quả rỗng/dang dở.
- Ba bước của toàn pipeline ghi các file riêng; nếu bước sau lỗi, một số output bước trước có thể đã được cập nhật. Chỉ dùng kết quả khi toàn lần chạy đã thành công.

## 8. Ví dụ để trình bày luồng xử lý

Với câu **“Môn NLP có bao nhiêu tín chỉ?”**:

1. Chuẩn hóa thành `môn nlp có bao nhiêu tín chỉ`.
2. Ghép được `["môn", "nlp", "có", "bao nhiêu", "tín chỉ"]`.
3. ChartParser nhận câu hỏi Wh, có chủ ngữ `môn NLP`, động từ `có`, phần hỏi số lượng `bao nhiêu tín chỉ`.
4. Chọn cây đại diện và xuất vào dòng tương ứng trong `parse-results.txt`.

Cấu trúc dưới đây đã rút gọn để dễ đọc, không phải toàn bộ output:

```text
WhClause
├── Subject
│   └── môn NLP
└── WhPredicate
    ├── ObjectVerb: có
    └── WhNounPhrase
        ├── WhQuantity: bao nhiêu
        └── UnitNoun: tín chỉ
```

**Phần I chưa trả lời số tín chỉ là bao nhiêu.** Việc tìm thông tin đó trong knowledge base thuộc phần tiếp theo.

Các câu hồi quy quan trọng:

| Câu | Điều được kiểm tra |
|---|---|
| `Tài liệu tham khảo môn học này và môn học đó không?` | Cách tách tạo cây nằm sau bốn ứng viên đầu vẫn được xét |
| `Tài liệu tham khảo môn học.` | Cây của cách tách sau có điểm tốt hơn được chọn |
| `Tài liệu của phần này ở đâu?` | `phần này` được gom trong một danh ngữ |
| `NLP tín chỉ bao nhiêu có?` | Từ đều có trong lexicon vẫn bị từ chối khi không có cấu trúc phù hợp |

Một vài câu hồi quy nhằm kiểm tra cơ chế cú pháp; không dùng chúng để chứng minh hệ thống hiểu đúng ý nghĩa thực tế.

## 9. Các quyết định thiết kế và lý do

| Quyết định | Lý do | Giới hạn/đánh đổi |
|---|---|---|
| Dùng NLTK CFG/ChartParser/Tree | Có cấu trúc chuẩn, ít code hạ tầng; cây dùng tiếp được ở Phần II | Nhóm cần hiểu grammar và cách parser tìm cây, không chỉ biết gọi thư viện |
| Một grammar nguồn | Parser và generator có cùng định nghĩa câu hợp lệ | Grammar quá rộng ảnh hưởng cả nhập nhằng lẫn chất lượng câu sinh |
| Terminal nhiều âm tiết | Khớp các cụm tiếng Việt trong grammar hiện tại | Cần ghép từ trước khi đưa vào ChartParser |
| Thử nhiều cách tách | Cụm dài nhất chưa chắc tạo cây hợp lý nhất | Có thể tốn thời gian khi nhiều từ chồng lấp |
| Giữ dấu bên trong mã | Bảo toàn `L.O.2.5`, `LLM/RAG` | Không phải bộ chuẩn hóa mọi kiểu dấu câu |
| Chọn cây bằng điểm đơn giản | Dễ giải thích, hỗ trợ chọn cấu trúc tham chiếu | Heuristic không thay thế diễn giải ngữ nghĩa |
| Sinh bằng mở rộng đệ quy có giới hạn | Bám grammar, cho phép dùng lại luật; dễ theo dõi hơn template/tổ hợp/GCD | Lấy mẫu không liệt kê hết câu hợp lệ, vẫn có câu thiếu tự nhiên |
| Seed và thứ tự luật ổn định | Tái lập lỗi, kiểm thử và demo | Không cam kết output giống nhau giữa mọi phiên bản/mọi grammar |
| Giới hạn sâu/độ dài/lượt thử | Luôn dừng trong phạm vi tìm kiếm đã đặt | Có thể chưa sinh đủ số câu mong muốn |
| Mặc định 5.000 câu | Nằm trong giới hạn tối đa 10.000 của đề | Không tự thể hiện độ phủ toàn bộ intent |
| `()` chỉ dùng khi không có cây | Phân biệt kết quả cú pháp với lỗi vận hành | Phải đọc thông báo lỗi khi quá trình chạy không thành công |
| Dependency tối thiểu | Chỉ cài những gì code hiện dùng, hỗ trợ đóng gói sạch | Khi làm LLM/RAG cần bổ sung thư viện thực sự được sử dụng |

## 10. Kiểm thử và kết quả

Bộ test gồm **20 kiểm thử**, có thể chạy lại bằng lệnh ở mục 1. Test dùng thư mục tạm; không ghi đè output thật của nhóm.

- Parser: lỗi cách tách thứ 5, so cây giữa các cách tách, dấu trong mã, tham chiếu, câu sai trật tự và input trống.
- File: giữ số dòng, thiếu input không ghi đè output, lỗi thư viện có thông tin dòng.
- Generator: dùng lại production, seed, hết số câu khác nhau, grammar đệ quy, các giới hạn và output 0 câu.
- Tích hợp: chạy ba bước, kiểm tra grammar xuất, 51 kết quả parse, 5.000 câu khác nhau và parse lại toàn bộ 5.000 câu bằng cùng grammar.

Kết quả kiểm tra ngày **09/11/2026**:

| Kiểm tra | Kết quả |
|---|---|
| Bộ test tự động | 20/20 đạt trên môi trường phát triển, venv cài mới và Docker |
| Grammar xuất | Khớp nguồn |
| Câu sinh mặc định | 5.000 câu khác nhau sau 12.392 lần thử |
| Parse lại các câu sinh | 5.000/5.000 có cây |
| Input mẫu | 51 dòng: 48 có cây, 3 `()` |
| Cài đặt trong venv sạch | Cài từ `project/requirements.txt`, `pip check` đạt, 20/20 test đạt; file chuyển tiếp tại gốc cũng hoạt động |
| Docker | Build thành công; pipeline ghi output qua volume; 20/20 test đạt trong container |

Đây là kiểm tra tính đúng của pipeline cú pháp. Input mẫu có nhiều câu đã được dùng khi thiết kế grammar; 48 câu có cây không phải đánh giá độc lập về khả năng hiểu ngôn ngữ hay trả lời câu hỏi.

**Kiểm tra thủ công chất lượng câu sinh:** mẫu output vẫn có câu như `ngày mai bao nhiêu tín chỉ` hoặc `hãy là nó về tra cứu chúng tôi`. Chúng dẫn xuất được từ CFG nhưng thiếu tự nhiên về nghĩa. Phiên bản này giữ rõ giới hạn đó; chưa có bộ lọc ngữ nghĩa. Khi cải thiện, nên bổ sung ràng buộc có lý do và test cụ thể, tránh sửa output bằng tay.

## 11. Chạy bằng Docker

Từ `NLP/`:

```bash
docker build -t course-assistant-part1 ./project
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/project/output:/app/output" course-assistant-part1
docker run --rm course-assistant-part1 python -m unittest discover -s tests -v
```

Nếu đang đứng trong `project/`, thay build context `./project` bằng `.`, và đường dẫn volume bằng `"$(pwd)/output:/app/output"`.

Docker dùng Python 3.14 (bản dockerfile mẫu thầy để 3.8 -- nghiên cứu lại phần này); build context phải là `project/` để `requirements.txt` và `main.py` có đường dẫn đúng. Volume giúp lấy output về máy; nếu không gắn volume thì output chỉ nằm trong container bị xóa khi chạy với `--rm`. Trên Linux, `--user` dùng UID/GID của người đang chạy để các file output thuộc đúng người dùng đó.

## 12. Thêm từ vựng hoặc cấu trúc mới

Ví dụ cần hỗ trợ một chủ đề mới:

1. Thêm terminal chữ thường vào `TopicName` trong grammar nguồn.
2. Nếu câu dùng cấu trúc mới, xác định nonterminal và production cần bổ sung.
3. Thêm câu kiểm tra có kỳ vọng rõ ràng; với luật dễ nhập nhằng, thêm cả trường hợp dễ hiểu sai.
4. Chạy test, chạy `main.py` để xuất lại grammar và samples, rồi xem các thay đổi output.
5. Cập nhật README nếu thay đổi quy tắc hoặc giới hạn của hệ thống.

Thêm một từ không bảo đảm mọi câu chứa từ đó đều parse được. Câu còn phải phù hợp cấu trúc grammar.

## 13. Lỗi thường gặp và hạn chế

| Hiện tượng | Cách kiểm tra |
|---|---|
| Không import được NLTK | Đã kích hoạt đúng venv và cài đúng requirements chưa? |
| Không tìm thấy input/grammar | Kiểm tra cấu trúc `project/` và đường dẫn trong lỗi |
| Câu cho `()` | Kiểm tra từ/cụm từ có trong grammar, sau đó kiểm tra cấu trúc |
| Sinh chưa đủ số câu | Đọc cảnh báo; xem giới hạn và số câu khác nhau mà grammar có thể tạo |
| Parse chậm hoặc lỗi trích xuất cây | Grammar/câu quá nhập nhằng; xem dòng được báo, giảm câu thử hoặc điều chỉnh grammar có kiểm thử |
| Docker không thấy `requirements.txt` | Dùng đúng build context trong mục 11 |
| Không thấy output Docker trên máy | Kiểm tra volume và quyền ghi thư mục |

Parser hiện phù hợp các câu ngắn, trong từ vựng hữu hạn. Xét hết cách tách/cây không bảo đảm xử lý nhanh câu dài. Chưa có timeout riêng của nhóm; giới hạn nội bộ NLTK có thể phát lỗi và được báo ra.

## 14. Giao diện để làm tiếp LLM/RAG

`CourseParser.parse()` trả `nltk.Tree` hoặc `None`. Phần sau có thể duyệt cây để lấy nhãn và thực thể, rồi xây biểu diễn ngữ nghĩa/truy vấn retrieval.

Ví dụ Python dưới đây chạy từ `project/`:

```python
from src.grammar import CourseParser, load_cfg

parser = CourseParser(load_cfg("src/grammar/grammar.txt"))
tree = parser.parse("Môn NLP có bao nhiêu tín chỉ?")
if tree is not None:
    for node in tree.subtrees():
        if node.label() in {"CourseName", "WhQuantity", "UnitNoun"}:
            print(node.label(), node.leaves())
```

Kết quả gồm `CourseName ['nlp']`, `WhQuantity ['bao nhiêu']`, `UnitNoun ['tín chỉ']`. Việc ánh xạ chúng thành intent/entity rồi truy xuất context và gọi LLM là phần cần triển khai tiếp. LLM/RAG vẫn phải thể hiện vai trò của grammar/parser theo yêu cầu chung của đề.

## 15. Nguồn và hỗ trợ

- Đề BTL CO3085 — Course Assistant: Phần I, lựa chọn 2 và yêu cầu pipeline chung.
- Dữ liệu mẫu đi kèm trong `data/`; nguồn/phạm vi mô tả tại `data/00_README.txt`.
- NLTK 3.10.4: `nltk.CFG`, `nltk.ChartParser`, `nltk.Tree` và docstring/mã nguồn của các thành phần này.
- Python standard library: `random`, `pathlib`, `unittest`.