FROM python:3.12-slim

# Build context là project/: requirements.txt và main.py ở cùng thư mục.
# Hỗ trợ cả mount /app/output lẫn /nlp/input, /nlp/output (qua default_dirs trong main.py).
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "main.py"]
