FROM python:3.12-slim

# 2. Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# 3. Cài đặt thư viện hệ thống cần thiết
# Đã thêm: libmagic1 (check file), poppler-utils (xử lý PDF), libgl1 (xử lý ảnh/opencv)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    libmagic1 \
    file \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Tạo thư mục tessdata (phòng khi thiếu)
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata    
# 4. Thiết lập thư mục làm việc
WORKDIR /code

# Python 3.12 cần setuptools mới nhất
RUN pip install --upgrade pip setuptools wheel

# 5. Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy source code
COPY . .

#Copy file ngôn ngữ tiếng Việt vào container
COPY vie.traineddata /usr/share/tesseract-ocr/5/tessdata/

# 7. Tạo User Non-root
RUN adduser -u 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser /code \
    && chown -R appuser /usr/share/tesseract-ocr/5/tessdata

# --- Xử lý thư mục uploads ---
RUN mkdir -p /code/uploads && chown -R appuser /code/uploads

# 8. Chuyển user
USER appuser

# 9. Mở port
EXPOSE 8000

# 10. Chạy Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "app.main:app"]
