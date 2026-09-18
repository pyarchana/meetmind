FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Honour whatever port the host injects. Cloud Run sets PORT, Hugging Face
# routes to app_port from the Space README. Hardcoding 8080 happened to match
# Cloud Run's default and would have broken silently anywhere else.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
