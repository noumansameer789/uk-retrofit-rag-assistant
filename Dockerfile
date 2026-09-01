FROM python:3.12-slim
WORKDIR /app
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "retrofit_rag.app"]
