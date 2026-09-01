FROM python:3.14-slim
WORKDIR /app
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && addgroup --system app \
    && adduser --system --ingroup app app
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
COPY --chown=app:app scripts ./scripts
ENV PYTHONPATH=/app/src
EXPOSE 8000
USER app
CMD ["uvicorn", "retrofit_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
