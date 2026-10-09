FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DICE_DETECTOR_MODEL_DIR=/models

WORKDIR /app

COPY pyproject.toml README.md CHANGELOG.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /models \
    && chown app:app /models

USER app

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=2)"]

CMD ["uvicorn", "dice_detector.api:app", "--host", "0.0.0.0", "--port", "18080"]
