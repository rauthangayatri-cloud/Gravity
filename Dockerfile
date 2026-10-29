FROM python:3.13.7-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GRAVITY_ML_BUNDLE=/app/data/outputs/bundles/image-demo

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m src.generate_data \
    && python -m src.clean \
    && python -m src.build_outputs --unified data/clean/unified.csv --outcomes data/clean/outcomes.csv --bundle-dir "$GRAVITY_ML_BUNDLE" \
    && test -s "$GRAVITY_ML_BUNDLE/manifest.json" \
    && test -s "$GRAVITY_ML_BUNDLE/bundle.joblib" \
    && test -s "$GRAVITY_ML_BUNDLE/students.csv"

EXPOSE 8000
RUN useradd --uid 10001 --create-home gravity \
    && mkdir /state \
    && chown gravity:gravity /state
ENV RUNTIME_STATE_DB=/state/demo.sqlite3
USER gravity
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ready', timeout=4)"
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
