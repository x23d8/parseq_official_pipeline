FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PARSEQ_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/parseq_demo_yolo

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --chown=app:app demo/requirements-runtime.txt demo/requirements-runtime.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r demo/requirements-runtime.txt

COPY --chown=app:app . .

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/api/health', timeout=4)"

CMD ["python", "-m", "demo"]
