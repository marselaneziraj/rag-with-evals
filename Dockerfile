# Two stages: a builder that installs the package, and a slim runtime.
# The image ships the demo corpus sources and builds the PDFs at image-build
# time, so `docker run` gives you a working UI with no further setup.

FROM python:3.12-slim AS builder
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt streamlit>=1.36 \
 && /opt/venv/bin/pip install --no-deps -e .

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="rag-with-evals" \
      org.opencontainers.image.description="RAG QA with a reproducible evaluation suite" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAGEVALS_LOG=INFO \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src ./src
COPY configs ./configs
COPY data ./data
COPY app ./app
COPY tests ./tests
COPY Makefile pyproject.toml README.md ./

# Build the demo PDFs and the index so the container starts ready to answer.
RUN python -m ragevals.cli samples \
 && python -m ragevals.cli ingest --config configs/default.yaml

RUN useradd --create-home --uid 10001 ragevals \
 && chown -R ragevals:ragevals /app
USER ragevals

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

# Default: the UI. Override to run the evaluation instead, e.g.
#   docker run --rm rag-with-evals python -m ragevals.cli evaluate
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
