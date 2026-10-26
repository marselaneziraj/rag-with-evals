# rag-with-evals
#
# `make all` is the whole pipeline on a clean checkout: build the demo corpus,
# index it, run the tests, evaluate the gold set, sweep the configurations, run
# the injection suite and regenerate RESULTS.md. No API key, no model download.

PY ?= python
CONFIG ?= configs/default.yaml
SWEEP ?= configs/experiments/sweep.yaml
GOLD ?= data/gold/gold_set.yaml
PORT ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install install-dev install-all samples ingest ask evaluate experiments \
        injection report test lint clean clean-cache app docker-build docker-run \
        propose gold-stats all ci

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## install the core package
	$(PY) -m pip install -e .

install-dev:  ## install with the dev extras (pytest, ruff)
	$(PY) -m pip install -e ".[dev]"

install-all:  ## install every optional backend (embeddings, chroma, faiss, ui)
	$(PY) -m pip install -e ".[all]"

samples:  ## build the demo PDFs from data/samples/*.md
	$(PY) -m ragevals.cli samples

ingest: samples  ## extract, chunk, embed and index the corpus
	$(PY) -m ragevals.cli ingest --config $(CONFIG)

ask:  ## ask one question: make ask Q="your question"
	$(PY) -m ragevals.cli ask --config $(CONFIG) "$(Q)"

propose:  ## draft candidate gold questions for manual curation
	$(PY) -m ragevals.cli propose --config $(CONFIG) -n 80 \
	  --out data/gold/candidates.yaml

gold-stats:  ## validate and summarise the gold set
	$(PY) -m ragevals.cli gold-stats --gold $(GOLD)

evaluate: ingest  ## run the gold set through the pipeline
	$(PY) -m ragevals.cli evaluate --config $(CONFIG) --gold $(GOLD)

experiments: ingest  ## run the sweep and emit the comparison table + chart
	$(PY) -m ragevals.cli experiment --sweep $(SWEEP) --gold $(GOLD) --out results

injection: ingest  ## run the prompt-injection suite
	$(PY) -m ragevals.cli injection --config $(CONFIG)

report:  ## assemble RESULTS.md from results/
	$(PY) -m ragevals.cli report --results results --out RESULTS.md

test:  ## run the test suite
	$(PY) -m pytest tests -q

lint:  ## ruff check
	$(PY) -m ruff check src tests app scripts

clean-cache:  ## drop retrieval/judge caches and indexes
	$(PY) -c "import shutil; shutil.rmtree('artifacts', ignore_errors=True)"

clean: clean-cache  ## also drop results and generated PDFs
	$(PY) -c "import shutil; shutil.rmtree('results', ignore_errors=True)"
	$(PY) -c "import shutil; shutil.rmtree('data/corpus', ignore_errors=True)"

app:  ## launch the Streamlit UI
	streamlit run app/streamlit_app.py --server.port $(PORT)

docker-build:  ## build the image
	docker build -t rag-with-evals .

docker-run:  ## run the UI in docker on $(PORT)
	docker run --rm -p $(PORT):8501 rag-with-evals

ci: test evaluate experiments injection report  ## what the CI workflow runs

all: test evaluate experiments injection report  ## everything, end to end
	@echo ""
	@echo "done. see RESULTS.md, results/comparison.png and results/injection/injection.png"
