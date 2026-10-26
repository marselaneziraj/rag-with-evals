"""Streamlit UI: the answer, the chunks that produced it, and the score.

    streamlit run app/streamlit_app.py

The UI is deliberately an inspection tool rather than a chat product. Three
things are always on screen together -- the answer, the exact passages the
model was shown, and the per-answer faithfulness -- because the point of this
repo is that you cannot judge a RAG answer without seeing its evidence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from ragevals.citations import parse_citations  # noqa: E402
from ragevals.config import load_config  # noqa: E402
from ragevals.evaluation.metrics import faithfulness  # noqa: E402
from ragevals.generation.prompts import REFUSAL_MARKER  # noqa: E402
from ragevals.ingest.pipeline import build_chunks  # noqa: E402
from ragevals.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="rag-with-evals", page_icon="*", layout="wide")


@st.cache_resource(show_spinner="Building the index...")
def load_pipeline(config_path: str, overrides: tuple):
    cfg = load_config(config_path)
    patch: dict = {}
    for key, value in overrides:
        node = patch
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if patch:
        cfg = cfg.merged(patch)
    chunks, report = build_chunks(cfg)
    return RAGPipeline.from_config(cfg, chunks=chunks), report, cfg


def main() -> None:
    st.title("rag-with-evals")
    st.caption("Retrieval-augmented QA over EU AI Act / GDPR documents, with the "
               "evidence and the score shown next to every answer.")

    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Config file", value="configs/default.yaml")
        mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"], index=0)
        top_k = st.slider("Chunks in context (top_k)", 1, 12, 5)
        backend = st.selectbox("Generator", ["offline", "ollama", "openai"], index=0,
                               help="offline needs no key; ollama needs a local server; "
                                    "openai needs OPENAI_API_KEY")
        sanitize = st.checkbox("Sanitise retrieved text", value=True)
        validate = st.checkbox("Validate the answer before showing it", value=True)
        st.divider()
        st.caption("Changing any control rebuilds the pipeline and the index.")

    overrides = (("retrieval.mode", mode), ("retrieval.top_k", top_k),
                 ("generation.backend", backend),
                 ("safety.sanitize_context", sanitize),
                 ("safety.validate_output", validate))

    try:
        pipeline, report, cfg = load_pipeline(config_path, overrides)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.info("Run `make samples && make ingest` first.")
        return

    with st.sidebar:
        st.metric("Chunks indexed", report.chunks)
        st.metric("Pages", report.pages)
        st.code(f"fingerprint {cfg.fingerprint()}", language=None)

    question = st.text_input(
        "Question",
        value="Within what time must a personal data breach be notified?",
        placeholder="Ask something answerable from the corpus -- or something that is not.")
    ask = st.button("Ask", type="primary")

    if not (ask and question.strip()):
        st.info("Try an unanswerable question too: the system should refuse rather "
                "than improvise.")
        return

    started = time.perf_counter()
    result = pipeline.answer(question)
    elapsed = time.perf_counter() - started

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Answer")
        if result.refused:
            st.warning(f"**{REFUSAL_MARKER}** - the retrieved passages do not support "
                       "an answer, so the system declined to produce one.")
            if result.validation_reasons:
                st.caption("Validator: " + ", ".join(result.validation_reasons))
        else:
            st.markdown(result.answer)
            cites = parse_citations(result.answer)
            if cites:
                st.caption("Citations: " + ", ".join(str(c) for c in cites))

        score = faithfulness(result.answer, result.contexts)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness", f"{score.value:.2f}",
                  help="Share of the answer's claims that the retrieved context supports.")
        c2.metric("Valid citations", f"{result.valid_citations}",
                  delta=f"-{result.invalid_citations} invalid"
                  if result.invalid_citations else None,
                  delta_color="inverse")
        c3.metric("Latency", f"{elapsed:.2f}s")
        c4.metric("Cost", f"${result.cost_usd:.5f}")

        if score.detail.get("per_claim"):
            with st.expander("Per-claim support"):
                for claim in score.detail["per_claim"]:
                    bar = "#" * int(round(10 * float(claim["support"])))
                    st.write(f"`{float(claim['support']):.2f}` {bar:<10} {claim['claim']}")

        if result.sanitizer_flags:
            st.error("Input filter triggered on the retrieved text: "
                     + ", ".join(result.sanitizer_flags)
                     + ". A retrieved document contained something shaped like an "
                       "instruction; it was neutralised before the prompt was built.")

    with right:
        st.subheader(f"Retrieved context ({len(result.hits)} chunks)")
        for hit in result.hits:
            label = (f"#{hit.rank}  {hit.chunk.source} p.{hit.chunk.page}  "
                     f"({hit.source_ranker}, {hit.score:.4f})")
            with st.expander(label, expanded=hit.rank == 1):
                if hit.chunk.section:
                    st.caption(hit.chunk.section)
                st.write(hit.chunk.text)

    with st.expander("Raw result (JSON)"):
        st.json(result.to_dict())


if __name__ == "__main__":
    main()
