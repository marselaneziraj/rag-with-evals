"""Shared fixtures.

The whole suite runs offline, with no model download and no network: the
hashing embedder, the numpy store and the offline generator are all
deterministic, which is what makes these tests assertions rather than
observations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragevals.config import Config  # noqa: E402
from ragevals.ingest.chunking import Chunk  # noqa: E402
from ragevals.ingest.pdf import Block, PageText  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config()
    config.embedding.backend = "hashing"
    config.store.backend = "numpy"
    config.generation.backend = "offline"
    config.paths.index_dir = str(tmp_path / "index")
    config.paths.cache_dir = str(tmp_path / "cache")
    config.paths.results_dir = str(tmp_path / "results")
    config.retrieval.cache = False
    return config


@pytest.fixture
def pages() -> list[PageText]:
    """A two-page document with headings, used across the chunking tests."""
    return [
        PageText(source="demo.pdf", page=1, blocks=[
            Block(text="Article 33 - Breach notification", page=1, bbox=(0, 0, 0, 0),
                  size=14.0, bold=True, is_heading=True),
            Block(text=("The controller shall notify the supervisory authority without "
                        "undue delay and, where feasible, not later than 72 hours after "
                        "having become aware of the personal data breach. Where the "
                        "notification is not made within 72 hours it shall be accompanied "
                        "by reasons for the delay."), page=1, bbox=(0, 0, 0, 0), size=10.0),
        ]),
        PageText(source="demo.pdf", page=2, blocks=[
            Block(text="Article 35 - Impact assessment", page=2, bbox=(0, 0, 0, 0),
                  size=14.0, bold=True, is_heading=True),
            Block(text=("Where processing is likely to result in a high risk to the rights "
                        "and freedoms of natural persons the controller shall carry out an "
                        "assessment of the impact of the envisaged processing operations."),
                  page=2, bbox=(0, 0, 0, 0), size=10.0),
        ]),
    ]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", text=("The controller must notify the supervisory authority of a "
                             "personal data breach within 72 hours of becoming aware of it."),
              source="gdpr.pdf", page=3, page_end=3, section="Article 33"),
        Chunk(id="c2", text=("A data protection impact assessment is required where "
                             "processing is likely to result in a high risk to the rights "
                             "and freedoms of natural persons."),
              source="gdpr.pdf", page=4, page_end=4, section="Article 35"),
        Chunk(id="c3", text=("Administrative fines of up to 20 000 000 EUR or 4 per cent of "
                             "total worldwide annual turnover may be imposed."),
              source="gdpr.pdf", page=9, page_end=9, section="Article 83"),
    ]
