#!/usr/bin/env python
"""Build the demo PDFs from data/samples/*.md (same as `make samples`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragevals.samples import build_sample_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/samples")
    parser.add_argument("--out", default="data/corpus")
    args = parser.parse_args()
    paths = build_sample_corpus(args.samples, args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
