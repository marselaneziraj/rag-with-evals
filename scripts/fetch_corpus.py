#!/usr/bin/env python
"""Fetch the real 200-500 page corpus (EU AI Act + GDPR) from EUR-Lex.

The repository ships a small generated demo corpus so that everything runs
offline. This script replaces it with the real thing.

    python scripts/fetch_corpus.py --out data/corpus

Why the URLs are in a table rather than hard-coded into the pipeline: official
documents move, and a corpus you cannot re-fetch is a corpus you cannot
reproduce. Each entry records the CELEX identifier, which is stable even when
the URL is not. After fetching, re-run:

    python -m ragevals.cli ingest --config configs/default.yaml

and re-curate the gold set -- page numbers in the real documents will not match
the demo corpus, so the shipped gold annotations will no longer apply.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCES = [
    # (filename, CELEX id, description, approximate pages)
    ("eu_ai_act_2024_1689.pdf", "32024R1689",
     "Regulation (EU) 2024/1689 - Artificial Intelligence Act", 144),
    ("gdpr_2016_679.pdf", "32016R0679",
     "Regulation (EU) 2016/679 - General Data Protection Regulation", 88),
    ("data_act_2023_2854.pdf", "32023R2854",
     "Regulation (EU) 2023/2854 - Data Act", 71),
    ("dsa_2022_2065.pdf", "32022R2065",
     "Regulation (EU) 2022/2065 - Digital Services Act", 102),
]

URL_TEMPLATE = ("https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/"
                "?uri=CELEX:{celex}")


def fetch(celex: str, destination: Path, timeout: float = 120.0) -> bool:
    url = URL_TEMPLATE.format(celex=celex)
    request = urllib.request.Request(url, headers={"Accept": "application/pdf"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  FAILED {celex}: {exc}", file=sys.stderr)
        return False
    if not payload.startswith(b"%PDF"):
        print(f"  FAILED {celex}: the response was not a PDF "
              f"({len(payload)} bytes). EUR-Lex may be rate-limiting; retry later "
              f"or download {url} manually.", file=sys.stderr)
        return False
    destination.write_bytes(payload)
    print(f"  wrote {destination} ({len(payload) / 1_048_576:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/corpus", type=Path)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these filenames")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    failures = 0
    for filename, celex, description, pages in SOURCES:
        if args.only and filename not in args.only:
            continue
        destination = args.out / filename
        print(f"{description} [CELEX {celex}, ~{pages} pages]")
        if destination.exists() and not args.force:
            print(f"  already present: {destination} (use --force to refetch)")
            total_pages += pages
            continue
        if fetch(celex, destination):
            total_pages += pages
        else:
            failures += 1

    print(f"\ncorpus in {args.out}: approximately {total_pages} pages")
    if failures:
        print(f"{failures} document(s) failed. The demo corpus in data/samples/ "
              f"still works: run `python -m ragevals.cli samples`.", file=sys.stderr)
    print("\nNext: python -m ragevals.cli ingest --config configs/default.yaml")
    print("Then re-curate data/gold/gold_set.yaml -- the shipped page annotations "
          "refer to the demo corpus, not to these documents.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
