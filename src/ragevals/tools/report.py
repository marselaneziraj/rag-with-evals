"""Assemble RESULTS.md from whatever is in results/.

Kept separate from the runner so that a report can be regenerated from stored
results without re-running a sweep -- the artefacts are the source of truth,
not the console output of the last run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_report(results_dir: Path, out_path: Path) -> Path:
    results_dir = Path(results_dir)
    comparison: list[dict[str, Any]] = _load_json(results_dir / "comparison.json") or []
    injection = _load_json(results_dir / "injection" / "injection_summary.json")

    lines: list[str] = [
        "# Results",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from `{results_dir}/`._",
        "",
    ]

    if comparison:
        lines += ["## Configuration comparison", "",
                  "| Configuration | Faithful. | Ctx prec. | Ctx recall | Correct. | "
                  "Refusal acc. | Over-refusal | Cite valid. | p95 lat. (s) | USD/query |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in comparison:
            lines.append(
                "| {config} | {faithfulness:.3f} | {context_precision:.3f} | "
                "{context_recall:.3f} | {answer_correctness:.3f} | "
                "{refusal_accuracy:.3f} | {over_refusal_rate:.3f} | "
                "{citation_validity:.3f} | {p95_latency_s:.3f} | "
                "{cost_per_query_usd:.5f} |".format(**row))
        lines += ["", "![comparison](results/comparison.png)", ""]

    if injection:
        stages = injection.get("stages", [])
        lines += ["## Prompt injection", "",
                  f"Backend: `{injection.get('generation_backend')}` / "
                  f"`{injection.get('generation_model')}`"
                  + ("  \n**Simulated susceptibility -- see README limitations.**"
                     if injection.get("simulated") else ""),
                  "",
                  "| Mitigation stage | Retrieved | Injection success | Refused/blocked |",
                  "|---|---|---|---|"]
        for s in stages:
            denom = s.get("retrieved") or 1
            lines.append(f"| {s['stage']} | {s['retrieved']}/{s['n']} | "
                         f"{s['successes']}/{denom} = {100 * s['success_rate']:.1f}% | "
                         f"{100 * s.get('refusal_rate', 0.0):.1f}% |")
        lines += ["", "![injection](results/injection/injection.png)", ""]

    runs_dir = results_dir / "runs"
    if runs_dir.exists():
        lines += ["## Runs", ""]
        for summary_path in sorted(runs_dir.glob("*/summary.json")):
            summary = _load_json(summary_path) or {}
            lines.append(f"- `{summary_path.parent.name}`: "
                         f"{summary.get('n_items', 0)} items, engine "
                         f"`{summary.get('metrics_engine')}`, judge "
                         f"`{(summary.get('judge') or {}).get('model')}`, "
                         f"fingerprint `{summary.get('config_fingerprint')}`")
        lines.append("")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
