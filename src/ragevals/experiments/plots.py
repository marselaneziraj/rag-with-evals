"""Comparison figures.

Grouped bars for the quality metrics (all on [0, 1], so they share an axis)
plus two separate panels for latency and cost, which do not. Putting cost on
the same axis as a proportion is the standard way to make a chart that looks
informative and says nothing.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

QUALITY_METRICS = [
    ("faithfulness", "Faithfulness"),
    ("context_precision", "Context precision"),
    ("context_recall", "Context recall"),
    ("answer_correctness", "Answer correctness"),
    ("refusal_accuracy", "Refusal accuracy"),
]

PALETTE = ["#2f4b7c", "#f95d6a", "#ffa600", "#665191", "#2a9d8f", "#a05195",
           "#457b9d", "#e76f51"]


def plot_comparison(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(r["config"]) for r in rows]
    n = len(names)
    x = np.arange(len(QUALITY_METRICS))
    width = min(0.82 / max(n, 1), 0.3)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.0),
                             gridspec_kw={"width_ratios": [3.2, 1, 1]})
    ax = axes[0]
    for i, (name, row) in enumerate(zip(names, rows, strict=False)):
        offset = (i - (n - 1) / 2) * width
        values = [float(row.get(key, 0.0)) for key, _ in QUALITY_METRICS]
        bars = ax.bar(x + offset, values, width * 0.92, label=name,
                      color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, values, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in QUALITY_METRICS], fontsize=9)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Quality metrics by configuration", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    for ax_i, (key, title, unit) in zip(
            axes[1:], [("p95_latency_s", "p95 latency", "seconds"),
                       ("cost_per_query_usd", "Cost per query", "USD")], strict=False):
        values = [float(r.get(key, 0.0)) for r in rows]
        top = max(values)
        bars = ax_i.bar(names, values, color=[PALETTE[i % len(PALETTE)]
                                              for i in range(len(names))],
                        edgecolor="white", linewidth=0.5)
        if top <= 0:
            # An empty panel is worse than an honest one: say why it is zero.
            ax_i.set_ylim(0, 1.0)
            ax_i.text(0.5, 0.5, "0.00000 for every configuration\n(offline backend: no API spend)",
                      transform=ax_i.transAxes, ha="center", va="center", fontsize=9,
                      color="#555555")
        else:
            for bar, value in zip(bars, values, strict=False):
                label = f"{value:.4g}" if key == "cost_per_query_usd" else f"{value:.3f}"
                ax_i.text(bar.get_x() + bar.get_width() / 2, value + top * 0.02, label,
                          ha="center", va="bottom", fontsize=7)
            ax_i.set_ylim(0, top * 1.25)
        ax_i.set_title(title, fontsize=11, loc="left")
        ax_i.set_ylabel(unit)
        ax_i.tick_params(axis="x", labelrotation=35, labelsize=7.5)
        for tick in ax_i.get_xticklabels():
            tick.set_ha("right")
        ax_i.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax_i.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n, 5), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("rag-with-evals: configuration comparison", fontsize=13, x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_injection(stages: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Injection success rate by mitigation stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [s["stage"] for s in stages]
    rates = [100.0 * float(s["success_rate"]) for s in stages]
    refusals = [100.0 * float(s.get("refusal_rate", 0.0)) for s in stages]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(names))
    bars = ax.bar(x - 0.2, rates, 0.4, label="injection success rate",
                  color="#d62828", edgecolor="white")
    bars2 = ax.bar(x + 0.2, refusals, 0.4, label="refused / blocked",
                   color="#2a9d8f", edgecolor="white")
    for group in (bars, bars2):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("% of adversarial documents")
    ax.set_ylim(0, 112)
    ax.set_title("Prompt-injection success by mitigation stage", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
