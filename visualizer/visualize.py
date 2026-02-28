#!/usr/bin/env python3
"""Render logprob response files as a token × record heatmap matrix.

Each record in the input file becomes one row in the matrix.  Columns are
token positions.  Cell colour encodes the log probability; cell text shows
the token string and its log-prob value.

Usage
-----
    python visualizer/visualize.py data/responses/single_word_logprobs_<id>.jsonl
    python visualizer/visualize.py data/responses/single_word_logprobs_<id>.jsonl -o out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict]:
    """Parse a pretty-printed JSONL file into a list of dicts."""
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        records.append(obj)
        pos = end
    return records


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _row_label(record: dict) -> str:
    """One-line label for a record row: model + variable values."""
    model = record.get("model", "")
    variables = record.get("variables") or {}
    var_str = ", ".join(f"{k}={v}" for k, v in variables.items())
    return f"{model}\n{var_str}" if var_str else model


def render(records: list[dict], output: Path | None = None) -> None:
    """Build and display (or save) a heatmap matrix for *records*.

    Records that have no logprobs data are silently skipped.
    """
    lp_records = [r for r in records if r.get("logprobs")]
    if not lp_records:
        print("No records with logprobs data found.", file=sys.stderr)
        sys.exit(1)

    # --- build one figure per record --------------------------------------
    # Each record gets its own matrix: rows = rank, cols = token position.
    # logprobs.content[pos].top_logprobs[rank] holds each cell's data.

    for record in lp_records:
        content = (record.get("logprobs") or {}).get("content") or []
        if not content:
            print(f"Skipping record {record.get('prompt_id')} — empty content.", file=sys.stderr)
            continue

        n_cols = len(content)
        n_rows = max(len(pos["top_logprobs"]) for pos in content)

        token_grid = np.empty((n_rows, n_cols), dtype=object)
        logprob_grid = np.full((n_rows, n_cols), np.nan)

        for col, pos_data in enumerate(content):
            for row, alt in enumerate(pos_data["top_logprobs"]):
                token_grid[row, col] = alt["token"]
                # Convert the logprob back to the softmaxed value.
                logprob_grid[row, col] = math.exp(alt["logprob"])

        # --- figure layout ------------------------------------------------
        CELL_W, CELL_H = 2.2, 1.4          # fixed inches per cell
        fig_w = n_cols * CELL_W + 3        # +3 for y-axis labels
        fig_h = n_rows * CELL_H + 2        # +2 for title and x-axis labels

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # --- heatmap ------------------------------------------------------
        # Values are now probabilities in [0, 1]; fix the scale accordingly.
        norm = plt.Normalize(vmin=0, vmax=1)
        im = ax.imshow(
            logprob_grid,
            aspect="auto",
            cmap="RdYlGn",
            norm=norm,
            interpolation="nearest",
        )

        # --- cell annotations ---------------------------------------------
        cmap = plt.cm.RdYlGn
        for row in range(n_rows):
            for col in range(n_cols):
                tok = token_grid[row, col]
                val = logprob_grid[row, col]
                if tok is None or np.isnan(val):
                    continue
                # Adaptive text colour: white on dark cells, black on light.
                r, g, b, _ = cmap(norm(val))
                text_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.45 else "white"
                # Replace whitespace chars so they render visibly.
                display = tok.replace(" ", "·").replace("\n", "↵").replace("\t", "→")
                ax.text(
                    col, row,
                    f"{display}\n{val:.3f}",
                    ha="center", va="center",
                    fontsize=10,
                    color=text_color,
                )

        # --- axes labels --------------------------------------------------
        selected = [pos["token"].replace(" ", "·").replace("\n", "↵") for pos in content]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(selected, fontsize=9, rotation=45, ha="right")
        ax.set_xlabel("Selected token at each position", fontsize=10)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"rank {i + 1}" for i in range(n_rows)], fontsize=10)

        title = f"{record.get('prompt_id', '')}  |  {record.get('model', '')}"
        if record.get("variables"):
            title += f"  |  {record['variables']}"
        ax.set_title(title, fontsize=12, pad=12)

        # --- colorbar -----------------------------------------------------
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label("probability", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout()

        if output:
            idx = lp_records.index(record)
            out_path = (
                output.with_name(f"{output.stem}_{idx}{output.suffix or '.png'}")
                if len(lp_records) > 1
                else output
            )
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_path}")
        else:
            plt.show()

        plt.close(fig)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a logprob JSONL response file as a token heatmap matrix."
    )
    parser.add_argument(
        "input",
        help="Path to a logprobs response JSONL file (from data/responses/).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save the figure to this path (PNG/PDF/SVG) instead of displaying it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = _load_records(Path(args.input))
    render(records, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
