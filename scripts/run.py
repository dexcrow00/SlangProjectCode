#!/usr/bin/env python3
"""CLI entry point for PromptingSlang."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.client import TogetherClient
from src.collector import ResponseCollector
from src.prompts import load_prompts
from src.runner import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_MODELS = [
    "ServiceNow-AI/Apriel-1.6-15b-Thinker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt open-source LLMs via TogetherAI and collect responses."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        default="data/prompts/example.jsonl",
        help="Path to a JSONL file of prompt templates.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Together model IDs to query (space-separated). Defaults to a curated list.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSONL path. Defaults to data/responses/<run_id>.jsonl "
            "relative to repo root."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        dest="max_tokens",
        help="Maximum tokens to generate per response (default: 512).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Explicit run identifier; auto-generated if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex
    repo_root = Path(__file__).resolve().parents[1]
    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "data" / "responses" / f"{run_id}.jsonl"
    )

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found — check your JSONL file.", file=sys.stderr)
        sys.exit(1)

    client = TogetherClient()
    gen_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    with ResponseCollector(output_path) as collector:
        runner = Runner(
            client=client,
            collector=collector,
            models=args.models,
            gen_kwargs=gen_kwargs,
            run_id=run_id,
        )
        runner.run(prompts)

    print(f"\nDone. Responses written to: {output_path}")


if __name__ == "__main__":
    main()
