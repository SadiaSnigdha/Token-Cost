"""Run the 50-pair Ollama experiment and write results.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ollama_client import OllamaHTTPBackend, run_call

ROOT = Path(__file__).parent
SYSTEM_PROMPT = "Answer accurately, helpfully, and concisely. Follow the user's requested format."
FIELDS = ["id", "language", "prompt", "output", "input_tokens",
          "output_tokens", "cost_usd", "latency_s"]


def read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"id", "category", "english", "bangla"}
    if len(rows) != 50 or (rows and set(rows[0]) != required):
        raise ValueError("dataset.tsv must contain exactly 50 rows and the required columns")
    if not rows or [row["id"] for row in rows] != [str(i) for i in range(1, 51)]:
        raise ValueError("dataset.tsv ids must be the consecutive values 1 through 50")
    if any(not row["english"].strip() or not row["bangla"].strip()
           for row in rows):
        raise ValueError("dataset.tsv prompts must not be empty")
    return rows


def run_experiment(model: str, max_tokens: int, output_path: Path,
                   keep_alive: int | str) -> None:
    backend = OllamaHTTPBackend()
    rows = read_dataset(ROOT / "dataset.tsv")
    # This is deliberately a separate call with the same system prompt.
    baseline = run_call(backend, model=model, system_prompt=SYSTEM_PROMPT,
                        prompt="x", max_tokens=max_tokens,
                        keep_alive=keep_alive)
    metadata_path = output_path.with_name("run_metadata.json")
    metadata_path.write_text(json.dumps({
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "keep_alive": keep_alive,
        "baseline_prompt": "x",
        "baseline_input_tokens": baseline.input_tokens,
        "baseline_output_tokens": baseline.output_tokens,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"baseline: input_tokens={baseline.input_tokens}, "
          f"output_tokens={baseline.output_tokens} (saved to {metadata_path})",
          flush=True)
    output_rows: list[dict[str, object]] = []
    for row in rows:
        for language in ("english", "bangla"):
            result = run_call(backend, model=model, system_prompt=SYSTEM_PROMPT,
                              prompt=row[language], max_tokens=max_tokens,
                              keep_alive=keep_alive)
            output_rows.append({
                "id": row["id"], "language": language, "prompt": row[language],
                "output": result.output, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": f"{result.cost_usd:.9f}",
                "latency_s": f"{result.latency_s:.4f}",
            })
            print(f"{row['id']:>2} {language:7} in={result.input_tokens:>4} "
                  f"out={result.output_tokens:>4} latency={result.latency_s:.2f}s",
                  flush=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:latest")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--keep-alive", default=0,
        help="Ollama model retention, e.g. 10m; use 0 for a fresh model state per call")
    parser.add_argument("--output", type=Path, default=ROOT / "results.csv")
    args = parser.parse_args()
    run_experiment(args.model, args.max_tokens, args.output, args.keep_alive)


if __name__ == "__main__":
    main()
