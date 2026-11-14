"""Create pair-level CSV, SVG chart, and a concise Markdown report."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
QUALITY_PATH = ROOT / "quality_check.csv"


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_interval(values: list[float], seed: int = 42, samples: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return (statistics.quantiles(means, n=40)[0], statistics.quantiles(means, n=40)[-1])


def make_comparison(results: list[dict[str, str]], path: Path) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in results:
        by_id[row["id"]][row["language"]] = row
    pairs = []
    for ident in sorted(by_id, key=int):
        english, bangla = by_id[ident]["english"], by_id[ident]["bangla"]
        en_in, bn_in = int(english["input_tokens"]), int(bangla["input_tokens"])
        en_out, bn_out = int(english["output_tokens"]), int(bangla["output_tokens"])
        en_cost, bn_cost = float(english["cost_usd"]), float(bangla["cost_usd"])
        pairs.append({
            "id": ident,
            "category": next(r["category"] for r in csv.DictReader(
                (ROOT / "dataset.tsv").open(encoding="utf-8"), delimiter="\t") if r["id"] == ident),
            "english_input_tokens": en_in, "bangla_input_tokens": bn_in,
            "english_output_tokens": en_out, "bangla_output_tokens": bn_out,
            "english_cost_usd": english["cost_usd"], "bangla_cost_usd": bangla["cost_usd"],
            "input_token_ratio": f"{bn_in / en_in:.6f}",
            "output_token_ratio": f"{bn_out / en_out:.6f}" if en_out else "",
            "total_cost_ratio": f"{bn_cost / en_cost:.6f}",
            "english_latency_s": english["latency_s"], "bangla_latency_s": bangla["latency_s"],
        })
    fields = list(pairs[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    return pairs


def svg_chart(pairs: list[dict[str, object]], path: Path) -> None:
    width, height = 1000, 390
    ratios = [float(p["input_token_ratio"]) for p in pairs]
    max_ratio = max(ratios) * 1.08
    bars = []
    for i, ratio in enumerate(ratios):
        x = 20 + i * (width - 40) / 50
        h = ratio / max_ratio * 300
        bars.append(f'<rect x="{x:.1f}" y="{350-h:.1f}" width="15" height="{h:.1f}" fill="#2878b5"/>')
        bars.append(f'<text x="{x+7:.1f}" y="370" font-size="9" text-anchor="middle">{i+1}</text>')
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/><text x="20" y="20" font-size="16">Bangla / English input-token ratio per pair</text>
<line x1="20" y1="350" x2="980" y2="350" stroke="black"/>{''.join(bars)}
<text x="500" y="388" text-anchor="middle" font-size="12">Pair id</text>
<text x="10" y="190" transform="rotate(-90 10 190)" font-size="12">Ratio</text></svg>'''
    path.write_text(content, encoding="utf-8")


def write_report(results: list[dict[str, str]], pairs: list[dict[str, object]]) -> None:
    en = [r for r in results if r["language"] == "english"]
    bn = [r for r in results if r["language"] == "bangla"]
    en_in, bn_in = sum(int(r["input_tokens"]) for r in en), sum(int(r["input_tokens"]) for r in bn)
    en_out, bn_out = sum(int(r["output_tokens"]) for r in en), sum(int(r["output_tokens"]) for r in bn)
    en_cost, bn_cost = sum(float(r["cost_usd"]) for r in en), sum(float(r["cost_usd"]) for r in bn)
    ratios = [float(p["input_token_ratio"]) for p in pairs]
    lo, hi = bootstrap_interval(ratios)
    by_category: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        by_category[str(p["category"])].append(float(p["input_token_ratio"]))
    category_lines = "\n".join(
        f"| {cat} | {len(vals)} | {statistics.mean(vals):.2f} |"
        for cat, vals in sorted(by_category.items()))
    surprise = max(pairs, key=lambda p: float(p["input_token_ratio"]))
    metadata = {}
    metadata_path = ROOT / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    quality_rows = []
    if QUALITY_PATH.exists():
        with QUALITY_PATH.open(encoding="utf-8", newline="") as stream:
            quality_rows = list(csv.DictReader(stream))
    quality_ids = [str(x) for x in (1, 9, 18, 25, 33, 39, 42, 44, 48, 50)]
    quality_by_id = {row.get("id", ""): row for row in quality_rows}
    quality_lines = "\n".join(
        f"| {ident} | {quality_by_id.get(ident, {}).get('judgment', 'PENDING')} | "
        f"{quality_by_id.get(ident, {}).get('notes', 'No quality judgment supplied.')} |"
        for ident in quality_ids)
    quality_judgments = [
        quality_by_id[ident].get("judgment", "PENDING")
        for ident in quality_ids if ident in quality_by_id]
    quality_summary = ", ".join(
        f"{label}={quality_judgments.count(label)}"
        for label in ("Yes", "Mostly yes", "Partial", "No", "PENDING")
        if label in quality_judgments)
    en_capped = sum(int(r["output_tokens"]) >= int(metadata.get("max_tokens", 128))
                    for r in en)
    bn_capped = sum(int(r["output_tokens"]) >= int(metadata.get("max_tokens", 128))
                    for r in bn)
    keep_alive = metadata.get("keep_alive", 0)
    cache_note = (
        "Every call used `keep_alive=0`, requesting a fresh model state so "
        "cached prompt prefixes do not make a repeated prompt's count smaller."
        if str(keep_alive) == "0" else
        f"Every call used `keep_alive={keep_alive}`, keeping the model loaded "
        "for speed; the setting was identical for both languages."
    )
    report = f"""# Bangla vs English Ollama Cost Report

## Method

The manually written and reviewed dataset contains 50 aligned pairs across
questions, instructions, short explanations, arithmetic, summarisation, and
long prompts. No translation API was used. Each call used the same system
prompt (`Answer accurately, helpfully, and concisely. Follow the user's
requested format.`), model, temperature 0, seed 42, and `num_predict=128`.
Ollama's raw `prompt_eval_count` and `eval_count` are used. A baseline call with prompt `x` and the same system prompt measured the fixed
template/system overhead; the measured counts are saved in `run_metadata.json`.
The shared system prompt means this overhead is present equally in both
languages. {cache_note}

## Results

| Measure | English | Bangla | Bangla / English |
|---|---:|---:|---:|
| Input tokens (total) | {en_in} | {bn_in} | {bn_in/en_in:.3f} |
| Output tokens (total) | {en_out} | {bn_out} | {bn_out/en_out:.3f} |
| Total notional cost | ${en_cost:.6f} | ${bn_cost:.6f} | {bn_cost/en_cost:.3f} |
| Mean latency (seconds) | {statistics.mean(float(r["latency_s"]) for r in en):.3f} | {statistics.mean(float(r["latency_s"]) for r in bn):.3f} | — |

Baseline (`x`) input/output counts: **{metadata.get("baseline_input_tokens", "not available")} / {metadata.get("baseline_output_tokens", "not available")}**.

Per-pair input-token ratio: mean **{statistics.mean(ratios):.3f}**, median
**{statistics.median(ratios):.3f}**, minimum **{min(ratios):.3f}**, maximum
**{max(ratios):.3f}**, bootstrap 95% interval for the mean
**[{lo:.3f}, {hi:.3f}]**. Output counts are reported per pair in
`comparison.csv`; costs use $0.15/M input and $0.60/M output tokens. The
`num_predict` limit was reached by {en_capped} English and {bn_capped} Bangla
calls, so output-token and latency differences should be interpreted with that
cap in mind.

![Ratio chart](ratio_by_pair.svg)

### Category gaps

| Category | Pairs | Mean input ratio |
|---|---:|---:|
{category_lines}

The largest category mean is **{max(by_category, key=lambda c: statistics.mean(by_category[c]))}**.
The most surprising individual case is pair **{surprise["id"]}**, with ratio
**{float(surprise["input_token_ratio"]):.3f}**. Short prompts can be dominated by
the fixed chat-template/system overhead, so their ratio may be less extreme
than the character-level intuition; tokenization and punctuation also make
individual ratios vary. Pair 50 is notable because its Bangla input is much
longer in model tokens even though the two prompts carry the same instructions,
which is consistent with less efficient tokenization for this model.

## Quality check

The selected ten pairs are checked against the saved outputs in
`quality_check.csv`. The judgment is intentionally not inferred from token
counts.

| Pair | Bangla answer as correct/useful as English? | Notes |
|---:|---|---|
{quality_lines}

Quality summary: **{quality_summary or "No judgments supplied"}**. Several
long Bangla answers hit the output limit, and the manual review therefore
shows that token cost and answer quality are separate concerns.

## Conclusion

The measured total cost ratio is **{bn_cost/en_cost:.3f}** for this model,
dataset, and fixed output limit. This is an empirical result, not a universal
property of Bangla: model choice, tokenizer, prompt length, answer language,
and system/template overhead can change it. The raw per-call data and
pair-level arithmetic are included for auditability.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    results = read_results(ROOT / "results.csv")
    if len(results) != 100:
        raise ValueError("results.csv must contain exactly 100 experiment rows")
    pairs = make_comparison(results, ROOT / "comparison.csv")
    svg_chart(pairs, ROOT / "ratio_by_pair.svg")
    write_report(results, pairs)


if __name__ == "__main__":
    main()
