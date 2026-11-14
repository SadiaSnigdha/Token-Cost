# Bangla vs English Ollama Token Cost

This project measures the token and notional cost difference between 50 aligned
English/Bangla prompts using a local Ollama model.

## Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`)
- A model pulled locally, for example `ollama pull qwen2.5:latest`

The code uses only Python's standard library. The default model is
`qwen2.5:latest`; override it with `--model`.

## Reproduce from PowerShell

Start Ollama in another terminal if it is not already running:

```powershell
ollama serve
ollama pull qwen2.5:latest
```

In this project directory, verify the local API:

```powershell
Invoke-WebRequest http://localhost:11434/api/tags
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Run the 100-call experiment. `-u` makes progress visible immediately:

```powershell
python -u runner.py --model qwen2.5:latest --max-tokens 128 --keep-alive 10m
```

Use `--keep-alive 0` for a fresh model state per call. This more strictly
avoids prompt-prefix reuse but is much slower because Ollama unloads the model
after every request. `10m` is faster and is applied identically to English and
Bangla. The selected value is recorded in `run_metadata.json`.

Generate the pair-level comparison, chart, and report:

```powershell
python analyze.py
```

`runner.py` first makes one baseline request with the fixed system prompt and
the tiny prompt `x`, then makes 100 calls. Every call uses the same model,
system prompt, temperature (`0`), seed (`42`), and output limit. Ollama's
`prompt_eval_count` and `eval_count` are copied without modification into
`results.csv`. The baseline counts and all run settings are stored in
`run_metadata.json`.

`analyze.py` reads `results.csv`, creates `comparison.csv`, and writes
`report.md` and `ratio_by_pair.svg` (no plotting dependency is needed).
After reviewing pairs 1, 9, 18, 25, 33, 39, 42, 44, 48, and 50 in
`results.csv`, create `quality_check.csv` with this format:

```csv
id,judgment,notes
1,Yes,"Both answers identify Tokyo and are useful."
9,Yes,"Both provide three practical electricity-saving methods."
```

Add the remaining eight selected pairs, then rerun `python analyze.py`. The
report will include the judgments; missing judgments are shown as `PENDING`.

If Ollama is unavailable, the runner fails explicitly rather than producing
success-shaped fake results. Tests use a fake backend and do not require Ollama.

## Files

- `dataset.tsv`: 50 aligned prompts with varied categories and lengths.
- `ollama_client.py`: reusable client and token/cost helpers.
- `runner.py`: baseline and 100-call experiment runner.
- `analyze.py`: comparison, bootstrap interval, chart, and report generator.
- `results.csv`: one row per completed call (created by the runner).
- `comparison.csv`: one row per aligned pair (created by analysis).
- `run_metadata.json`: model, decoding settings, cache setting, and baseline counts.
- `quality_check.csv`: optional manual judgments for ten answer pairs.
- `report.md` and `ratio_by_pair.svg`: generated analysis outputs.
- `tests/`: focused fake-backend tests.

The dataset was written and reviewed manually; no translation API was used.
