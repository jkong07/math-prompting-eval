# math-prompting-eval

An empirical study of LLM prompting strategies on the MATH benchmark using local open-source models via Ollama.

**Research question:** Do prompting strategies (chain-of-thought, few-shot, self-consistency) still meaningfully improve LLM performance on current open-source models, and how does the answer vary by problem difficulty?

## Setup

```bash
git clone git@github.com:yourusername/math-prompting-eval.git
cd math-prompting-eval
uv sync
```

Requires [Ollama](https://ollama.com) for local inference.

## Project structure

```
src/mpe/
  data.py         — stratified sampler (n problems per difficulty level 1–5)
  inference.py    — Ollama wrapper (complete())
  extraction.py   — extract_boxed() parser for \boxed{} answers
  prompts.py      — prompt builders: zero_shot, zero_shot_cot, few_shot, few_shot_cot
  eval.py         — run() eval loop with self-consistency (majority vote), score()
  analysis.py     — accuracy_table(), McNemar pairwise significance test, difficulty correlation

scripts/
  run_eval.py         — CLI: run all strategies and write JSON Lines results
  smoke_inference.py  — quick sanity check for Ollama connectivity
  validate_extractor.py — validate extract_boxed against the MATH train split

notebooks/
  01_results_analysis.ipynb — accuracy table, figures, significance tests
```

## Running an evaluation

```bash
uv run python scripts/run_eval.py --model llama3.1:8b --n 20 --out results/run_01.jsonl
```

This runs `zero_shot`, `zero_shot_cot`, `few_shot`, and `few_shot_cot` (greedy) plus a self-consistency pass (`zero_shot_cot` with 8 samples) on `n` problems per difficulty level (5 levels → `n*5` problems total). Results are written as JSON Lines for downstream analysis.

| flag | default | description |
|------|---------|-------------|
| `--model` | required | Ollama model name, e.g. `llama3.1:8b` |
| `--n` | `20` | problems per difficulty level |
| `--seed` | `42` | random seed for stratified sampling |
| `--out` | `results/latest.jsonl` | output path |

## Prompting strategies

| strategy | description |
|----------|-------------|
| `zero_shot` | direct instruction, no examples |
| `zero_shot_cot` | adds "think step by step" to the instruction |
| `few_shot` | 4 hand-written examples (one per difficulty level 1–4) |
| `few_shot_cot` | few-shot examples + step-by-step instruction |
| `zero_shot_cot_sc8` | self-consistency: 8 samples of `zero_shot_cot`, majority vote |

## Analysis

Load results and compute statistics in `notebooks/01_results_analysis.ipynb` or directly:

```python
from mpe.analysis import load_results, accuracy_table, mcnemar_pairwise, difficulty_correlation

df = load_results("results/run_01.jsonl")
print(accuracy_table(df))
print(mcnemar_pairwise(df, "zero_shot", "zero_shot_cot"))
print(difficulty_correlation(df))
```

## Author

Johnny Kong — CS + Math, University of Michigan