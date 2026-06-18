"""Eval loop: run a prompting strategy over a problem set and score results."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from mpe.extraction import extract_boxed
from mpe.inference import complete


def _majority(answers: list[str | None]) -> str | None:
    """Return the most common non-None answer, or None if all are None."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def run(
    *,
    problems: list[dict],
    prompt_fn: Callable[[dict], str],
    model: str,
    strategy_name: str,
    sc_samples: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
    output_path: Path | None = None,
) -> list[dict]:
    """Run a single strategy over `problems` and return per-problem result dicts.

    Args:
        problems: list of problem dicts from data.get_problems()
        prompt_fn: one of the builders from prompts.py
        model: Ollama model name, e.g. "llama3.1:8b"
        strategy_name: label written into results (e.g. "zero_shot_cot")
        sc_samples: how many times to sample per problem (>1 = self-consistency)
        temperature: passed to Ollama; use 0 for greedy, >0 for self-consistency
        seed: base seed; incremented per sample so samples are independent
        output_path: if given, write results as JSON Lines to this file
    """
    results = []

    for i, prob in enumerate(problems):
        prompt = prompt_fn(prob)
        raw_outputs = []

        for s in range(sc_samples):
            raw = complete(model, prompt, temperature=temperature, seed=seed + s)
            raw_outputs.append(raw)

        extracted = [extract_boxed(r) for r in raw_outputs]
        predicted = _majority(extracted) if sc_samples > 1 else extracted[0]
        correct = predicted == prob["answer"]

        result = {
            "strategy": strategy_name,
            "model": model,
            "problem_id": prob.get("unique_id", i),
            "level": prob["level"],
            "subject": prob["subject"],
            "answer": prob["answer"],
            "predicted": predicted,
            "correct": correct,
            "sc_samples": sc_samples,
            "raw_outputs": raw_outputs,
        }
        results.append(result)

        status = "OK" if correct else "WRONG" if predicted else "NONE"
        print(f"[{i+1}/{len(problems)}] level={prob['level']} {status} "
              f"pred={predicted!r} gold={prob['answer']!r}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} results to {output_path}")

    return results


def score(results: list[dict]) -> dict:
    """Summarize accuracy overall and by difficulty level."""
    overall = sum(r["correct"] for r in results) / len(results)
    by_level = {}
    for level in [1, 2, 3, 4, 5]:
        subset = [r for r in results if r["level"] == level]
        if subset:
            by_level[level] = sum(r["correct"] for r in subset) / len(subset)
    return {"overall": overall, "by_level": by_level}
