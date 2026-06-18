"""Run all prompting strategies against a model and save results.

Usage:
    uv run python scripts/run_eval.py --model llama3.1:8b --n 20 --out results/run_01.jsonl

This runs zero_shot, zero_shot_cot, few_shot, few_shot_cot on n problems per
difficulty level (n*5 total), plus a self-consistency pass (8 samples) on
zero_shot_cot. Results are written as JSON Lines for analysis.
"""

import argparse
from pathlib import Path

from mpe.data import get_problems
from mpe.eval import run, score
from mpe.prompts import STRATEGIES

SC_SAMPLES = 8
SC_TEMPERATURE = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--n", type=int, default=20, help="Problems per difficulty level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()

    problems = get_problems(n_per_level=args.n, seed=args.seed)
    print(f"Loaded {len(problems)} problems ({args.n} per level)")

    all_results = []

    # Greedy strategies
    for strategy_name, prompt_fn in STRATEGIES.items():
        print(f"\n=== {strategy_name} ===")
        results = run(
            problems=problems,
            prompt_fn=prompt_fn,
            model=args.model,
            strategy_name=strategy_name,
            sc_samples=1,
            temperature=0.0,
            seed=args.seed,
            output_path=None,
        )
        s = score(results)
        print(f"  accuracy: {s['overall']:.1%}  by level: {s['by_level']}")
        all_results.extend(results)

    # Self-consistency on zero_shot_cot
    print(f"\n=== zero_shot_cot_sc{SC_SAMPLES} ===")
    from mpe.prompts import zero_shot_cot
    sc_results = run(
        problems=problems,
        prompt_fn=zero_shot_cot,
        model=args.model,
        strategy_name=f"zero_shot_cot_sc{SC_SAMPLES}",
        sc_samples=SC_SAMPLES,
        temperature=SC_TEMPERATURE,
        seed=args.seed,
    )
    s = score(sc_results)
    print(f"  accuracy: {s['overall']:.1%}  by level: {s['by_level']}")
    all_results.extend(sc_results)

    # Write everything
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        import json
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_results)} total results to {args.out}")


if __name__ == "__main__":
    main()
