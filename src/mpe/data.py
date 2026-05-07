from datasets import load_dataset
from collections import defaultdict
import random

_DS = None

def _get_dataset():
    global _DS
    if _DS is None:
        _DS = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
    return _DS


def get_problems(n_per_level: int, seed: int = 42) -> list[dict]:
    """Return a list of n_per_level problems from each of the 5 difficulty levels.
    Total returned = n_per_level * 5. Sampling is deterministic given the seed."""

    ds = _get_dataset()
    groups = defaultdict(list)
    for row in ds:
        groups[row["level"]].append(row)

    rng = random.Random(seed)  # local RNG for reproducibility
    sampled = []
    for level in [1, 2, 3, 4, 5]:
        sampled.extend(rng.sample(groups[level], n_per_level))
    return sampled