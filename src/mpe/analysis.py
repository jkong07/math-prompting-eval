"""Load results and compute statistics for the prompting strategy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a JSON Lines results file into a DataFrame."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r.pop("raw_outputs", None)  # drop verbose field
                rows.append(r)
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return accuracy by strategy × difficulty level, plus an Overall column."""
    strategies = df["strategy"].unique()
    levels = sorted(df["level"].unique())

    rows = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        row = {"strategy": strat}
        for lvl in levels:
            lvl_sub = sub[sub["level"] == lvl]
            row[f"level_{lvl}"] = lvl_sub["correct"].mean() if len(lvl_sub) else float("nan")
        row["overall"] = sub["correct"].mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy")


def mcnemar_pairwise(df: pd.DataFrame, strategy_a: str, strategy_b: str) -> dict:
    """McNemar's test comparing two strategies on the same problem set.

    Returns a dict with the test statistic, p-value, and a plain-English
    interpretation. McNemar is the right test here because the same problems
    are evaluated under both strategies (paired binary outcomes).
    """
    merged = df[df["strategy"] == strategy_a][["problem_id", "correct"]].merge(
        df[df["strategy"] == strategy_b][["problem_id", "correct"]],
        on="problem_id",
        suffixes=("_a", "_b"),
    )

    # Contingency table: [[both correct, a correct b wrong], [a wrong b correct, both wrong]]
    both_correct = ((merged["correct_a"]) & (merged["correct_b"])).sum()
    a_only       = ((merged["correct_a"]) & (~merged["correct_b"])).sum()
    b_only       = ((~merged["correct_a"]) & (merged["correct_b"])).sum()
    both_wrong   = ((~merged["correct_a"]) & (~merged["correct_b"])).sum()

    table = np.array([[both_correct, a_only], [b_only, both_wrong]])
    result = mcnemar(table, exact=False, correction=True)

    return {
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "n_pairs": len(merged),
        "a_only": int(a_only),
        "b_only": int(b_only),
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "significant_at_05": result.pvalue < 0.05,
    }


def difficulty_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between difficulty level and accuracy, per strategy."""
    rows = []
    for strat, grp in df.groupby("strategy"):
        by_level = grp.groupby("level")["correct"].mean()
        corr = by_level.index.to_series().corr(by_level)
        rows.append({"strategy": strat, "level_vs_accuracy_corr": corr})
    return pd.DataFrame(rows).set_index("strategy")
