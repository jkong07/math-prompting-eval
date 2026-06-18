"""Prompt builders for each prompting strategy.

Each builder takes a problem dict (with a 'problem' key) and returns a prompt
string ready to pass to inference.complete().
"""

# ---------------------------------------------------------------------------
# Few-shot examples (hand-picked, one per difficulty level 1-4, AMC/AIME style)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = [
    {
        "problem": "What is the sum of all positive integers less than 20 that are divisible by 3?",
        "solution": (
            "The positive integers less than 20 divisible by 3 are 3, 6, 9, 12, 15, 18.\n"
            "Their sum is 3+6+9+12+15+18 = 63.\n"
            "The answer is \\boxed{63}."
        ),
    },
    {
        "problem": (
            "If $3x + 7 = 22$, what is the value of $x$?"
        ),
        "solution": (
            "Subtract 7 from both sides: $3x = 15$.\n"
            "Divide both sides by 3: $x = 5$.\n"
            "The answer is \\boxed{5}."
        ),
    },
    {
        "problem": (
            "How many ways can you arrange the letters in the word LEVEL?"
        ),
        "solution": (
            "LEVEL has 5 letters with L appearing twice and E appearing twice.\n"
            "The number of distinct arrangements is $\\frac{5!}{2! \\cdot 2!} = \\frac{120}{4} = 30$.\n"
            "The answer is \\boxed{30}."
        ),
    },
    {
        "problem": (
            "Find the sum of the roots of $x^2 - 5x + 6 = 0$."
        ),
        "solution": (
            "By Vieta's formulas, the sum of the roots of $ax^2+bx+c=0$ is $-b/a$.\n"
            "Here $a=1$, $b=-5$, so the sum is $-(-5)/1 = 5$.\n"
            "The answer is \\boxed{5}."
        ),
    },
]

_FEW_SHOT_BLOCK = "\n\n".join(
    f"Problem: {ex['problem']}\nSolution: {ex['solution']}"
    for ex in _FEW_SHOT_EXAMPLES
)

_INSTRUCTION = (
    "Solve the following math problem. "
    "Put your final answer inside \\boxed{}."
)

_COT_SUFFIX = " Think step by step before giving your final answer."


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def zero_shot(problem: dict) -> str:
    return f"{_INSTRUCTION}\n\nProblem: {problem['problem']}\nSolution:"


def zero_shot_cot(problem: dict) -> str:
    return f"{_INSTRUCTION}{_COT_SUFFIX}\n\nProblem: {problem['problem']}\nSolution:"


def few_shot(problem: dict) -> str:
    return (
        f"{_INSTRUCTION}\n\n"
        f"{_FEW_SHOT_BLOCK}\n\n"
        f"Problem: {problem['problem']}\nSolution:"
    )


def few_shot_cot(problem: dict) -> str:
    return (
        f"{_INSTRUCTION}{_COT_SUFFIX}\n\n"
        f"{_FEW_SHOT_BLOCK}\n\n"
        f"Problem: {problem['problem']}\nSolution:"
    )


STRATEGIES = {
    "zero_shot": zero_shot,
    "zero_shot_cot": zero_shot_cot,
    "few_shot": few_shot,
    "few_shot_cot": few_shot_cot,
}
