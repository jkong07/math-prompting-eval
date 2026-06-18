https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark

## Dataset structure (nlile mirror)

Schema: problem, solution, level, type, answer
- `solution` is the full worked solution with \boxed{} containing the final answer
- `answer` is the pre-extracted answer (saves us writing extract_gold for the gold side)
- Will still write extract_boxed for model outputs, and use it on `solution` as a sanity check vs `answer`

## Answer formats observed
- Integers: 2, 18, 28
- Fractions: \dfrac{7}{20}, \frac{2\sqrt{149}}{3} (note: \dfrac AND \frac both appear)
- Decimals: 1.36
- Complex expressions with nested LaTeX (sqrt, fractions, etc.)
- 92.9% of answers are 0-16 chars; long tail is where equivalence checking gets hard

## 05/06/26
Validated extract_boxed against answer column on test split: 500/500 match. Note: dataset's answer column likely produced via similar \boxed{} extraction, so this validates convention-matching rather than absolute correctness. The real extractor stress test will come from hand-validating against actual model outputs once Phase 2 produces them.