from datasets import load_dataset
from mpe.extraction import extract_boxed
ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
print(len(ds))
matches = mismatches = nones = 0
examples = []

for row in ds:
    answer = extract_boxed(row["solution"])
    if answer is None:
        nones += 1
    elif answer == row["answer"]:
        matches += 1
    else:
        mismatches += 1
        examples.append({
            "got": answer,
            "expected": row["answer"],
            "solution_tail": row["solution"][-150:],  # last 150 chars, where the box usually is
        })

print(f"matches: {matches}")
print(f"mismatches: {mismatches}")
print(f"nones: {nones}")
print()
for ex in examples:
    print(ex)
    print("---")
    
