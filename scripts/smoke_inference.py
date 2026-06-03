"""Manual smoke test for mpe.inference.complete against a live Ollama server.

Run with the daemon up and the model pulled:
    ollama serve            # if not already running
    ollama pull llama3.1:8b
    uv run python scripts/smoke_inference.py
"""

from mpe.inference import complete

MODEL = "llama3.1:8b"


def main():
    # Deterministic sanity check: temperature=0 + fixed seed should be stable
    # and obviously correct.
    out = complete(MODEL, "What is 2+2? Reply with only the number.", temperature=0, seed=0)
    print(f"prompt -> {out!r}")
    assert out.strip(), "got empty string — is `ollama serve` running and the model pulled?"

    # Failure path: a model that doesn't exist should return "" (not raise).
    bad = complete("definitely-not-a-real-model", "hi")
    print(f"bad model -> {bad!r}")
    assert bad == "", "expected empty string on failure"

    print("OK")


if __name__ == "__main__":
    main()
