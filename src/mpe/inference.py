import sys

import ollama


def complete(model: str, prompt: str, **kwargs) -> str:
    """Generate a completion from a local Ollama model.

    Sampling parameters (temperature, seed, top_p, num_predict, ...) are passed
    as keyword args and forwarded to Ollama's `options` dict. Returns the model's
    text response, or "" on failure so a single bad call never crashes the eval.
    """
    try:
        # Trap #1: sampling params are NOT top-level args to generate(); they
        # live in `options`. Forwarding kwargs wholesale routes temperature/seed
        # to the right place.
        response = ollama.generate(model=model, prompt=prompt, options=kwargs)
        # Trap #2: `response` is a GenerateResponse object, not a str. The text
        # is under `.response`.
        return response.response
    # Trap #3: a bare `except Exception` deliberately does NOT catch
    # KeyboardInterrupt (that's a BaseException), so Ctrl-C still aborts a bad
    # run. We log to stderr so a high failure rate is visible and never gets
    # silently misread as the model being wrong.
    except Exception as e:
        print(f"[complete] {type(e).__name__}: {e}", file=sys.stderr)
        return ""
