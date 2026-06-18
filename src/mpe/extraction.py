def extract_boxed(text: str) -> str | None:
    """Return the content inside the last \\boxed{...} in `text`, or None if not found."""
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None

    i = start + len(r"\boxed{")
    depth = 1

    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(r"\boxed{"):i]
        i += 1

    return None