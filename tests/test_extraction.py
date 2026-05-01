from mpe.extraction import extract_boxed


def test_extract_boxed_simple():
    assert extract_boxed("The answer is \\boxed{42}.") == "42"


def test_extract_boxed_nested_fraction():
    assert extract_boxed("...so we get \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"


def test_extract_boxed_multiple_takes_last():
    assert extract_boxed("First \\boxed{3}, then \\boxed{7}.") == "7"


def test_extract_boxed_missing_returns_none():
    assert extract_boxed("No box here.") is None