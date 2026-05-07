from mpe.data import get_problems

def test_returns_correct_total():
    problems = get_problems(n_per_level=10)
    assert len(problems) == 50

def test_each_level_represented_equally():
    problems = get_problems(n_per_level=10)
    levels = [p["level"] for p in problems]
    for level in [1, 2, 3, 4, 5]:
        assert levels.count(level) == 10

def test_deterministic_with_same_seed():
    a = get_problems(n_per_level=5, seed=123)
    b = get_problems(n_per_level=5, seed=123)
    assert [p["unique_id"] for p in a] == [p["unique_id"] for p in b]

def test_different_seeds_give_different_samples():
    a = get_problems(n_per_level=5, seed=1)
    b = get_problems(n_per_level=5, seed=2)
    assert [p["unique_id"] for p in a] != [p["unique_id"] for p in b]