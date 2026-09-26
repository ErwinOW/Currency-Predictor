from automation.pipeline import run_step


def test_successful_step_returns_true_and_runs():
    calls = []
    ok = run_step("ok_step", lambda: calls.append("ran"))

    assert ok is True
    assert calls == ["ran"]


def test_failing_step_returns_false_without_raising():
    def broken():
        raise ValueError("simulated failure")

    ok = run_step("broken_step", broken)  # must not raise

    assert ok is False


def test_a_failing_step_does_not_block_the_next_one():
    """The whole point of run_step: one bad data source (a dead API, a
    network blip) shouldn't take down the rest of the day's pipeline.
    """
    calls = []

    def broken():
        raise RuntimeError("simulated failure")

    def works():
        calls.append("ran")

    run_step("broken_step", broken)
    run_step("working_step", works)

    assert calls == ["ran"]


def test_args_and_kwargs_are_passed_through():
    received = {}

    def step(a, b, keyword=None):
        received["a"] = a
        received["b"] = b
        received["keyword"] = keyword

    run_step("step_with_args", step, 1, 2, keyword="value")

    assert received == {"a": 1, "b": 2, "keyword": "value"}
