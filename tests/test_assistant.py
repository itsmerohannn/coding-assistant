from src.assistant import build_prompt


def test_build_prompt_includes_code_and_context():
    prompt = build_prompt("Review", "print('x')", "find issue", "ctx")
    assert "Task: Review" in prompt
    assert "print('x')" in prompt
    assert "ctx" in prompt


def test_build_prompt_fallback_task():
    prompt = build_prompt("Invalid", "x=1", "", "No retrieved context.")
    assert "Task: Explain" in prompt