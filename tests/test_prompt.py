from twitter_scrapper.prompt import SYSTEM_PROMPT, user_prompt


def test_prompt_contains_strict_filtering_rules() -> None:
    assert "talking-head" in SYSTEM_PROMPT
    assert "plain screen recording" in SYSTEM_PROMPT
    assert "eligible_for_collection" in SYSTEM_PROMPT


def test_user_prompt_preserves_filename_and_video_context() -> None:
    prompt = user_prompt("fintech-launch.mp4", 31.5, 2)
    assert "'fintech-launch.mp4'" in prompt
    assert "31.50 seconds" in prompt
    assert "2 contact sheet(s)" in prompt
