from pathlib import Path

PROMPT_DIR = Path("app/prompts")


def load_prompt(path: str) -> str:
    return (PROMPT_DIR / path).read_text(encoding="utf-8")