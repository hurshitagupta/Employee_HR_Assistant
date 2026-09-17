MAX_INPUT_TOKENS = 500
MAX_STEPS = 1


def validate_input(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    if not text.strip():
        raise ValueError("Input cannot be empty.")

    estimated_tokens = len(text.split())

    if estimated_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Token budget exceeded: {estimated_tokens}/{MAX_INPUT_TOKENS}"
        )

    return text.strip()


def validate_output(output: str) -> str:
    if not isinstance(output, str):
        raise ValueError("Model output must be a string.")

    if not output.strip():
        raise ValueError("Model returned empty output.")

    return output.strip()


def check_step_limit(step: int) -> None:
    if step > MAX_STEPS:
        raise RuntimeError(
            f"Step limit reached: maximum {MAX_STEPS} model hop allowed per branch."
        )