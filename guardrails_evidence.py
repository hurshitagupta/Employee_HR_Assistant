import asyncio

from langchain_core.runnables import RunnableLambda

from guardrails import (
    MAX_INPUT_TOKENS,
    check_step_limit,
    validate_input
)


def show_step_limit():
    print("\n=== STEP LIMIT ===")

    try:
        check_step_limit(2)

    except RuntimeError as error:
        print(error)


async def slow_operation():
    await asyncio.sleep(2)
    return "completed"


async def show_timeout():
    print("\n=== TIMEOUT ===")

    try:
        await asyncio.wait_for(
            slow_operation(),
            timeout=0.5,
        )

    except asyncio.TimeoutError:
        print("Timeout fired: operation exceeded 0.5 seconds")



retry_attempts = 0


def unstable_operation(data):
    global retry_attempts

    retry_attempts += 1

    print(f"Retry attempt {retry_attempts}")

    if retry_attempts < 3:
        raise ValueError("Transient failure")

    return "Retry succeeded"


def show_retry():
    global retry_attempts
    retry_attempts = 0

    print("\n=== RETRY ===")

    retry_chain = RunnableLambda(
        unstable_operation
    ).with_retry(
        retry_if_exception_type=(ValueError,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )

    result = retry_chain.invoke({})

    print(result)
    print("Retry limit: maximum 3 attempts")



def show_token_budget():
    print("\n=== TOKEN BUDGET ===")

    oversized_input = "word " * (MAX_INPUT_TOKENS + 1)

    try:
        validate_input(oversized_input)

    except ValueError as error:
        print(error)

     

async def main():

    print("GUARDRAIL EVIDENCE")

    show_step_limit()

    await show_timeout()

    show_retry()

    show_token_budget()


if __name__ == "__main__":
    asyncio.run(main())