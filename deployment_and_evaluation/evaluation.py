from grounded_knowledge.grounded_knowledge import answer_question

EVALUATION_CASES = [
    # Annual Leave
    {
        "question": "How many paid annual leave days do full-time employees receive?",
        "expected": "18",
        "should_refuse": False,
    },
    {
        "question": "How many annual leave days are provided each year?",
        "expected": "18",
        "should_refuse": False,
    },
    {
        "question": "How early should annual leave normally be requested?",
        "expected": "5",
        "should_refuse": False,
    },
    {
        "question": "How many unused annual leave days can be carried forward?",
        "expected": "5",
        "should_refuse": False,
    },
    {
        "question": "By when must carried-forward annual leave be used?",
        "expected": "31 March",
        "should_refuse": False,
    },

    # Sick Leave
    {
        "question": "How many paid sick leave days are available each year?",
        "expected": "8",
        "should_refuse": False,
    },
    {
        "question": "When may medical documentation be required for sick leave?",
        "expected": "3",
        "should_refuse": False,
    },

    # Employment
    {
        "question": "How long is the probation period?",
        "expected": "90",
        "should_refuse": False,
    },
    {
        "question": "What are the standard working hours?",
        "expected": "9:30",
        "should_refuse": False,
    },
    {
        "question": "What days do full-time employees normally work?",
        "expected": "Monday",
        "should_refuse": False,
    },

    # Employee Referral
    {
        "question": "What is the employee referral bonus?",
        "expected": "20,000",
        "should_refuse": False,
    },
    {
        "question": "After how many days does a referred employee become eligible for the referral bonus?",
        "expected": "90",
        "should_refuse": False,
    },

    # Internal Mobility / Performance
    {
        "question": "What is the guideline for internal transfer eligibility?",
        "expected": "6",
        "should_refuse": False,
    },
    {
        "question": "How often are performance reviews conducted?",
        "expected": "twice",
        "should_refuse": False,
    },

    # Payroll / Expenses
    {
        "question": "When is salary normally paid?",
        "expected": "last working day",
        "should_refuse": False,
    },
    {
        "question": "Within how many days should expense claims be submitted?",
        "expected": "30",
        "should_refuse": False,
    },
    {
        "question": "What month basis is used for loss of pay calculations?",
        "expected": "30",
        "should_refuse": False,
    },

    # Notice Period
    {
        "question": "What is the notice period for confirmed employees?",
        "expected": "30",
        "should_refuse": False,
    },
    {
        "question": "What is the notice period during probation?",
        "expected": "15",
        "should_refuse": False,
    },

    # Security / AI
    {
        "question": "How should API keys and credentials be stored?",
        "expected": "environment",
        "should_refuse": False,
    },
    {
        "question": "Can employees freely share confidential information with AI tools?",
        "expected": "confidential",
        "should_refuse": False,
    },

    # Unsupported Questions
    {
        "question": "Does the company provide free gym membership?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Does the company provide free lunch every day?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Which health insurance company does Northstar use?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Does the company provide employees with a company car?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the CEO's home address?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Does every employee receive a MacBook Pro?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "Does the company pay for Netflix subscriptions?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "What is the company's annual revenue?",
        "expected": None,
        "should_refuse": True,
    },
    {
        "question": "How many customers does the company have?",
        "expected": None,
        "should_refuse": True,
    },
]


def evaluate_case(case):
    response = answer_question(case["question"])

    if case["should_refuse"]:
        passed = response.refused is True

    else:
        passed = (
            response.refused is False
            and case["expected"].lower() in response.answer.lower()
        )

    return {
        "question": case["question"],
        "expected": case["expected"],
        "actual": response.answer,
        "refused": response.refused,
        "passed": passed,
    }


def run_evaluation():
    results = []

    print("=== EMPLOYEE HR ASSISTANT EVALUATION ===\n")

    for number, case in enumerate(EVALUATION_CASES, start=1):

        result = evaluate_case(case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{number:02d}. [{status}] "
            f"{result['question']}"
        )

        if not result["passed"]:
            print(f"    Expected: {result['expected']}")
            print(f"    Actual: {result['actual']}")

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = total - passed
    pass_rate = (passed / total) * 100

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.2f}%")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "results": results,
    }


if __name__ == "__main__":
    run_evaluation()