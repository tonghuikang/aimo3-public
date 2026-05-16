"""
Claude Code Hook: Stop Validator.

Validates that edits are followed by tests and include confirmation phrase.
"""

import json

PHRASE_TO_CHECK = "I have addressed every query from the user."

CHECKING_INSTRUCTIONS = """
Review your work.

You will

1) Enumerate over every requirement from the user
    - State the requirement
    - Cite the user instruction
    - Add to the `TaskCreate` tool

2) Check whether the user instruction is followed
    - For each task
    - Reason whether you have addressed the requirement

If you have made edits, you will ALSO

1) Run tests
    - Search for appropriate tests,
    - Read up how to run the test.
    - Run the test.
2) Run the formatter
    - See CLAUDE.md for instructions
"""

BASH_AFTER_EDIT_REMINDER = "It seems that you did not run bash after your last edit."

TASK_CREATE_AFTER_EDIT_REMINDER = (
    "It seems that you did not use the TaskCreate tool after your last edit."
)


def validate_stop(transcript_path: str) -> list[str]:
    """Validate that edits are followed by bash commands and include confirmation phrase."""
    issues = []

    return issues
