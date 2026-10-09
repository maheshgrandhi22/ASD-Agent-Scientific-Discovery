
def validate_research_question(question):
    """Validate a research question before processing it."""

    if not isinstance(question, str):
        return False, "Question must be text."

    question = question.strip()

    if not question:
        return False, "Research question cannot be empty."

    if len(question) < 10:
        return False, "Question must contain at least 10 characters."

    if len(question) > 500:
        return False, "Question cannot exceed 500 characters."

    return True, "Research question is valid."


if __name__ == "__main__":
    question = input("Enter your research question: ")
    valid, message = validate_research_question(question)

    print(message)