import json
import re

import pytest

from questions import DEFAULT_REQUEST, parse_request

QUESTION = {"type": "noul", "instructions": "Is someone speaking?"}


def request_text(questions: object) -> str:
    return json.dumps({"questions": questions})


def test_default_request_is_valid() -> None:
    parse_request(DEFAULT_REQUEST)


def test_accepts_the_most_questions_with_the_longest_ids() -> None:
    questions = {str(index).zfill(256): QUESTION for index in range(100)}

    request = parse_request(request_text(questions))

    assert list(request.questions) == list(questions)


def test_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match=r"isn't valid JSON: .* at line 1, column 15"):
        parse_request('{"questions": }')


def test_rejects_json_that_is_not_an_object() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_request(json.dumps([QUESTION]))


def test_rejects_a_repeated_question_id() -> None:
    text = """
    {
      "questions": {
        "tone": {"type": "noul", "instructions": "Does the speaker sound calm?"},
        "tone": {"type": "noul", "instructions": "Does the speaker sound tired?"}
      }
    }
    """

    with pytest.raises(ValueError, match='repeats the key "tone"'):
        parse_request(text)


@pytest.mark.parametrize(
    ("question", "location"),
    [
        pytest.param(
            {"type": "noul"}, "questions.q.instructions", id="no instructions"
        ),
        pytest.param(
            {"type": "yes_or_no", "instructions": "Is someone speaking?"},
            "questions.q",
            id="unknown type",
        ),
        pytest.param(
            {"type": "score", "instructions": "How loud is it?"},
            "questions.q.criteria",
            id="score without levels",
        ),
    ],
)
def test_rejects_a_question_that_does_not_match_the_sdk_model(
    question: dict[str, str], location: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(location)}: "):
        parse_request(request_text({"q": question}))


@pytest.mark.parametrize("count", [0, 101])
def test_rejects_a_question_count_outside_the_limit(count: int) -> None:
    questions = {f"q{index}": QUESTION for index in range(count)}

    with pytest.raises(ValueError, match=f"and this one has {count}"):
        parse_request(request_text(questions))


@pytest.mark.parametrize("length", [0, 257])
def test_rejects_a_question_id_length_outside_the_limit(length: int) -> None:
    with pytest.raises(ValueError, match=f"and one has {length}"):
        parse_request(request_text({"q" * length: QUESTION}))
