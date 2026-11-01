import json

from hume_audio_decisions import DecisionsRequest
from pydantic import ValidationError

MAX_QUESTIONS = 100
MAX_QUESTION_ID_LENGTH = 256

DEFAULT_REQUEST = json.dumps(
    {
        "context": (
            "A recording of one or more people speaking, such as a phone call, "
            "a meeting or a voice note."
        ),
        "questions": {
            "background_noise": {
                "type": "noul",
                "instructions": "Is there distracting background noise?",
            },
            "emotion": {
                "type": "choice",
                "instructions": "Which emotion comes across most in the speech?",
                "criteria": [
                    {"name": "Happy"},
                    {"name": "Sad"},
                    {"name": "Angry"},
                    {"name": "Neutral"},
                ],
            },
            "energy": {
                "type": "score",
                "instructions": "How energetic does the speech sound?",
                "criteria": ["Very low", "Low", "Moderate", "High", "Very high"],
            },
            "laughter": {
                "type": "moment",
                "instructions": "When does someone laugh?",
            },
        },
    },
    indent=2,
    ensure_ascii=False,
)


def parse_request(text: str) -> DecisionsRequest:
    try:
        data = json.loads(text, object_pairs_hook=_reject_repeated_keys)
    except json.JSONDecodeError as error:
        position = f"line {error.lineno}, column {error.colno}"
        message = f"The request isn't valid JSON: {error.msg} at {position}."
        raise ValueError(message) from error
    if not isinstance(data, dict):
        raise ValueError("The request must be a JSON object.")

    try:
        request = DecisionsRequest.model_validate(data)
    except ValidationError as error:
        raise ValueError(_describe(error)) from error

    count = len(request.questions)
    if not 1 <= count <= MAX_QUESTIONS:
        raise ValueError(
            f"questions: A request takes 1 to {MAX_QUESTIONS} questions, "
            f"and this one has {count}."
        )
    for question_id in request.questions:
        if not 1 <= len(question_id) <= MAX_QUESTION_ID_LENGTH:
            raise ValueError(
                f"questions: A question ID takes 1 to {MAX_QUESTION_ID_LENGTH} "
                f"characters, and one has {len(question_id)}."
            )
    return request


def _reject_repeated_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # json.loads keeps only the last value of a repeated key, so a repeated
    # question ID would drop a question without a word.
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f'The request repeats the key "{key}".')
        data[key] = value
    return data


def _describe(error: ValidationError) -> str:
    first = error.errors()[0]
    match first["loc"]:
        # Pydantic adds the question's type after its ID, which is not a key in
        # the request.
        case ("questions", question_id, _, *rest):
            location = ["questions", question_id, *rest]
        case other:
            location = list(other)
    return f"{'.'.join(map(str, location))}: {first['msg']}."
