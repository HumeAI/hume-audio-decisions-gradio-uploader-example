import csv
import json
from pathlib import Path

import httpx
import pytest
from hume_audio_decisions import (
    BadRequestError,
    ContentTooLargeError,
    DecisionsRequest,
    DecisionsResponse,
    Error,
    GatewayError,
    UnauthorizedError,
)
from hume_audio_decisions.audio_helpers import ChunkResult

from evaluation import BatchStopped, Finished, PartResult, Started, Stretch, Unreadable
from questions import DEFAULT_REQUEST, parse_request
from results import Results

REQUEST = parse_request(DEFAULT_REQUEST)
CALL = Path("call.wav")
LONG_CALL = Path("long_call.wav")


def response(
    *, tokens: int = 1000, moments: tuple[int, ...] = (), run_id: str = "run"
) -> DecisionsResponse:
    return DecisionsResponse.model_validate(
        {
            "run_id": run_id,
            "answers": {
                "background_noise": {"type": "noul", "noul": 0.1274},
                "emotion": {
                    "type": "choice",
                    "choice": "Happy",
                    "confidence": 0.3412,
                    "probabilities": {
                        "Happy": 0.5402,
                        "Sad": 0.2611,
                        "Angry": 0.1527,
                        "Neutral": 0.046,
                    },
                },
                "energy": {
                    "type": "score",
                    "score": 2.9214,
                    "confidence": 0.4503,
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3},
                    "legend": {
                        "0": "Very low",
                        "1": "Low",
                        "2": "Moderate",
                        "3": "High",
                        "4": "Very high",
                    },
                },
                "laughter": {
                    "type": "moment",
                    "moments": [
                        {
                            "audio_start_ms": start,
                            "audio_end_ms": start + 1000,
                            "probability": 0.9,
                        }
                        for start in moments
                    ],
                },
            },
            "received": {"input_tokens": tokens},
        }
    )


def file_error(code: str) -> BadRequestError:
    return BadRequestError(
        body=Error(type="error", code=code, message=f"{code}.", request_id="req")
    )


def unauthorized() -> UnauthorizedError:
    return UnauthorizedError(body=GatewayError(message="Invalid API key."))


def answered(
    index: int,
    start_ms: int,
    end_ms: int,
    *,
    tokens: int = 1000,
    moments: tuple[int, ...] = (),
) -> ChunkResult:
    return ChunkResult(
        index,
        start_ms,
        end_ms,
        response=response(tokens=tokens, moments=moments, run_id=f"run-{index}"),
        error=None,
    )


def failed(index: int, start_ms: int, end_ms: int, error: Exception) -> ChunkResult:
    return ChunkResult(index, start_ms, end_ms, response=None, error=error)


def not_sent(index: int, start_ms: int, end_ms: int) -> ChunkResult:
    return ChunkResult(index, start_ms, end_ms, response=None, error=None)


def started(*paths: Path, request: DecisionsRequest = REQUEST) -> Results:
    results = Results()
    results.set_recordings(paths)
    results.start_run(request)
    return results


def send(results: Results, path: Path, *parts: ChunkResult, finish: bool) -> None:
    stretch = Stretch(path, 0)
    results.record(Started(stretch))
    for part in parts:
        results.record(PartResult(stretch, part))
    if finish:
        results.record(Finished(stretch))


def cells(results: Results) -> list[tuple[str, ...]]:
    return [row.cells for row in results.rows()]


def test_each_question_type_has_its_own_cell() -> None:
    results = started(LONG_CALL)
    part = answered(1, 274_000, 548_000, moments=(88_140, 247_310))

    send(results, LONG_CALL, part, finish=True)

    assert results.headers() == [
        "Recording",
        "Part",
        "Status",
        "background_noise",
        "emotion",
        "energy",
        "laughter",
    ]
    assert cells(results) == [
        (
            "long_call.wav",
            "4:34 to 9:08",
            "Answered",
            "0.13",
            "Happy 0.54",
            "2.92 of 4",
            "6:02, 8:41",
        )
    ]


@pytest.mark.parametrize(
    ("start_ms", "moments", "cell"),
    [
        pytest.param(0, (), "none", id="no moments"),
        pytest.param(3_540_000, (61_000,), "1:00:01", id="past an hour"),
    ],
)
def test_a_moment_cell_gives_times_in_the_whole_recording(
    start_ms: int, moments: tuple[int, ...], cell: str
) -> None:
    results = started(LONG_CALL)

    send(
        results,
        LONG_CALL,
        answered(0, start_ms, start_ms + 120_000, moments=moments),
        finish=True,
    )

    assert cells(results)[0][-1] == cell


@pytest.mark.parametrize(
    ("error", "status"),
    [
        pytest.param(
            file_error("audio_too_short"), "Failed: audio_too_short", id="code"
        ),
        pytest.param(unauthorized(), "Failed: 401", id="status without a code"),
        pytest.param(httpx.ReadTimeout("timed out"), "Failed: timeout", id="timeout"),
        pytest.param(
            httpx.ConnectError("refused"),
            "Failed: connection error",
            id="no connection",
        ),
    ],
)
def test_a_failed_row_names_the_reason(error: Exception, status: str) -> None:
    results = started(CALL)

    send(results, CALL, failed(0, 0, 60_000, error), finish=True)

    assert [row[:3] for row in cells(results)] == [("call.wav", "0:00 to 1:00", status)]


def test_rows_follow_a_recording_through_a_run() -> None:
    results = started(LONG_CALL, CALL)
    stretch = Stretch(LONG_CALL, 0)
    waiting = [
        ("long_call.wav", "", "Waiting"),
        ("call.wav", "", "Waiting"),
    ]
    assert [row[:3] for row in cells(results)] == waiting

    results.record(Started(stretch))
    results.record(PartResult(stretch, answered(0, 0, 274_000)))
    results.record(Unreadable(Stretch(CALL, 0), ValueError("The audio is empty.")))

    assert [row[:3] for row in cells(results)] == [
        ("long_call.wav", "0:00 to 4:34", "Answered"),
        ("long_call.wav", "from 4:34", "Sending"),
        ("call.wav", "", "Failed: unreadable file"),
    ]


def test_the_status_line_counts_recordings_and_tokens() -> None:
    paths = [Path(f"{name}.wav") for name in ("a", "b", "c", "d", "e")]
    results = started(*paths)

    send(results, paths[0], answered(0, 0, 60_000, tokens=1200), finish=True)
    send(
        results,
        paths[1],
        answered(0, 0, 300_000, tokens=3000),
        answered(1, 300_000, 600_000, tokens=2934),
        finish=True,
    )
    send(
        results,
        paths[2],
        answered(0, 0, 60_000, tokens=5000),
        failed(1, 60_000, 120_000, httpx.ReadTimeout("timed out")),
        finish=True,
    )
    send(results, paths[3], finish=False)

    assert results.status() == (
        "Answered 2 of 5 recordings. 1 failed, 1 sending, 1 waiting. "
        "12,134 input tokens billed."
    )


def test_the_status_line_says_why_the_batch_stopped() -> None:
    results = started(CALL)
    error = unauthorized()

    send(results, CALL, failed(0, 0, 60_000, error), finish=False)
    results.record(BatchStopped(error))
    results.end_run()

    assert results.status() == (
        "Answered 0 of 1 recordings. 1 failed, 0 sending, 0 waiting. "
        "0 input tokens billed. Sending stopped because the API key was rejected."
    )


def test_a_stopped_recording_resumes_where_it_stopped() -> None:
    results = started(LONG_CALL, CALL)
    send(results, LONG_CALL, answered(0, 0, 274_000), finish=False)

    results.end_run()

    assert [row[:3] for row in cells(results)] == [
        ("long_call.wav", "0:00 to 4:34", "Answered"),
        ("long_call.wav", "from 4:34", "Waiting"),
        ("call.wav", "", "Waiting"),
    ]
    assert results.start_run(REQUEST) == [Stretch(LONG_CALL, 274_000), Stretch(CALL, 0)]


def test_a_failure_that_can_pass_is_sent_again_with_the_rest() -> None:
    results = started(LONG_CALL)
    send(
        results,
        LONG_CALL,
        answered(0, 0, 274_000),
        failed(1, 274_000, 548_000, httpx.ReadTimeout("timed out")),
        not_sent(2, 548_000, 822_000),
        finish=True,
    )

    stretches = results.start_run(REQUEST)
    results.record(Started(stretches[0]))

    assert stretches == [Stretch(LONG_CALL, 274_000)]
    assert [row[:3] for row in cells(results)] == [
        ("long_call.wav", "0:00 to 4:34", "Answered"),
        ("long_call.wav", "from 4:34", "Sending"),
    ]


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(
            (
                answered(0, 0, 274_000),
                failed(1, 274_000, 548_000, file_error("audio_decode_failed")),
                answered(2, 548_000, 822_000),
            ),
            id="a part that failed alone",
        ),
        pytest.param(
            (
                failed(0, 0, 274_000, file_error("audio_decode_failed")),
                not_sent(1, 274_000, 548_000),
            ),
            id="a recording the error stopped",
        ),
        pytest.param(
            (
                failed(
                    0,
                    0,
                    274_000,
                    ContentTooLargeError(
                        body=Error(
                            type="error",
                            code="payload_too_large",
                            message="payload_too_large.",
                            request_id="req",
                        )
                    ),
                ),
            ),
            id="a part too large to send",
        ),
    ],
)
def test_an_error_that_would_repeat_is_not_sent_again(
    parts: tuple[ChunkResult, ...],
) -> None:
    results = started(LONG_CALL)
    send(results, LONG_CALL, *parts, finish=True)

    assert results.start_run(REQUEST) == []


def test_an_unreadable_file_is_not_sent_again() -> None:
    results = started(CALL)
    results.record(Started(Stretch(CALL, 0)))
    results.record(Unreadable(Stretch(CALL, 0), ValueError("The audio is empty.")))

    assert results.start_run(REQUEST) == []


@pytest.mark.parametrize(
    "questions",
    [
        pytest.param({"laughter": REQUEST.questions["laughter"]}, id="different"),
        pytest.param(dict(reversed(REQUEST.questions.items())), id="reordered"),
    ],
)
def test_a_changed_request_is_refused_while_the_table_holds_answers(
    questions: dict[str, object],
) -> None:
    changed = DecisionsRequest.model_validate(
        {"context": REQUEST.context, "questions": questions}
    )
    results = started(CALL)
    send(results, CALL, answered(0, 0, 60_000), finish=True)

    with pytest.raises(ValueError, match="Press Clear results"):
        results.check_request(changed)
    results.check_request(REQUEST)

    results.clear()
    results.check_request(changed)


def test_a_changed_request_without_answers_starts_over() -> None:
    results = started(CALL)
    send(results, CALL, failed(0, 0, 60_000, unauthorized()), finish=False)
    changed = DecisionsRequest.model_validate(
        {"questions": {"laughter": REQUEST.questions["laughter"]}}
    )

    stretches = results.start_run(changed)

    assert stretches == [Stretch(CALL, 0)]
    assert results.headers() == ["Recording", "Part", "Status", "laughter"]
    assert cells(results) == [("call.wav", "", "Waiting", "")]


def test_removing_a_recording_drops_its_answers() -> None:
    results = started(CALL, LONG_CALL)
    send(results, CALL, answered(0, 0, 60_000, tokens=500), finish=True)
    send(results, LONG_CALL, answered(0, 0, 60_000, tokens=700), finish=True)

    results.set_recordings([LONG_CALL, Path("new.wav")])

    assert [row[:3] for row in cells(results)] == [
        ("long_call.wav", "0:00 to 1:00", "Answered"),
        ("new.wav", "", "Waiting"),
    ]
    assert results.status().endswith("700 input tokens billed.")


def test_each_row_carries_its_response_or_error() -> None:
    results = started(LONG_CALL, CALL)
    send(
        results,
        LONG_CALL,
        answered(0, 0, 274_000),
        failed(1, 274_000, 548_000, file_error("audio_decode_failed")),
        not_sent(2, 548_000, 822_000),
        finish=True,
    )
    results.record(Unreadable(Stretch(CALL, 0), ValueError("The audio is empty.")))

    assert [row.detail for row in results.rows()] == [
        response(run_id="run-0").model_dump(mode="json"),
        {
            "status_code": 400,
            "body": {
                "type": "error",
                "code": "audio_decode_failed",
                "message": "audio_decode_failed.",
                "request_id": "req",
            },
        },
        None,
        {"message": "The audio is empty."},
    ]


def test_the_downloads_hold_the_table_and_every_response_and_error(
    tmp_path: Path,
) -> None:
    results = started(LONG_CALL, CALL)
    error = file_error("audio_decode_failed")
    send(
        results,
        LONG_CALL,
        answered(0, 0, 274_000, moments=(1_000,)),
        failed(1, 274_000, 548_000, error),
        finish=False,
    )
    send(
        results, CALL, failed(0, 0, 60_000, httpx.ConnectError("refused")), finish=True
    )
    results.end_run()

    csv_path, json_path = results.write_downloads(tmp_path)

    with csv_path.open(newline="", encoding="utf-8") as file:
        table = list(csv.reader(file))
    assert table == [results.headers(), *(list(row) for row in cells(results))]
    export = json.loads(json_path.read_text(encoding="utf-8"))
    assert export == {
        "request": json.loads(DEFAULT_REQUEST),
        "parts": [
            {
                "recording": "long_call.wav",
                "start_ms": 0,
                "end_ms": 274_000,
                "run_id": "run-0",
                "response": response(moments=(1_000,), run_id="run-0").model_dump(
                    mode="json"
                ),
            },
            {
                "recording": "long_call.wav",
                "start_ms": 274_000,
                "end_ms": 548_000,
                "error": {
                    "status_code": 400,
                    "body": {
                        "type": "error",
                        "code": "audio_decode_failed",
                        "message": "audio_decode_failed.",
                        "request_id": "req",
                    },
                },
            },
            {
                "recording": "call.wav",
                "start_ms": 0,
                "end_ms": 60_000,
                "error": {"message": "refused"},
            },
        ],
    }


def test_a_resumed_mp3_starts_where_its_last_answer_ended() -> None:
    recording = Path("call.mp3")
    results = started(recording)
    # Each MP3 part after the first starts a few frames early, to repeat the frames
    # it needs to decode.
    send(
        results,
        recording,
        answered(0, -26, 274_000),
        failed(1, 273_948, 548_000, httpx.ReadTimeout("timed out")),
        not_sent(2, 547_948, 822_000),
        finish=True,
    )

    assert results.start_run(REQUEST) == [Stretch(recording, 274_000)]
