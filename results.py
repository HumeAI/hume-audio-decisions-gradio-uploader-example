import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import assert_never

import httpx
from hume_audio_decisions import (
    Answer,
    Answer_Choice,
    Answer_Moment,
    Answer_Noul,
    Answer_Score,
    DecisionsRequest,
    Error,
)
from hume_audio_decisions.audio_helpers import ChunkResult
from hume_audio_decisions.core import ApiError
from pydantic import BaseModel

from evaluation import (
    BatchStopped,
    Finished,
    PartResult,
    Progress,
    Started,
    Stretch,
    Unreadable,
)

COLUMNS = ("Recording", "Part", "Status")

# These errors come from the file itself, so sending it again unchanged would fail
# the same way.
_FILE_ERROR_STATUSES = frozenset({413, 415})
_FILE_ERROR_CODES = frozenset(
    {
        "audio_decode_failed",
        "audio_format_unsupported",
        "audio_too_short",
        "audio_too_long",
    }
)


@dataclass(frozen=True)
class Row:
    recording: Path
    part: ChunkResult | None
    cells: tuple[str, ...]


@dataclass
class _Recording:
    parts: list[ChunkResult] = field(default_factory=list)
    finished: bool = False
    unreadable: ValueError | None = None
    sending_from: int | None = None


class Results:
    def __init__(self) -> None:
        self._request: DecisionsRequest | None = None
        self._recordings: dict[Path, _Recording] = {}
        self._stopped_by: ApiError | None = None

    def set_recordings(self, paths: Iterable[Path]) -> None:
        self._recordings = {
            path: self._recordings.get(path, _Recording()) for path in paths
        }

    def check_request(self, request: DecisionsRequest) -> None:
        if self._has_answers() and not _same(request, self._request):
            raise ValueError(
                "The table holds answers to a different request. "
                "Press Clear results before sending this one."
            )

    def start_run(self, request: DecisionsRequest) -> list[Stretch]:
        self.check_request(request)
        if not _same(request, self._request):
            self._clear_parts()
            self._request = request
        self._stopped_by = None
        return [
            Stretch(path, start_ms)
            for path, recording in self._recordings.items()
            if (start_ms := _resend_from(recording)) is not None
        ]

    def record(self, progress: Progress) -> None:
        match progress:
            case Started(stretch):
                recording = self._recordings[stretch.path]
                recording.parts = [
                    part for part in recording.parts if part.end_ms <= stretch.start_ms
                ]
                recording.finished = False
                recording.unreadable = None
                recording.sending_from = stretch.start_ms
            case PartResult(stretch, part):
                recording = self._recordings[stretch.path]
                recording.parts.append(part)
                recording.sending_from = part.end_ms
            case Finished(stretch):
                recording = self._recordings[stretch.path]
                recording.finished = True
                recording.sending_from = None
            case Unreadable(stretch, error):
                recording = self._recordings[stretch.path]
                recording.unreadable = error
                recording.sending_from = None
            case BatchStopped(error):
                self._stopped_by = error
            case _:
                assert_never(progress)

    def end_run(self) -> None:
        for recording in self._recordings.values():
            recording.sending_from = None

    def clear(self) -> None:
        self._request = None
        self._stopped_by = None
        self._clear_parts()

    def headers(self) -> list[str]:
        return [*COLUMNS, *self._questions()]

    def rows(self) -> list[Row]:
        return [
            row
            for path, recording in self._recordings.items()
            for row in self._recording_rows(path, recording)
        ]

    def status(self) -> str:
        counts = dict.fromkeys(("answered", "failed", "sending", "waiting"), 0)
        for recording in self._recordings.values():
            counts[_state(recording)] += 1
        tokens = sum(
            part.response.received.input_tokens
            for recording in self._recordings.values()
            for part in recording.parts
            if part.response is not None
        )
        line = (
            f"Answered {counts['answered']} of {len(self._recordings)} recordings. "
            f"{counts['failed']} failed, {counts['sending']} sending, "
            f"{counts['waiting']} waiting. {tokens:,} input tokens billed."
        )
        if self._stopped_by is not None:
            line += f" {_stop_reason(self._stopped_by)}"
        return line

    def write_downloads(self, directory: Path) -> tuple[Path, Path]:
        csv_path = directory / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(self.headers())
            writer.writerows(row.cells for row in self.rows())

        json_path = directory / "results.json"
        export = {
            "request": None
            if self._request is None
            else self._request.model_dump(mode="json"),
            "parts": list(self._exported_parts()),
        }
        json_path.write_text(
            json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return csv_path, json_path

    def _has_answers(self) -> bool:
        return any(
            part.response is not None
            for recording in self._recordings.values()
            for part in recording.parts
        )

    def _clear_parts(self) -> None:
        self._recordings = {path: _Recording() for path in self._recordings}

    def _questions(self) -> list[str]:
        return [] if self._request is None else list(self._request.questions)

    def _recording_rows(self, path: Path, recording: _Recording) -> Iterator[Row]:
        questions = self._questions()
        for part in recording.parts:
            span = f"{_clock(part.start_ms)} to {_clock(part.end_ms)}"
            yield Row(path, part, (path.name, span, *_part_cells(part, questions)))

        last = recording.parts[-1] if recording.parts else None
        if recording.unreadable is not None:
            status, span = "Failed: unreadable file", ""
        elif recording.sending_from is not None:
            status, span = "Sending", f"from {_clock(recording.sending_from)}"
        elif recording.finished or (last is not None and _resent(last)):
            # A failure that can pass is sent again with the rest of the recording,
            # so its row already stands for what is left.
            return
        else:
            status = "Waiting"
            span = "" if last is None else f"from {_clock(last.end_ms)}"
        yield Row(path, None, (path.name, span, status, *[""] * len(questions)))

    def _exported_parts(self) -> Iterator[dict[str, object]]:
        for path, recording in self._recordings.items():
            for part in recording.parts:
                entry: dict[str, object] = {
                    "recording": path.name,
                    "start_ms": part.start_ms,
                    "end_ms": part.end_ms,
                }
                if part.response is not None:
                    yield {
                        **entry,
                        "run_id": part.response.run_id,
                        "response": part.response.model_dump(mode="json"),
                    }
                elif part.error is not None:
                    yield {**entry, "error": describe_error(part.error)}
            if recording.unreadable is not None:
                yield {
                    "recording": path.name,
                    "start_ms": None,
                    "end_ms": None,
                    "error": describe_error(recording.unreadable),
                }


def describe_error(error: Exception) -> dict[str, object]:
    if isinstance(error, ApiError):
        body = error.body
        if isinstance(body, BaseModel):
            body = body.model_dump(mode="json")
        return {"status_code": error.status_code, "body": body}
    return {"message": str(error) or type(error).__name__}


def _same(request: DecisionsRequest, other: DecisionsRequest | None) -> bool:
    # DecisionsRequest equality ignores the order of the questions, but the model
    # answers them together, so a reordered request can answer differently.
    return (
        other is not None
        and request == other
        and list(request.questions) == list(other.questions)
    )


def _resend_from(recording: _Recording) -> int | None:
    if recording.unreadable is not None:
        return None
    start_ms = 0
    for part in recording.parts:
        if _resent(part):
            return start_ms
        if part.status == "not_attempted":
            return None
        # The next part starts where this one ends, not at its own start_ms, because
        # an MP3 part's start_ms includes frames repeated from before it.
        start_ms = part.end_ms
    return None if recording.finished else start_ms


def _resent(part: ChunkResult) -> bool:
    error = part.error
    if error is None:
        return False
    if not isinstance(error, ApiError):
        return True
    if error.status_code in _FILE_ERROR_STATUSES:
        return False
    return not (isinstance(error.body, Error) and error.body.code in _FILE_ERROR_CODES)


def _part_cells(part: ChunkResult, questions: list[str]) -> list[str]:
    if part.response is None:
        status = "Not sent" if part.error is None else f"Failed: {_reason(part.error)}"
        return [status, *[""] * len(questions)]
    answers = part.response.answers
    return [
        "Answered",
        *(
            _cell(answers[question], part.start_ms) if question in answers else ""
            for question in questions
        ),
    ]


def _state(recording: _Recording) -> str:
    if recording.unreadable is not None or any(
        part.status == "failed" for part in recording.parts
    ):
        return "failed"
    if recording.finished:
        return "answered"
    if recording.sending_from is not None:
        return "sending"
    return "waiting"


def _reason(error: Exception) -> str:
    if isinstance(error, ApiError):
        if isinstance(error.body, Error):
            return str(error.body.code)
        return str(error.status_code)
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    return "connection error"


def _stop_reason(error: ApiError) -> str:
    if error.status_code == httpx.codes.UNAUTHORIZED:
        return "Sending stopped because the API key was rejected."
    if isinstance(error.body, Error):
        return f"Sending stopped: {error.body.message}"
    return f"Sending stopped by a {error.status_code} error."


def _cell(answer: Answer, start_ms: int) -> str:
    match answer:
        case Answer_Noul():
            return f"{answer.noul:.2f}"
        case Answer_Choice():
            return f"{answer.choice} {answer.probabilities[answer.choice]:.2f}"
        case Answer_Score():
            return f"{answer.score:.2f} of {len(answer.legend) - 1}"
        case Answer_Moment():
            times = (_clock(start_ms + m.audio_start_ms) for m in answer.moments)
            return ", ".join(times) or "none"
        case _:
            assert_never(answer)


def _clock(ms: int) -> str:
    # The first part of an MP3 file can start a few tens of milliseconds before 0.
    minutes, seconds = divmod(max(ms, 0) // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes}:{seconds:02}"
