import csv
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

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
    Estimated,
    EstimateProgress,
    Finished,
    PartResult,
    Progress,
    Started,
    Stretch,
    TooShort,
    Unreadable,
    is_file_error,
)

COLUMNS = ("Recording", "Part", "Status", "Tokens")

_RecordingState: TypeAlias = Literal["answered", "failed", "sending", "waiting"]


@dataclass(frozen=True)
class Row:
    recording: Path
    part: ChunkResult | None
    cells: tuple[str, ...]
    error: Exception | None = None
    # The stretch to cut to get exactly the audio sent for the part.
    audio: Stretch | None = None

    @property
    def key(self) -> tuple[Path, int | None]:
        return self.recording, None if self.part is None else self.part.start_ms

    @property
    def title(self) -> str:
        name, span, *_ = self.cells
        return f"{name}, {span}" if span else name

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def detail(self) -> dict[str, object] | None:
        if self.part is not None and self.part.response is not None:
            return self.part.response.model_dump(mode="json")
        return None if self.error is None else _describe_error(self.error)


@dataclass
class _Recording:
    parts: list[ChunkResult] = field(default_factory=list)
    finished: bool = False
    unreadable: ValueError | None = None
    sending_from: int | None = None


@dataclass
class _Estimate:
    tokens: int = 0
    parts: int = 0
    too_short: int = 0
    unreadable: int = 0
    finished: bool = False
    error: ApiError | httpx.HTTPError | None = None


class Results:
    def __init__(self) -> None:
        self._request: DecisionsRequest | None = None
        self._recordings: dict[Path, _Recording] = {}
        self._stopped_by: ApiError | httpx.HTTPError | None = None
        self._estimate: _Estimate | None = None

    def set_recordings(self, paths: Iterable[Path]) -> None:
        self._recordings = {
            path: self._recordings.get(path, _Recording()) for path in paths
        }
        self._estimate = None

    def check_request(self, request: DecisionsRequest) -> None:
        if self._has_answers() and not _same(request, self._request):
            raise ValueError(
                "The table holds answers to a different request. "
                "Press Clear results before sending this one."
            )

    def start_run(self, request: DecisionsRequest) -> list[Stretch]:
        stretches = self._unsent(request)
        if not _same(request, self._request):
            self._clear_parts()
            self._request = request
        self._stopped_by = None
        self._estimate = None
        return stretches

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

    def can_estimate(self) -> bool:
        return not self._has_answers() and any(
            _resend_from(recording) is not None
            for recording in self._recordings.values()
        )

    def start_estimate(self, request: DecisionsRequest) -> list[Stretch]:
        stretches = self._unsent(request)
        self._estimate = _Estimate()
        return stretches

    def record_estimate(self, progress: EstimateProgress) -> None:
        estimate = self._estimate
        # An error ends the estimate, so updates queued behind the error are dropped.
        if estimate is None or estimate.error is not None:
            return
        match progress:
            case Estimated(tokens=tokens):
                estimate.tokens += tokens
                estimate.parts += 1
            case TooShort():
                estimate.too_short += 1
            case Unreadable():
                estimate.unreadable += 1
            case BatchStopped(error):
                estimate.error = error
            case _:
                assert_never(progress)

    def end_estimate(self, *, stopped: bool) -> None:
        if stopped:
            self._estimate = None
        elif self._estimate is not None:
            self._estimate.finished = True

    def clear_estimate(self) -> None:
        self._estimate = None

    def clear(self) -> None:
        self._request = None
        self._stopped_by = None
        self._estimate = None
        self._clear_parts()

    def can_clear(self) -> bool:
        return self._request is not None

    def headers(self) -> list[str]:
        return [*COLUMNS, *self._questions()]

    def numeric_columns(self) -> list[bool]:
        questions = [] if self._request is None else self._request.questions.values()
        return [
            *(column == "Tokens" for column in COLUMNS),
            *(question.type in ("noul", "score") for question in questions),
        ]

    def rows(self) -> list[Row]:
        return [
            row
            for path, recording in self._recordings.items()
            for row in self._recording_rows(path, recording)
        ]

    def status(self) -> str:
        if not self._recordings:
            return "Add MP3 or WAV recordings to start."
        counts = Counter(_state(recording) for recording in self._recordings.values())
        tokens = sum(
            part.response.received.input_tokens
            for recording in self._recordings.values()
            for part in recording.parts
            if part.response is not None
        )
        line = (
            f"Answered {counts['answered']} of {len(self._recordings)} recordings. "
            f"{counts['failed']} failed, {counts['sending']} sending, "
            f"{counts['waiting']} waiting. Total input tokens billed: {tokens:,}."
        )
        if self._stopped_by is not None:
            line += f" {_stop_reason('Sending', self._stopped_by)}"
        return line

    def estimate_status(self) -> str:
        return "" if self._estimate is None else _describe_estimate(self._estimate)

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

    def _unsent(self, request: DecisionsRequest) -> list[Stretch]:
        self.check_request(request)
        if not _same(request, self._request):
            return [Stretch(path) for path in self._recordings]
        return [
            Stretch(path, start_ms)
            for path, recording in self._recordings.items()
            if (start_ms := _resend_from(recording)) is not None
        ]

    def _questions(self) -> list[str]:
        return [] if self._request is None else list(self._request.questions)

    def _recording_rows(self, path: Path, recording: _Recording) -> Iterator[Row]:
        questions = self._questions()
        start_ms = 0
        for part in recording.parts:
            span = f"{_clock(part.start_ms)} to {_clock(part.end_ms)}"
            cells = (path.name, span, *_part_cells(part, questions))
            # An MP3 part starts with frames repeated from before where it was cut,
            # so it is cut again from where the part before it ended.
            audio = Stretch(path, start_ms, part.end_ms)
            yield Row(path, part, cells, error=part.error, audio=audio)
            start_ms = part.end_ms

        last = recording.parts[-1] if recording.parts else None
        if recording.unreadable is not None:
            status, span = "Failed: unreadable file", ""
        elif recording.sending_from is not None:
            status, span = "Sending", f"from {_clock(recording.sending_from)}"
        elif recording.finished or (last is not None and _will_resend(last)):
            # A failure that can pass is sent again with the rest of the recording,
            # so its row already stands for what is left.
            return
        else:
            status = "Waiting"
            span = "" if last is None else f"from {_clock(last.end_ms)}"
        cells = (path.name, span, status, "", *[""] * len(questions))
        yield Row(path, None, cells, error=recording.unreadable)

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
                    yield {**entry, "error": _describe_error(part.error)}
            if recording.unreadable is not None:
                yield {
                    "recording": path.name,
                    "start_ms": None,
                    "end_ms": None,
                    "error": _describe_error(recording.unreadable),
                }


def _describe_error(error: Exception) -> dict[str, object]:
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
        if _will_resend(part):
            return start_ms
        if part.status == "not_attempted":
            return None
        # The next part starts where this one ends, not at its own start_ms, because
        # an MP3 part's start_ms includes frames repeated from before it.
        start_ms = part.end_ms
    return None if recording.finished else start_ms


def _will_resend(part: ChunkResult) -> bool:
    return part.error is not None and not is_file_error(part.error)


def _part_cells(part: ChunkResult, questions: list[str]) -> list[str]:
    if part.response is None:
        status = "Not sent" if part.error is None else f"Failed: {_reason(part.error)}"
        return [status, "", *[""] * len(questions)]
    answers = part.response.answers
    return [
        "Answered",
        f"{part.response.received.input_tokens:,}",
        *(
            _cell(answers[question], part.start_ms) if question in answers else ""
            for question in questions
        ),
    ]


def _state(recording: _Recording) -> _RecordingState:
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


def _stop_reason(activity: str, error: ApiError | httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPError):
        return f"{activity} stopped by a {_reason(error)}."
    if error.status_code == httpx.codes.UNAUTHORIZED:
        return f"{activity} stopped because the API key was rejected."
    if isinstance(error.body, Error):
        return f"{activity} stopped: {error.body.message}"
    return f"{activity} stopped by a {error.status_code} error."


def _describe_estimate(estimate: _Estimate) -> str:
    # A total that leaves out the parts that weren't estimated would understate the run.
    if estimate.error is not None:
        return _stop_reason("Estimating", estimate.error)
    total = f"{estimate.tokens:,} input tokens over {_count(estimate.parts, 'part')}"
    if not estimate.finished:
        return f"Estimating the next run: {total} so far."
    notes = []
    if estimate.too_short:
        notes.append(f"{_count(estimate.too_short, 'part')} too short to send")
    if estimate.unreadable:
        notes.append(f"{_count(estimate.unreadable, 'file')} unreadable")
    line = f"The next run is estimated at {total}."
    return f"{line} {', '.join(notes)}." if notes else line


def _count(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


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
