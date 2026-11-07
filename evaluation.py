import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeGuard, TypeVar

import httpx
from hume_audio_decisions import (
    AsyncDecisionsClient,
    DecisionsRequest,
    DefaultAsyncHttpxClient,
    Error,
    InputTokensResponse,
)
from hume_audio_decisions.audio_helpers import (
    AudioChunk,
    ChunkResult,
    iter_audio_chunks,
)
from hume_audio_decisions.core import ApiError

# Every API key in an organization shares a limit of 600 requests a minute, so this
# leaves room for other work.
REQUESTS_PER_MINUTE = 50
CONCURRENT_RECORDINGS = 4
MIN_PART_MS = 1_000
MAX_PART_MS = 300_000
MAX_RETRIES = 5

# These errors come from the key or the request, which every recording shares, so
# every other recording would fail the same way.
_BATCH_STOPPING_STATUSES = frozenset({401, 402, 403})
_BATCH_STOPPING_CODES = frozenset({"message_invalid", "question_invalid"})
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

_T = TypeVar("_T")


@dataclass(frozen=True)
class Stretch:
    path: Path
    start_ms: int = 0
    end_ms: int | None = None


@dataclass(frozen=True)
class Started:
    stretch: Stretch


@dataclass(frozen=True)
class PartResult:
    stretch: Stretch
    part: ChunkResult


@dataclass(frozen=True)
class Finished:
    stretch: Stretch


@dataclass(frozen=True)
class Unreadable:
    stretch: Stretch
    error: ValueError


@dataclass(frozen=True)
class Estimated:
    stretch: Stretch
    duration_ms: int
    tokens: int


@dataclass(frozen=True)
class TooShort:
    stretch: Stretch
    duration_ms: int


@dataclass(frozen=True)
class BatchStopped:
    error: ApiError | httpx.HTTPError


Progress = Started | PartResult | Finished | Unreadable | BatchStopped
EstimateProgress = Estimated | TooShort | Unreadable | BatchStopped


def make_client(
    *,
    requests_per_minute: float = REQUESTS_PER_MINUTE,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncDecisionsClient:
    pacer = _Pacer(interval=60 / requests_per_minute)
    http_client = DefaultAsyncHttpxClient(
        event_hooks={"request": [pacer]}, transport=transport
    )
    return AsyncDecisionsClient(httpx_client=http_client, max_retries=MAX_RETRIES)


def evaluate(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretches: Iterable[Stretch],
    stop: asyncio.Event,
) -> AsyncIterator[Progress]:
    return _run_workers(partial(_send_stretch, client, request), stretches, stop)


def estimate(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretches: Iterable[Stretch],
    stop: asyncio.Event,
) -> AsyncIterator[EstimateProgress]:
    return _run_workers(partial(_estimate_stretch, client, request), stretches, stop)


async def check_with_api(
    client: AsyncDecisionsClient, request: DecisionsRequest
) -> None:
    # Estimating input tokens runs the same checks on the questions and context as
    # a decisions request, without audio and without billing, so the app keeps no
    # copy of the API's limits.
    try:
        await client.decisions.input_tokens(
            audio_duration_ms=MIN_PART_MS,
            questions=request.questions,
            context=request.context,
        )
    except ApiError as error:
        body = error.body
        if isinstance(body, Error) and body.code in _BATCH_STOPPING_CODES:
            raise ValueError(_describe_request_error(body)) from error
    # Any other error comes from the key, the account or the connection, which Run
    # and Estimate report on their own.
    except httpx.HTTPError:
        pass


def is_file_error(error: Exception) -> bool:
    return _is_api_error(error, _FILE_ERROR_STATUSES, _FILE_ERROR_CODES)


def cut(stretch: Stretch) -> bytes:
    # A row's stretch is at most one part long, so it cuts to a single part.
    return next(_parts(stretch)).data


class _Pacer:
    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next_start = time.monotonic()

    async def __call__(self, _request: httpx.Request) -> None:
        now = time.monotonic()
        start = max(now, self._next_start)
        # Each request claims its start before sleeping, so requests that wait at
        # the same time queue up without a lock.
        self._next_start = start + self._interval
        await asyncio.sleep(start - now)


class _HaltError(Exception):
    pass


async def _run_workers(
    work: Callable[[Stretch, Callable[[_T], None]], Awaitable[None]],
    stretches: Iterable[Stretch],
    stop: asyncio.Event,
) -> AsyncIterator[_T]:
    updates: asyncio.Queue[_T | None] = asyncio.Queue()
    # The workers share one iterator, so each stretch goes to only one of them.
    remaining = iter(stretches)

    async def worker() -> None:
        for stretch in remaining:
            await work(stretch, updates.put_nowait)

    async def run() -> None:
        try:
            async with asyncio.TaskGroup() as group:
                watcher = group.create_task(_halt_when_set(stop))
                workers = [
                    group.create_task(worker()) for _ in range(CONCURRENT_RECORDINGS)
                ]
                await asyncio.wait(workers)
                watcher.cancel()
        except* _HaltError:
            pass
        finally:
            updates.put_nowait(None)

    # The workers run in their own task because a generator that yields inside a
    # TaskGroup lets the group cancel whatever its consumer is awaiting.
    running = asyncio.create_task(run())
    try:
        while (update := await updates.get()) is not None:
            yield update
        await running
    finally:
        running.cancel()


async def _halt_when_set(stop: asyncio.Event) -> None:
    await stop.wait()
    raise _HaltError


async def _send_stretch(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretch: Stretch,
    report: Callable[[Progress], None],
) -> None:
    report(Started(stretch))
    file = await asyncio.to_thread(stretch.path.open, "rb")
    with file:
        # The SDK leaves client.decisions untyped, so this annotation is what lets
        # mypy check the parts.
        parts: AsyncIterator[ChunkResult] = client.decisions.iter_create_chunked(
            request=request,
            file=(stretch.path.name, file),
            max_chunk_ms=MAX_PART_MS,
            start_ms=stretch.start_ms,
            end_ms=stretch.end_ms,
        )
        try:
            async for part in parts:
                report(PartResult(stretch, part))
                if _is_api_error(
                    part.error, _BATCH_STOPPING_STATUSES, _BATCH_STOPPING_CODES
                ):
                    report(BatchStopped(part.error))
                    raise _HaltError
        # The SDK raises ValueError only before it sends anything, when it can't
        # split the file.
        except ValueError as error:
            report(Unreadable(stretch, error))
        else:
            report(Finished(stretch))


async def _estimate_stretch(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretch: Stretch,
    report: Callable[[EstimateProgress], None],
) -> None:
    try:
        durations = await asyncio.to_thread(_durations, stretch)
    except ValueError as error:
        report(Unreadable(stretch, error))
        return
    for duration_ms in durations:
        # The API rejects a part this short, so Run's request for it would fail
        # without being billed.
        if duration_ms < MIN_PART_MS:
            report(TooShort(stretch, duration_ms))
            continue
        try:
            # The SDK leaves client.decisions untyped, so this annotation is what
            # lets mypy check the response.
            response: InputTokensResponse = await client.decisions.input_tokens(
                audio_duration_ms=duration_ms,
                questions=request.questions,
                context=request.context,
            )
        # The SDK has already retried what can pass, and a total that leaves out
        # parts would understate the run, so any error stops the estimate.
        except (ApiError, httpx.HTTPError) as error:
            report(BatchStopped(error))
            raise _HaltError from error
        report(Estimated(stretch, duration_ms, response.estimated_input_tokens))


def _is_api_error(
    error: Exception | None, statuses: frozenset[int], codes: frozenset[str]
) -> TypeGuard[ApiError]:
    return isinstance(error, ApiError) and (
        error.status_code in statuses
        or (isinstance(error.body, Error) and error.body.code in codes)
    )


def _describe_request_error(body: Error) -> str:
    message = body.message.removesuffix(".")
    param = body.param
    if param is None or not param.startswith("/"):
        return f"{message}."
    # param is a JSON Pointer, so ~1 stands for / and ~0 for ~ in a question ID.
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in param[1:].split("/")
    ]
    return f"{'.'.join(tokens)}: {message}."


def _durations(stretch: Stretch) -> list[int]:
    return [part.end_ms - part.start_ms for part in _parts(stretch)]


def _parts(stretch: Stretch) -> Iterator[AudioChunk]:
    # iter_create_chunked cuts with the same defaults, so these are the parts it
    # sends. iter_audio_chunks reads the whole file before it returns.
    with stretch.path.open("rb") as file:
        return iter_audio_chunks(
            file,
            max_chunk_ms=MAX_PART_MS,
            start_ms=stretch.start_ms,
            end_ms=stretch.end_ms,
        )
