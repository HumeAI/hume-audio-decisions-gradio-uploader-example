import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import httpx
from hume_audio_decisions import AsyncDecisionsClient, DecisionsRequest, Error
from hume_audio_decisions.audio_helpers import ChunkResult, iter_audio_chunks
from hume_audio_decisions.core import ApiError

# Every API key in an organization shares a limit of 60 requests a minute, so this
# leaves room for other work.
REQUESTS_PER_MINUTE = 50
CONCURRENT_RECORDINGS = 4
MAX_PART_MS = 300_000
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60

# These errors come from the key or the request, which every recording shares, so
# every other recording would fail the same way.
_BATCH_STOPPING_STATUSES = frozenset({401, 402, 403})
_BATCH_STOPPING_CODES = frozenset({"message_invalid", "question_invalid"})


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
class BatchStopped:
    error: ApiError


Progress = Started | PartResult | Finished | Unreadable | BatchStopped


def make_client(
    *,
    requests_per_minute: float = REQUESTS_PER_MINUTE,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncDecisionsClient:
    pacer = _Pacer(interval=60 / requests_per_minute)
    http_client = httpx.AsyncClient(
        # The SDK sets its own 60-second timeout only on a client it makes itself.
        timeout=TIMEOUT_SECONDS,
        event_hooks={"request": [pacer]},
        transport=transport,
    )
    return AsyncDecisionsClient(httpx_client=http_client)


async def evaluate(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretches: Iterable[Stretch],
    stop: asyncio.Event,
) -> AsyncIterator[Progress]:
    updates: asyncio.Queue[Progress | None] = asyncio.Queue()

    async def send() -> None:
        try:
            await _send_all(client, request, stretches, stop, updates.put_nowait)
        finally:
            updates.put_nowait(None)

    # The workers run in their own task because a generator that yields inside a
    # TaskGroup lets the group cancel whatever its consumer is awaiting.
    sending = asyncio.create_task(send())
    try:
        while (update := await updates.get()) is not None:
            yield update
        await sending
    finally:
        sending.cancel()


def cut(stretch: Stretch) -> bytes:
    # iter_create_chunked cuts with the same defaults, so a part's stretch comes out
    # as exactly the audio that was sent for it.
    with stretch.path.open("rb") as file:
        chunks = iter_audio_chunks(
            file,
            max_chunk_ms=MAX_PART_MS,
            start_ms=stretch.start_ms,
            end_ms=stretch.end_ms,
        )
        return next(chunks).data


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


async def _send_all(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretches: Iterable[Stretch],
    stop: asyncio.Event,
    report: Callable[[Progress], None],
) -> None:
    remaining = iter(stretches)
    try:
        async with asyncio.TaskGroup() as group:
            watcher = group.create_task(_halt_when_set(stop))
            workers = [
                group.create_task(_send_stretches(client, request, remaining, report))
                for _ in range(CONCURRENT_RECORDINGS)
            ]
            await asyncio.wait(workers)
            watcher.cancel()
    except* _HaltError:
        pass


async def _halt_when_set(stop: asyncio.Event) -> None:
    await stop.wait()
    raise _HaltError


async def _send_stretches(
    client: AsyncDecisionsClient,
    request: DecisionsRequest,
    stretches: Iterator[Stretch],
    report: Callable[[Progress], None],
) -> None:
    for stretch in stretches:
        report(Started(stretch))
        file = await asyncio.to_thread(stretch.path.open, "rb")
        with file:
            parts: AsyncIterator[ChunkResult] = client.decisions.iter_create_chunked(
                request=request,
                file=(stretch.path.name, file),
                max_chunk_ms=MAX_PART_MS,
                start_ms=stretch.start_ms,
                end_ms=stretch.end_ms,
                request_options={"max_retries": MAX_RETRIES},
            )
            try:
                async for part in parts:
                    report(PartResult(stretch, part))
                    if _stops_batch(part.error):
                        report(BatchStopped(part.error))
                        raise _HaltError
            # The SDK raises ValueError only before it sends anything, when it can't
            # split the file.
            except ValueError as error:
                report(Unreadable(stretch, error))
                continue
        report(Finished(stretch))


def _stops_batch(error: Exception | None) -> TypeGuard[ApiError]:
    if not isinstance(error, ApiError):
        return False
    if error.status_code in _BATCH_STOPPING_STATUSES:
        return True
    return isinstance(error.body, Error) and error.body.code in _BATCH_STOPPING_CODES
