import asyncio
import itertools
import json
import time
import wave
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path

import httpx
import pytest
from hume_audio_decisions import AsyncDecisionsClient, DecisionsRequest, Question_Noul
from hume_audio_decisions.core import ApiError

from evaluation import (
    CONCURRENT_RECORDINGS,
    MAX_RETRIES,
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
    cut,
    estimate,
    evaluate,
    make_client,
)

pytestmark = pytest.mark.anyio

REQUEST = DecisionsRequest(
    questions={"speech": Question_Noul(instructions="Is someone speaking?")}
)
SAMPLE_RATE = 8000
FAST = 600_000
# Requests 100 ms apart leave time for the first error to stop the rest.
PACED = 600

Handler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUME_API_KEY", "test-key")


def write_wav(
    path: Path, duration_ms: int, *, channels: int = 1, rate: int = SAMPLE_RATE
) -> Stretch:
    with path.open("wb") as file, wave.open(file, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(rate * duration_ms // 1000 * 2 * channels))
    return Stretch(path)


def answered() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "run_id": "run",
            "answers": {"speech": {"type": "noul", "noul": 0.5}},
            "received": {"input_tokens": 100},
        },
    )


def failed(
    status: int, code: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        json={"type": "error", "code": code, "message": code, "request_id": "req"},
    )


def estimated(tokens: int = 1000) -> httpx.Response:
    return httpx.Response(200, json={"estimated_input_tokens": tokens})


def sent_duration(request: httpx.Request) -> int:
    return int(json.loads(request.content)["audio_duration_ms"])


def sent_file(request: httpx.Request) -> str:
    return request.content.split(b'filename="')[1].split(b'"')[0].decode()


def sent_audio(request: httpx.Request) -> bytes:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    for field in request.content.split(b"--" + boundary):
        headers, _, body = field.partition(b"\r\n\r\n")
        if b'filename="' in headers:
            return body.removesuffix(b"\r\n")
    raise AssertionError("The request has no file.")


def capture(reply: Callable[[], httpx.Response]) -> tuple[Handler, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return reply()

    return handle, requests


def mock_client(handler: Handler, requests_per_minute: float) -> AsyncDecisionsClient:
    return make_client(
        requests_per_minute=requests_per_minute,
        transport=httpx.MockTransport(handler),
    )


async def collect(
    handler: Handler,
    stretches: Sequence[Stretch],
    *,
    requests_per_minute: float = FAST,
    stop: asyncio.Event | None = None,
) -> list[Progress]:
    client = mock_client(handler, requests_per_minute)
    stop = stop or asyncio.Event()
    return [update async for update in evaluate(client, REQUEST, stretches, stop)]


async def collect_estimates(
    handler: Handler,
    stretches: Sequence[Stretch],
    *,
    requests_per_minute: float = FAST,
) -> list[EstimateProgress]:
    client = mock_client(handler, requests_per_minute)
    return [
        update async for update in estimate(client, REQUEST, stretches, asyncio.Event())
    ]


def parts(progress: list[Progress], stretch: Stretch) -> list[PartResult]:
    return [
        update
        for update in progress
        if isinstance(update, PartResult) and update.stretch == stretch
    ]


def api_error(result: PartResult) -> ApiError:
    assert isinstance(result.part.error, ApiError)
    return result.part.error


async def test_sends_a_long_recording_as_even_parts(tmp_path: Path) -> None:
    stretch = write_wav(tmp_path / "long_call.wav", 822_000)

    async def handle(_request: httpx.Request) -> httpx.Response:
        return answered()

    progress = await collect(handle, [stretch])

    assert progress == [Started(stretch), *parts(progress, stretch), Finished(stretch)]
    assert [
        (p.part.start_ms, p.part.end_ms, p.part.status)
        for p in parts(progress, stretch)
    ] == [
        (0, 274_000, "succeeded"),
        (274_000, 548_000, "succeeded"),
        (548_000, 822_000, "succeeded"),
    ]


@pytest.mark.parametrize(
    ("duration_ms", "channels", "rate", "start_ms"),
    [
        pytest.param(822_000, 1, SAMPLE_RATE, 274_000, id="parts of a stretch"),
        pytest.param(20_000, 2, 48_000, 0, id="mixed down and resampled"),
    ],
)
async def test_a_part_cut_again_is_the_audio_that_was_sent(
    tmp_path: Path, duration_ms: int, channels: int, rate: int, start_ms: int
) -> None:
    recording = write_wav(
        tmp_path / "call.wav", duration_ms, channels=channels, rate=rate
    )
    stretch = Stretch(recording.path, start_ms=start_ms)
    sent: list[bytes] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        sent.append(sent_audio(request))
        return answered()

    progress = await collect(handle, [stretch])

    ends = [p.part.end_ms for p in parts(progress, stretch)]
    starts = [stretch.start_ms, *ends[:-1]]
    assert [
        cut(Stretch(recording.path, start, end))
        for start, end in zip(starts, ends, strict=True)
    ] == sent


async def test_sends_only_the_stretch_it_is_given(tmp_path: Path) -> None:
    recording = write_wav(tmp_path / "long_call.wav", 822_000)
    stretch = Stretch(recording.path, start_ms=274_000)

    async def handle(_request: httpx.Request) -> httpx.Response:
        return answered()

    progress = await collect(handle, [stretch])

    assert [(p.part.start_ms, p.part.end_ms) for p in parts(progress, stretch)] == [
        (274_000, 548_000),
        (548_000, 822_000),
    ]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            httpx.Response(401, json={"message": "Invalid API key"}), id="401"
        ),
        pytest.param(failed(402, "payment_required"), id="402"),
        pytest.param(failed(403, "forbidden"), id="403"),
        pytest.param(failed(400, "message_invalid"), id="message_invalid"),
        pytest.param(failed(400, "question_invalid"), id="question_invalid"),
    ],
)
async def test_an_error_every_recording_would_hit_stops_the_batch(
    tmp_path: Path, response: httpx.Response
) -> None:
    stretches = [
        write_wav(tmp_path / f"{index}.wav", 5_000)
        for index in range(CONCURRENT_RECORDINGS + 2)
    ]
    handle, requests = capture(lambda: response)

    progress = await collect(handle, stretches, requests_per_minute=PACED)

    started = [update.stretch for update in progress if isinstance(update, Started)]
    assert started == stretches[:CONCURRENT_RECORDINGS]
    assert len(requests) == 1
    (result,) = [update for update in progress if isinstance(update, PartResult)]
    assert api_error(result).status_code == response.status_code
    assert progress[-1] == BatchStopped(api_error(result))


async def test_an_error_about_one_file_fails_only_its_recording(
    tmp_path: Path,
) -> None:
    bad = write_wav(tmp_path / "bad.wav", 5_000)
    good = write_wav(tmp_path / "good.wav", 5_000)

    async def handle(request: httpx.Request) -> httpx.Response:
        if sent_file(request) == "bad.wav":
            return failed(400, "audio_decode_failed")
        return answered()

    progress = await collect(handle, [bad, good])

    assert [api_error(p).status_code for p in parts(progress, bad)] == [400]
    assert [p.part.status for p in parts(progress, good)] == ["succeeded"]
    assert Finished(bad) in progress
    assert Finished(good) in progress
    assert not any(isinstance(update, BatchStopped) for update in progress)


async def test_a_file_the_sdk_cannot_split_fails_without_a_request(
    tmp_path: Path,
) -> None:
    unreadable = Stretch(tmp_path / "notes.wav")
    unreadable.path.write_bytes(b"These are notes, not audio.")
    good = write_wav(tmp_path / "good.wav", 5_000)
    handle, requests = capture(answered)

    progress = await collect(handle, [unreadable, good])

    assert [sent_file(request) for request in requests] == ["good.wav"]
    (failure,) = [u for u in progress if isinstance(u, Unreadable)]
    assert failure.stretch == unreadable
    assert Finished(unreadable) not in progress
    assert Finished(good) in progress


async def test_stop_cancels_the_request_in_flight_and_keeps_the_answers(
    tmp_path: Path,
) -> None:
    stretch = write_wav(tmp_path / "call.wav", 360_000)
    stop = asyncio.Event()
    cancelled = asyncio.Event()
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return answered()
        stop.set()
        unanswered: asyncio.Future[httpx.Response] = (
            asyncio.get_running_loop().create_future()
        )
        try:
            return await unanswered
        except asyncio.CancelledError:
            cancelled.set()
            raise

    # Without Stop, the request in flight would never return.
    async with asyncio.timeout(5):
        progress = await collect(handle, [stretch], stop=stop)

    assert cancelled.is_set()
    assert len(requests) == len(parts(progress, stretch)) + 1
    assert progress == [Started(stretch), *parts(progress, stretch)]
    assert [(p.part.start_ms, p.part.status) for p in parts(progress, stretch)] == [
        (0, "succeeded")
    ]


async def test_requests_are_paced_including_retries(tmp_path: Path) -> None:
    stretches = [
        write_wav(tmp_path / f"{index}.wav", 5_000)
        for index in range(CONCURRENT_RECORDINGS)
    ]
    sent_at: list[float] = []

    async def handle(_request: httpx.Request) -> httpx.Response:
        sent_at.append(time.monotonic())
        if len(sent_at) == 1:
            return httpx.Response(
                429, headers={"retry-after-ms": "1"}, json={"message": "Slow down"}
            )
        return answered()

    progress = await collect(handle, stretches, requests_per_minute=PACED)

    assert len(sent_at) == CONCURRENT_RECORDINGS + 1
    gaps = [later - earlier for earlier, later in itertools.pairwise(sent_at)]
    # The transport sees each request a little after the pacer lets it go, by an
    # amount that varies from request to request.
    assert min(gaps) >= 60 / PACED - 0.01
    for stretch in stretches:
        assert [p.part.status for p in parts(progress, stretch)] == ["succeeded"]


async def test_a_server_error_is_retried_before_its_part_fails(tmp_path: Path) -> None:
    stretch = write_wav(tmp_path / "call.wav", 5_000)
    handle, requests = capture(
        lambda: failed(503, "server_busy", headers={"retry-after-ms": "1"})
    )

    progress = await collect(handle, [stretch])

    assert len(requests) == MAX_RETRIES + 1
    assert [api_error(p).status_code for p in parts(progress, stretch)] == [503]


async def test_estimates_each_part_of_a_long_recording(tmp_path: Path) -> None:
    stretch = write_wav(tmp_path / "long_call.wav", 822_000)
    handle, requests = capture(lambda: estimated(900))

    progress = await collect_estimates(handle, [stretch])

    assert {request.url.path for request in requests} == {
        "/v0/decisions/file/input_tokens"
    }
    assert [sent_duration(request) for request in requests] == [274_000] * 3
    assert progress == [Estimated(stretch, 274_000, 900)] * 3


async def test_estimates_only_the_stretch_it_is_given(tmp_path: Path) -> None:
    recording = write_wav(tmp_path / "long_call.wav", 822_000)
    stretch = Stretch(recording.path, start_ms=274_000)
    handle, requests = capture(estimated)

    await collect_estimates(handle, [stretch])

    assert [sent_duration(request) for request in requests] == [274_000, 274_000]


async def test_a_part_too_short_to_send_is_counted_without_a_request(
    tmp_path: Path,
) -> None:
    stretch = write_wav(tmp_path / "short.wav", 3_000)
    handle, requests = capture(estimated)

    progress = await collect_estimates(handle, [stretch])

    assert progress == [TooShort(stretch, 3_000)]
    assert requests == []


async def test_an_unreadable_file_is_counted_without_a_request(
    tmp_path: Path,
) -> None:
    unreadable = Stretch(tmp_path / "notes.wav")
    unreadable.path.write_bytes(b"These are notes, not audio.")
    handle, requests = capture(estimated)

    (failure,) = await collect_estimates(handle, [unreadable])

    assert isinstance(failure, Unreadable)
    assert failure.stretch == unreadable
    assert requests == []


async def test_a_rejected_key_stops_the_estimate(tmp_path: Path) -> None:
    stretches = [
        write_wav(tmp_path / f"{index}.wav", 5_000)
        for index in range(CONCURRENT_RECORDINGS + 2)
    ]
    handle, requests = capture(
        lambda: httpx.Response(401, json={"message": "Invalid API key"})
    )

    (stopped,) = await collect_estimates(handle, stretches, requests_per_minute=PACED)

    assert len(requests) == 1
    assert isinstance(stopped, BatchStopped)
    assert isinstance(stopped.error, ApiError)
    assert stopped.error.status_code == httpx.codes.UNAUTHORIZED


async def test_a_timeout_stops_the_estimate(tmp_path: Path) -> None:
    stretches = [write_wav(tmp_path / f"{index}.wav", 5_000) for index in range(2)]

    async def handle(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    (stopped,) = await collect_estimates(handle, stretches, requests_per_minute=PACED)

    assert isinstance(stopped, BatchStopped)
    assert isinstance(stopped.error, httpx.ReadTimeout)
