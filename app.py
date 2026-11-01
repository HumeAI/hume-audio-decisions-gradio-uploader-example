import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr
from hume_audio_decisions import AsyncDecisionsClient

from evaluation import evaluate, make_client
from questions import DEFAULT_REQUEST, parse_request
from results import COLUMNS, Results, Row

ASSETS = Path(__file__).parent / "assets"
TITLE = "Evaluating many recordings"
INTRODUCTION = (
    "Ask the same questions about many recordings and compare every answer in one "
    "table. See the [Audio Decisions docs]"
    "(https://dev.hume.ai/audio-decisions/docs/overview)."
)
NO_SELECTION = (
    "### Selected part\n\nSelect a row to hear its part and see its response."
)


class Session:
    def __init__(self) -> None:
        self.results = Results()
        self.stop = asyncio.Event()
        self.shown: list[Row] = []
        self.downloads = Path(tempfile.mkdtemp())

    def close(self) -> None:
        shutil.rmtree(self.downloads, ignore_errors=True)


def build(client: AsyncDecisionsClient) -> gr.Blocks:
    # Gradio's temporary files include the uploads, which can be sensitive, so they
    # are deleted once they are a day old.
    demo = gr.Blocks(title=TITLE, delete_cache=(86400, 86400), fill_height=True)
    with demo:
        session = gr.State(Session, delete_callback=Session.close)

        with gr.Sidebar(width=420):
            recordings = gr.File(
                file_count="multiple",
                file_types=[".mp3", ".wav"],
                label="Recordings",
                height=250,
            )
            editor = gr.Code(
                value=DEFAULT_REQUEST,
                language="json",
                label="Request",
                wrap_lines=True,
                max_lines=16,
            )
            run_button = gr.Button("Run", variant="primary")
            with gr.Row():
                stop_button = gr.Button("Stop", variant="stop", interactive=False)
                clear_button = gr.Button("Clear results")

        with gr.Row():
            gr.Markdown(f"# {TITLE}\n\n{INTRODUCTION}")
            # gr.HTML takes an equal share of a row whatever its scale.
            with gr.Column(scale=0, min_width=124):
                gr.HTML(
                    _asset("header.html"),
                    css_template=_asset("header.css"),
                    js_on_load=_asset("header.js"),
                )
        with gr.Row(equal_height=True):
            status = gr.Markdown(Results().status())
            csv_button = gr.DownloadButton(
                "Download CSV", size="sm", scale=0, interactive=False
            )
            json_button = gr.DownloadButton(
                "Download JSON", size="sm", scale=0, interactive=False
            )
        table = gr.Dataframe(
            value={"headers": list(COLUMNS), "data": []},
            label="Results",
            interactive=False,
            show_row_numbers=True,
            show_search="search",
            max_height=410,
            scale=1,
        )
        heading = gr.Markdown(NO_SELECTION)
        with gr.Row(equal_height=True):
            audio = gr.Audio(label="Recording", interactive=False)
            response = gr.JSON(label="Response", max_height=200)

        recordings.change(
            _set_recordings,
            inputs=[recordings, session],
            outputs=[status, table],
            api_visibility="private",
        )
        run_button.click(
            partial(_run, client),
            inputs=[recordings, editor, session],
            outputs=[
                recordings,
                editor,
                run_button,
                stop_button,
                clear_button,
                csv_button,
                json_button,
                status,
                table,
            ],
            validator=_check,
            api_visibility="private",
        )
        stop_button.click(_stop, inputs=session, api_visibility="private")
        clear_button.click(
            _clear,
            inputs=session,
            outputs=[status, table, csv_button, json_button, heading, audio, response],
            api_visibility="private",
        )
        table.select(
            _select,
            inputs=session,
            outputs=[heading, audio, response],
            api_visibility="private",
        )
    return demo


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _set_recordings(paths: list[str] | None, session: Session) -> tuple[str, Any]:
    session.results.set_recordings(Path(path) for path in paths or [])
    return session.results.status(), _table(session)


def _check(paths: list[str] | None, text: str, session: Session) -> tuple[Any, ...]:
    try:
        session.results.check_request(parse_request(text))
        request_check = gr.validate(is_valid=True, message="")
    except ValueError as error:
        request_check = gr.validate(is_valid=False, message=str(error))
    recordings_check = gr.validate(
        is_valid=bool(paths), message="Add at least one MP3 or WAV recording."
    )
    return recordings_check, request_check, None


async def _run(
    client: AsyncDecisionsClient, _paths: list[str] | None, text: str, session: Session
) -> AsyncIterator[tuple[Any, ...]]:
    request = parse_request(text)
    results = session.results
    stretches = results.start_run(request)
    session.stop = asyncio.Event()

    def locked() -> tuple[Any, ...]:
        return (
            gr.File(interactive=False),
            gr.Code(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.DownloadButton(interactive=False),
            gr.DownloadButton(interactive=False),
            results.status(),
            _table(session),
        )

    yield locked()
    try:
        async for progress in evaluate(client, request, stretches, session.stop):
            results.record(progress)
            yield locked()
    finally:
        results.end_run()

    csv_path, json_path = await asyncio.to_thread(
        results.write_downloads, session.downloads
    )
    yield (
        gr.File(interactive=True),
        gr.Code(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
        gr.DownloadButton(value=csv_path, interactive=True),
        gr.DownloadButton(value=json_path, interactive=True),
        results.status(),
        _table(session),
    )


# Gradio runs plain functions in worker threads, and an asyncio.Event must be set
# on its own event loop.
async def _stop(session: Session) -> None:
    session.stop.set()


def _clear(session: Session) -> tuple[Any, ...]:
    session.results.clear()
    return (
        session.results.status(),
        _table(session),
        gr.DownloadButton(value=None, interactive=False),
        gr.DownloadButton(value=None, interactive=False),
        NO_SELECTION,
        None,
        None,
    )


def _select(session: Session, event: gr.SelectData) -> tuple[Any, ...]:
    row = session.shown[event.index[0]]
    name, span, *_ = row.cells
    start_ms = 0 if row.part is None else max(row.part.start_ms, 0)
    return (
        f"### Selected part: {name}, {span}" if span else f"### Selected part: {name}",
        gr.Audio(value=str(row.recording), playback_position=start_ms / 1000),
        row.detail,
    )


def _table(session: Session) -> dict[str, Any]:
    # A row is selected by its index in the table, so the rows on screen are kept.
    session.shown = session.results.rows()
    return {
        "headers": session.results.headers(),
        "data": [list(row.cells) for row in session.shown],
    }


def main() -> None:
    # Every tab shares the organization's rate limit, so the whole process shares
    # one client and its pacer.
    client = make_client()
    build(client).launch(max_file_size="1gb")


if __name__ == "__main__":
    main()
