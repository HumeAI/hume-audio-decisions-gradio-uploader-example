import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr
from hume_audio_decisions import AsyncDecisionsClient

from evaluation import Stretch, cut, evaluate, make_client
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
# Theme variables, so the styles follow the light and dark themes.
SELECTED_STYLE = (
    "background-color: var(--color-accent-soft); "
    "box-shadow: inset 0 2px var(--color-accent), inset 0 -2px var(--color-accent)"
)
# Gradio's dark error text color is for text on a red background, but its error
# icon color is red in both themes.
FAILED_STYLE = "color: var(--error-icon-color)"
NUMBER_STYLE = "text-align: right"
STATUS_COLUMN = COLUMNS.index("Status")


class Session:
    def __init__(self) -> None:
        self.results = Results()
        self.stop = asyncio.Event()
        self.shown: list[Row] = []
        self.selected: tuple[Path, int | None] | None = None
        self.chose_row = False
        self.downloads = Path(tempfile.mkdtemp())
        self.clips: dict[Stretch, Path] = {}

    def close(self) -> None:
        shutil.rmtree(self.downloads, ignore_errors=True)


def build(client: AsyncDecisionsClient) -> gr.Blocks:
    # Gradio's temporary files include the uploads, which can be sensitive, so they
    # are deleted once they are a day old.
    demo = gr.Blocks(
        title=TITLE, delete_cache=(86400, 86400), fill_height=True, fill_width=True
    )
    with demo:
        session = gr.State(Session, delete_callback=Session.close)

        with gr.Sidebar(width=420):
            recordings = gr.File(
                file_count="multiple",
                file_types=[".mp3", ".wav"],
                label="Recordings",
                height=276,
                elem_id="recordings",
            )
            editor = gr.Code(
                value=DEFAULT_REQUEST,
                language="json",
                label="Request",
                wrap_lines=True,
                elem_id="request",
            )
            # Stop takes Run's place during a run, because a button ignores clicks
            # while the event it started is still going.
            run_button = gr.Button("Run", variant="primary")
            stop_button = gr.Button("Stop", variant="stop", visible=False)
            clear_button = gr.Button("Clear results", interactive=False)

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
        with gr.Row(equal_height=True):
            with gr.Column():
                heading = gr.Markdown(NO_SELECTION)
                audio = gr.Audio(label="Audio", interactive=False)
            response = gr.JSON(label="Response", max_height=240)

        recordings.change(
            _set_recordings,
            inputs=[recordings, session],
            outputs=[status, table, heading, audio, response],
            show_progress="hidden",
            api_visibility="private",
        )
        # The status line and the table show a run's progress, so Gradio's own
        # progress border around every output is hidden.
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
                heading,
                audio,
                response,
            ],
            validator=_check,
            show_progress="hidden",
            api_visibility="private",
        )
        stop_button.click(
            _stop,
            inputs=session,
            outputs=stop_button,
            show_progress="hidden",
            api_visibility="private",
        )
        clear_button.click(
            _clear,
            inputs=session,
            outputs=[
                status,
                table,
                clear_button,
                csv_button,
                json_button,
                heading,
                audio,
                response,
            ],
            show_progress="hidden",
            api_visibility="private",
        )
        table.select(
            _select,
            inputs=session,
            outputs=[table, heading, audio, response],
            show_progress_on=audio,
            api_visibility="private",
        )
    return demo


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _set_recordings(paths: list[str] | None, session: Session) -> tuple[Any, ...]:
    recordings = [Path(path) for path in paths or []]
    session.results.set_recordings(recordings)
    selection: tuple[Any, ...] = (gr.skip(), gr.skip(), gr.skip())
    if session.selected is not None and session.selected[0] not in recordings:
        session.selected = None
        selection = (NO_SELECTION, None, None)
    return session.results.status(), _table(session), *selection


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
    session.chose_row = False

    def locked() -> tuple[Any, ...]:
        return (
            gr.File(interactive=False),
            gr.Code(interactive=False),
            gr.Button(visible=False),
            gr.Button("Stop", interactive=True, visible=True),
            gr.Button(interactive=False),
            gr.DownloadButton(interactive=False),
            gr.DownloadButton(interactive=False),
            results.status(),
            _table(session),
            gr.skip(),
            gr.skip(),
            gr.skip(),
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
        gr.Button(visible=True),
        gr.Button(visible=False),
        gr.Button(interactive=results.has_results()),
        gr.DownloadButton(value=csv_path, interactive=True),
        gr.DownloadButton(value=json_path, interactive=True),
        results.status(),
        _table(session),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    # A row picked during the run stays selected, so its playback isn't cut off.
    rows = results.rows()
    if rows and not session.chose_row:
        selection = await _show(session, rows[0])
        if not session.chose_row:
            yield *[gr.skip()] * 8, _table(session), *selection


# Gradio runs plain functions in worker threads, and an asyncio.Event must be set
# on its own event loop.
async def _stop(session: Session) -> gr.Button:
    session.stop.set()
    return gr.Button("Stopping", interactive=False)


def _clear(session: Session) -> tuple[Any, ...]:
    session.results.clear()
    session.selected = None
    return (
        session.results.status(),
        _table(session),
        gr.Button(interactive=False),
        gr.DownloadButton(value=None, interactive=False),
        gr.DownloadButton(value=None, interactive=False),
        NO_SELECTION,
        None,
        None,
    )


async def _select(session: Session, event: gr.SelectData) -> tuple[Any, ...]:
    session.chose_row = True
    shown = await _show(session, session.shown[event.index[0]])
    return _table(session), *shown


async def _show(session: Session, row: Row) -> tuple[Any, ...]:
    session.selected = _key(row)
    name, span, *_ = row.cells
    # The player gets the part on its own, so its times match the response's, which
    # count from the start of the part.
    audio = row.recording if row.audio is None else await _clip(session, row.audio)
    return (
        f"### Selected part: {name}, {span}" if span else f"### Selected part: {name}",
        str(audio),
        row.detail,
    )


async def _clip(session: Session, stretch: Stretch) -> Path:
    if stretch not in session.clips:
        session.clips[stretch] = await asyncio.to_thread(
            _write_clip, stretch, session.downloads
        )
    return session.clips[stretch]


def _write_clip(stretch: Stretch, directory: Path) -> Path:
    start, end = max(stretch.start_ms, 0) // 1000, (stretch.end_ms or 0) // 1000
    name = f"{stretch.path.stem}-{start}s-to-{end}s{stretch.path.suffix}"
    # Each clip gets its own folder, so recordings with the same name can't collide.
    path = Path(tempfile.mkdtemp(dir=directory)) / name
    path.write_bytes(cut(stretch))
    return path


def _table(session: Session) -> dict[str, Any]:
    # A row is selected by its index in the table, so the rows on screen are kept.
    session.shown = session.results.rows()
    numeric = session.results.numeric_columns()
    return {
        "headers": session.results.headers(),
        "data": [list(row.cells) for row in session.shown],
        "metadata": {
            "styling": [_styles(session, row, numeric) for row in session.shown]
        },
    }


def _styles(session: Session, row: Row, numeric: list[bool]) -> list[str]:
    styles = []
    for column in range(len(row.cells)):
        style = [SELECTED_STYLE] if _key(row) == session.selected else []
        if numeric[column]:
            style.append(NUMBER_STYLE)
        if row.failed and column == STATUS_COLUMN:
            style.append(FAILED_STYLE)
        styles.append("; ".join(style))
    return styles


def _key(row: Row) -> tuple[Path, int | None]:
    return row.recording, None if row.part is None else row.part.start_ms


def main() -> None:
    # Every tab shares the organization's rate limit, so the whole process shares
    # one client and its pacer.
    client = make_client()
    build(client).launch(max_file_size="1gb", css=_asset("app.css"))


if __name__ == "__main__":
    main()
