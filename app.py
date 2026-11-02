import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias

import gradio as gr
from gradio.components import Component
from hume_audio_decisions import AsyncDecisionsClient

from evaluation import Stretch, cut, estimate, evaluate, make_client
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
# An empty Markdown line collapses, but one holding a non-breaking space keeps its
# height.
BLANK_LINE = "&nbsp;"
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

# Gradio leaves unchanged every output that a returned dict leaves out.
Updates: TypeAlias = dict[Component, Any]


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


@dataclass(frozen=True)
class Page:
    recordings: gr.File
    editor: gr.Code
    run_button: gr.Button
    stop_button: gr.Button
    estimate_button: gr.Button
    clear_button: gr.Button
    status: gr.Markdown
    estimate_status: gr.Markdown
    csv_button: gr.DownloadButton
    json_button: gr.DownloadButton
    table: gr.Dataframe
    heading: gr.Markdown
    audio: gr.Audio
    response: gr.JSON


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
            # Stop takes Run's place during a run or an estimate, because a button
            # ignores clicks while the event it started is still going.
            with gr.Row():
                run_button = gr.Button("Run", variant="primary")
                stop_button = gr.Button("Stop", variant="stop", visible=False)
                estimate_button = gr.Button("Estimate tokens", interactive=False)
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
        # The estimate gets a line of its own across the page, kept while empty, so
        # showing one doesn't move the table.
        estimate_status = gr.Markdown(BLANK_LINE)
        table = gr.Dataframe(
            headers=list(COLUMNS),
            label="Results",
            interactive=False,
            show_row_numbers=True,
            show_search="search",
            max_height=372,
            scale=1,
        )
        with gr.Row(equal_height=True):
            with gr.Column():
                heading = gr.Markdown(NO_SELECTION)
                audio = gr.Audio(label="Audio", interactive=False)
            response = gr.JSON(label="Response", max_height=240)

        page = Page(
            recordings=recordings,
            editor=editor,
            run_button=run_button,
            stop_button=stop_button,
            estimate_button=estimate_button,
            clear_button=clear_button,
            status=status,
            estimate_status=estimate_status,
            csv_button=csv_button,
            json_button=json_button,
            table=table,
            heading=heading,
            audio=audio,
            response=response,
        )
        recordings.change(
            partial(_set_recordings, page),
            inputs=[recordings, session],
            outputs=[
                status,
                estimate_status,
                table,
                estimate_button,
                heading,
                audio,
                response,
            ],
            show_progress="hidden",
            api_visibility="private",
        )
        # The status line and the table show a run's progress, so Gradio's own
        # progress border around every output is hidden.
        run_button.click(
            partial(_run, client, page),
            inputs=[recordings, editor, session],
            outputs=[
                recordings,
                editor,
                run_button,
                stop_button,
                estimate_button,
                clear_button,
                csv_button,
                json_button,
                status,
                estimate_status,
                table,
                heading,
                audio,
                response,
            ],
            validator=_check,
            show_progress="hidden",
            api_visibility="private",
        )
        estimate_button.click(
            partial(_estimate, client, page),
            inputs=[recordings, editor, session],
            outputs=[
                recordings,
                editor,
                run_button,
                stop_button,
                estimate_button,
                clear_button,
                estimate_status,
            ],
            validator=_check,
            show_progress="hidden",
            api_visibility="private",
        )
        editor.input(
            _clear_estimate,
            inputs=session,
            outputs=estimate_status,
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
            partial(_clear, page),
            inputs=session,
            outputs=[
                status,
                estimate_status,
                table,
                estimate_button,
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


def _set_recordings(page: Page, paths: list[str] | None, session: Session) -> Updates:
    recordings = [Path(path) for path in paths or []]
    session.results.set_recordings(recordings)
    updates: Updates = {
        **_status_and_table(page, session),
        page.estimate_button: gr.Button(interactive=session.results.can_estimate()),
    }
    if session.selected is not None and session.selected[0] not in recordings:
        session.selected = None
        updates |= _no_selection(page)
    return updates


def _check(paths: list[str] | None, text: str, session: Session) -> tuple[Any, ...]:
    try:
        session.results.check_request(parse_request(text))
    except ValueError as error:
        request_check = gr.validate(is_valid=False, message=str(error))
    else:
        request_check = gr.validate(is_valid=True, message="")
    recordings_check = gr.validate(
        is_valid=bool(paths), message="Add at least one MP3 or WAV recording."
    )
    return recordings_check, request_check, None


async def _run(
    client: AsyncDecisionsClient,
    page: Page,
    _paths: list[str] | None,
    text: str,
    session: Session,
) -> AsyncIterator[Updates]:
    request = parse_request(text)
    results = session.results
    stretches = results.start_run(request)
    session.stop = asyncio.Event()
    session.chose_row = False

    yield {
        **_locked(page),
        page.csv_button: gr.DownloadButton(interactive=False),
        page.json_button: gr.DownloadButton(interactive=False),
        **_status_and_table(page, session),
    }
    try:
        async for progress in evaluate(client, request, stretches, session.stop):
            results.record(progress)
            yield _status_and_table(page, session)
    finally:
        results.end_run()

    csv_path, json_path = await asyncio.to_thread(
        results.write_downloads, session.downloads
    )
    yield {
        **_unlocked(page, session),
        page.csv_button: gr.DownloadButton(value=csv_path, interactive=True),
        page.json_button: gr.DownloadButton(value=json_path, interactive=True),
        **_status_and_table(page, session),
    }

    # A row picked during the run stays selected, so its playback isn't cut off.
    rows = results.rows()
    if rows and not session.chose_row:
        heading, audio, response = await _show(session, rows[0])
        if not session.chose_row:
            yield {
                page.table: _table(session),
                page.heading: heading,
                page.audio: audio,
                page.response: response,
            }


async def _estimate(
    client: AsyncDecisionsClient,
    page: Page,
    _paths: list[str] | None,
    text: str,
    session: Session,
) -> AsyncIterator[Updates]:
    request = parse_request(text)
    results = session.results
    stretches = results.start_estimate(request)
    session.stop = asyncio.Event()

    yield {**_locked(page), page.estimate_status: _estimate_line(session)}
    async for progress in estimate(client, request, stretches, session.stop):
        results.record_estimate(progress)
        yield {page.estimate_status: _estimate_line(session)}
    if session.stop.is_set():
        results.clear_estimate()
    else:
        results.finish_estimate()
    yield {**_unlocked(page, session), page.estimate_status: _estimate_line(session)}


def _clear_estimate(session: Session) -> str:
    session.results.clear_estimate()
    return _estimate_line(session)


# Gradio runs plain functions in worker threads, and an asyncio.Event must be set
# on its own event loop.
async def _stop(session: Session) -> gr.Button:
    session.stop.set()
    return gr.Button("Stopping", interactive=False)


def _clear(page: Page, session: Session) -> Updates:
    session.results.clear()
    session.selected = None
    return {
        **_status_and_table(page, session),
        page.estimate_button: gr.Button(interactive=session.results.can_estimate()),
        page.clear_button: gr.Button(interactive=False),
        page.csv_button: gr.DownloadButton(value=None, interactive=False),
        page.json_button: gr.DownloadButton(value=None, interactive=False),
        **_no_selection(page),
    }


# Gradio passes gr.SelectData by reading the handler's type hints, which a partial
# hides, so this handler can't be given the Page and returns its outputs in order.
async def _select(
    session: Session, event: gr.SelectData
) -> tuple[dict[str, Any], str, Path, dict[str, object] | None]:
    session.chose_row = True
    selection = await _show(session, session.shown[event.index[0]])
    return _table(session), *selection


async def _show(
    session: Session, row: Row
) -> tuple[str, Path, dict[str, object] | None]:
    session.selected = _key(row)
    name, span, *_ = row.cells
    title = f"{name}, {span}" if span else name
    # The player gets the part on its own, so its times match the response's, which
    # count from the start of the part.
    audio = row.recording if row.audio is None else await _clip(session, row.audio)
    return f"### Selected part: {title}", audio, row.detail


async def _clip(session: Session, stretch: Stretch) -> Path:
    if stretch not in session.clips:
        session.clips[stretch] = await asyncio.to_thread(
            _write_clip, stretch, session.downloads
        )
    return session.clips[stretch]


def _write_clip(stretch: Stretch, directory: Path) -> Path:
    start, end = stretch.start_ms // 1000, (stretch.end_ms or 0) // 1000
    name = f"{stretch.path.stem}-{start}s-to-{end}s{stretch.path.suffix}"
    # Each clip gets its own folder, so recordings with the same name can't collide.
    path = Path(tempfile.mkdtemp(dir=directory)) / name
    path.write_bytes(cut(stretch))
    return path


def _locked(page: Page) -> Updates:
    return {
        page.recordings: gr.File(interactive=False),
        page.editor: gr.Code(interactive=False),
        page.run_button: gr.Button(visible=False),
        page.stop_button: gr.Button("Stop", interactive=True, visible=True),
        page.estimate_button: gr.Button(interactive=False),
        page.clear_button: gr.Button(interactive=False),
    }


def _unlocked(page: Page, session: Session) -> Updates:
    return {
        page.recordings: gr.File(interactive=True),
        page.editor: gr.Code(interactive=True),
        page.run_button: gr.Button(visible=True),
        page.stop_button: gr.Button(visible=False),
        page.estimate_button: gr.Button(interactive=session.results.can_estimate()),
        page.clear_button: gr.Button(interactive=session.results.has_results()),
    }


def _status_and_table(page: Page, session: Session) -> Updates:
    return {
        page.status: session.results.status(),
        page.estimate_status: _estimate_line(session),
        page.table: _table(session),
    }


def _estimate_line(session: Session) -> str:
    return session.results.estimate_status() or BLANK_LINE


def _no_selection(page: Page) -> Updates:
    return {page.heading: NO_SELECTION, page.audio: None, page.response: None}


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
    selected = _key(row) == session.selected
    styles = []
    for column, number in enumerate(numeric):
        style = [SELECTED_STYLE] if selected else []
        if number:
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
    build(client).launch(max_file_size="1gb", css_paths=ASSETS / "app.css")


if __name__ == "__main__":
    main()
