# Evaluating many recordings

A Gradio example that asks the same questions about many recordings with the [Audio Decisions API](https://dev.hume.ai/audio-decisions/docs/overview) and compares every answer in one table. The [cookbook guide](https://dev.hume.ai/audio-decisions/cookbooks/evaluating-many-recordings) walks through the code.

## Prerequisites

1. Python 3.11 or later. [uv](https://docs.astral.sh/uv/) installs the version in `.python-version` for you.
2. An API key from the [API keys page](https://cloud.hume.ai/settings/organization/api-keys) of the Hume Platform.

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/HumeAI/hume-audio-decisions-gradio-uploader-example.git
   cd hume-audio-decisions-gradio-uploader-example
   ```

2. Install the dependencies with uv:

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set your API key:

   ```bash
   export HUME_API_KEY=your_api_key
   ```

4. Start the app with `uv run app.py`, or `python app.py` in the activated environment, and open [http://localhost:7860](http://localhost:7860).

## Development

1. `uv run ruff format --check` checks formatting, and `uv run ruff format` applies it.
2. `uv run ruff check` lints.
3. `uv run mypy .` type checks.

## License

[MIT](LICENSE)
