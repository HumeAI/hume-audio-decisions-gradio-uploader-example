import pytest
from hume_audio_decisions import AsyncDecisionsClient

from app import build


def test_no_endpoint_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building the app would otherwise send Gradio's usage analytics, and the tests
    # make no network calls.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build(AsyncDecisionsClient(api_key="test-key"))

    api = demo.get_api_info()

    assert api["named_endpoints"] == {}
    assert api["unnamed_endpoints"] == {}
