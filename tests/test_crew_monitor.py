import json
import re
import urllib.error
import urllib.request

import pytest

from pvcrew.crew import monitor
from pvcrew.crew.events import EventBus
from pvcrew.crew.monitor import MonitorServer, open_window


def get(url, path):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


@pytest.fixture
def served():
    bus = EventBus()
    srv = MonitorServer(bus, port=0)
    url = srv.start()
    yield bus, srv, url
    srv.stop()


def test_it_serves_the_page_on_localhost_only(served):
    bus, srv, url = served
    assert url.startswith("http://127.0.0.1:")
    status, ctype, body = get(url, "/")
    assert status == 200 and ctype.startswith("text/html") and b"AutoGen" in body
    assert get(url, "/index.html")[2] == body


def test_the_feed_returns_events_since_an_id_with_state(served):
    bus, srv, url = served
    bus.new_run()
    for i in range(5):
        bus.emit("system", text=f"e{i}")
    _, ctype, body = get(url, "/api/events?since=0")
    data = json.loads(body)
    assert ctype == "application/json" and [e["text"] for e in data["events"]] == [f"e{i}" for i in range(5)]
    assert data["epoch"] == bus.epoch and data["running"] is True
    assert [e["text"] for e in json.loads(get(url, "/api/events?since=3")[2])["events"]] == ["e3", "e4"]
    assert json.loads(get(url, "/api/events?since=99")[2])["events"] == []
    assert len(json.loads(get(url, "/api/events?since=junk")[2])["events"]) == 5  # bad input: start from zero
    assert len(json.loads(get(url, "/api/events")[2])["events"]) == 5


def test_state_endpoint_and_unknown_paths(served):
    bus, srv, url = served
    assert json.loads(get(url, "/api/state")[2])["epoch"] == bus.epoch
    with pytest.raises(urllib.error.HTTPError) as e:
        get(url, "/etc/passwd")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        get(url, "/../pvcrew/config.py")
    assert e.value.code == 404


def test_it_never_serves_files_from_disk_other_than_its_page(served):
    _, _, url = served
    for p in ("/monitor.html", "/app.py", "/.env", "/data/audit.jsonl"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(url, p)
        assert e.value.code == 404


def test_feed_is_not_cached(served):
    _, _, url = served
    with urllib.request.urlopen(url + "api/events") as r:
        assert r.headers["Cache-Control"] == "no-store"


def test_it_moves_to_the_next_port_when_busy():
    a, b = EventBus(), EventBus()
    first = MonitorServer(a, port=0)
    first.start()
    taken = first.port
    second = MonitorServer(b, port=taken)
    try:
        url = second.start()
        assert second.port == taken + 1 and url.endswith(f":{taken + 1}/")
        # each server shows its own bus
        a.emit("system", text="from a")
        assert json.loads(get(url, "/api/events")[2])["events"] == []
        assert json.loads(get(first.url, "/api/events")[2])["events"][0]["text"] == "from a"
    finally:
        first.stop()
        second.stop()


def test_it_gives_up_with_a_clear_message_when_no_port_is_free(monkeypatch):
    def refuse(*a, **k):
        raise OSError("in use")
    monkeypatch.setattr(monitor, "ThreadingHTTPServer", refuse)
    with pytest.raises(OSError, match="no free port"):
        MonitorServer(EventBus(), port=9000).start()


def test_stop_closes_the_port_and_is_safe_twice():
    srv = MonitorServer(EventBus(), port=0)
    url = srv.start()
    srv.stop()
    srv.stop()
    with pytest.raises(urllib.error.URLError):
        get(url, "/api/state")


def test_the_window_is_opened_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor, "_opened", False)
    monkeypatch.setattr(monitor.webbrowser, "open_new", lambda u: calls.append(u) or True)
    assert open_window("http://x/") is True
    assert open_window("http://x/") is False
    assert calls == ["http://x/"]


def test_a_missing_browser_never_breaks_the_app(monkeypatch):
    def boom(u):
        raise RuntimeError("no browser")
    monkeypatch.setattr(monitor, "_opened", False)
    monkeypatch.setattr(monitor.webbrowser, "open_new", boom)
    assert open_window("http://x/") is False


# ---------- guards on the page itself ----------

PAGE = monitor.PAGE.read_text(encoding="utf-8")


def test_page_does_not_write_report_text_as_html():
    """Report text and model output are untrusted: they must go in as text, never as markup."""
    assert "innerHTML" not in PAGE and "outerHTML" not in PAGE and "document.write" not in PAGE
    assert "insertAdjacentHTML" not in PAGE


def test_page_loads_nothing_from_the_internet():
    urls = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']|url\(\s*["']?([^)"']+)""", PAGE)
    flat = [u for pair in urls for u in pair if u]
    assert not [u for u in flat if re.match(r"(https?:)?//", u)], flat
    assert not re.search(r"""fetch\(\s*["']https?://""", PAGE)


def test_page_reads_the_feed_it_is_served_from():
    assert "api/events" in PAGE and "epoch" in PAGE
