"""Draws the monitor in a real headless browser. Skipped unless playwright and a Chromium are installed."""
import glob
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("autogen_agentchat")
sync_api = pytest.importorskip("playwright.sync_api")

from pvcrew import config  # noqa: E402
from pvcrew.coding import load_term_counts  # noqa: E402
from pvcrew.crew.agents import AGENT_NAMES, CrewTools  # noqa: E402
from pvcrew.crew.events import EventBus  # noqa: E402
from pvcrew.crew.monitor import MonitorServer  # noqa: E402
from pvcrew.crew.team import run_crew  # noqa: E402
from tests.crewfakes import REPORT, REVISE, APPROVE, FakeSession, RoleClient  # noqa: E402

HOSTILE = REPORT + ' <img src=x onerror="window.__pwned=1"> <script>window.__pwned=2</script>'


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = None
        try:
            b = p.chromium.launch(args=["--no-sandbox"])
        except Exception:  # noqa: BLE001 - fall back to any Chromium that is on the disk
            for exe in sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")):
                try:
                    b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
                    break
                except Exception:  # noqa: BLE001
                    continue
        if b is None:
            pytest.skip("no Chromium available for playwright")
        yield b
        b.close()


def crew_run(bus, report, client):
    """Playwright keeps an event loop running in the test thread and run_crew starts its own, so run it in a worker
    thread, which is also how the Streamlit page runs it."""
    counts = load_term_counts(config.SAMPLE_TERMS)
    tools = CrewTools(bus=bus, session=FakeSession(), client=client, terms=list(counts), counts=counts, max_revisions=1)
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(run_crew, report, tools).result(timeout=60)


@pytest.fixture
def page(browser):
    pg = browser.new_page(viewport={"width": 1300, "height": 860})
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    yield pg
    pg.close()


def open_monitor(page, bus):
    srv = MonitorServer(bus, port=0)
    url = srv.start()
    page.goto(url)
    return srv


def speakers(page):
    return page.locator("#messages .row.in:not(.typing) .who").all_inner_texts()


def test_the_group_chat_shows_each_agent_in_order_and_the_diagram_is_drawn(page):
    bus = EventBus()
    res = crew_run(bus, REPORT, RoleClient())
    srv = open_monitor(page, bus)
    try:
        page.wait_for_function("document.querySelector('#statustext').textContent.toLowerCase().includes('done') "
                               "|| document.querySelector('#statustext').textContent.toLowerCase().includes('finished')", timeout=30000)
        assert [s for s in speakers(page) if s in AGENT_NAMES] == AGENT_NAMES
        assert page.locator("#diagram .node").count() >= 7  # report, manager, four agents, reviewer
        for name in AGENT_NAMES:
            assert page.locator(f"#n-{name}").count() == 1
        assert page.locator("#timeline > div").count() > 5 and page.locator("#raw [data-row]").count() == 5  # user + 4 agents
        assert "Intake" in page.locator("#messages").inner_text() and res["narrative"][:30] in page.locator("#messages").inner_text()
        assert page.errors == []
    finally:
        srv.stop()


def test_a_revision_shows_intake_speaking_twice(page):
    bus = EventBus()
    crew_run(bus, REPORT, RoleClient(critic=[REVISE, APPROVE]))
    srv = open_monitor(page, bus)
    try:
        page.wait_for_function("document.querySelectorAll('#messages .row.in:not(.typing) .who').length >= 7", timeout=30000)
        assert [s for s in speakers(page) if s in AGENT_NAMES] == ["Intake", "Coding", "Critic", "Intake", "Coding", "Critic", "Narrative"]
        assert page.errors == []
    finally:
        srv.stop()


def test_report_and_model_text_is_shown_as_text_and_never_runs(page):
    bus = EventBus()
    crew_run(bus, HOSTILE, RoleClient())
    srv = open_monitor(page, bus)
    try:
        page.wait_for_function("document.querySelectorAll('#messages .row.in:not(.typing) .who').length >= 4", timeout=30000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.__pwned") is None
        assert page.locator("#messages img, #raw img, #timeline img").count() == 0
        assert page.locator("body").inner_text().count("onerror") >= 1  # visible as plain characters
        assert page.errors == []
    finally:
        srv.stop()


def test_the_page_picks_up_a_run_that_starts_after_it_opened(page):
    bus = EventBus()
    srv = open_monitor(page, bus)
    try:
        page.wait_for_selector("#empty", timeout=10000)
        crew_run(bus, REPORT, RoleClient())  # the Streamlit page starting a run
        page.wait_for_function("document.querySelectorAll('#messages .row.in:not(.typing) .who').length >= 4", timeout=30000)
        assert page.locator("#empty").count() == 0 or not page.locator("#empty").is_visible()
    finally:
        srv.stop()
