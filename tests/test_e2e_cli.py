"""The real command line, run as a subprocess, configured ONLY by a .env file, talking to a fake NIM server.

This is as close to your own machine as a test can get: fresh Python process, real openai SDK, real HTTP,
and a project folder with a space in its name (like C:\\projects\\Covigilance Assist).
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("openai")

from pvcrew import config  # noqa: E402
from pvcrew.jsonl import read_jsonl  # noqa: E402
from tests.fake_nim import FakeNimServer  # noqa: E402

ROOT = config.ROOT
CLEAN = ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MIN_INTERVAL", "PV_MODEL", "PV_PROVIDER", "PV_DATA",
         "PV_ENV_FILE", "OPENFDA_API_KEY", "OLLAMA_HOST")
KEY = "nvapi-from-dotenv-file-1234"
FIRST, SECOND = config.NIM_CANDIDATES[:2]


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "Covigilance Assist"
    folder.mkdir()
    return folder


@pytest.fixture
def server():
    servers = []

    def start(**kw):
        s = FakeNimServer(**kw).start()
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.stop()


def write_env(project, base_url=None, key=KEY, extra=""):
    lines = [f"NVIDIA_API_KEY={key}", "NIM_MIN_INTERVAL=0"]
    if base_url:
        lines.append(f"NIM_BASE_URL={base_url}")
    path = project / ".env"
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


def run(args, env_file, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN}
    env.update(PV_ENV_FILE=str(env_file), NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               PYTHONIOENCODING="utf-8")
    # Safety net: anything that is not localhost goes to a dead proxy, so a bug can never send a (fake) key
    # to the real NVIDIA servers from a test. The loopback fake server is exempt via NO_PROXY.
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        env[var] = "http://127.0.0.1:9"
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=120)


def test_the_safety_net_blocks_anything_but_localhost(project):
    """If .env were ignored, the client would aim at NVIDIA's real URL. It must fail here, not connect."""
    env_file = project / ".env"
    env_file.write_text("NVIDIA_API_KEY=nvapi-fake-key-for-test\nNIM_MIN_INTERVAL=0\n", encoding="utf-8")  # no NIM_BASE_URL
    out = run(["pvcrew.check"], env_file)
    assert out.returncode == 1 and "FAIL" in out.stdout and "OK:" not in out.stdout


def test_check_command_works_from_the_env_file_alone(project, server):
    srv = server(models=[FIRST, SECOND], retired=[FIRST])  # first preferred model is listed but retired
    out = run(["pvcrew.check"], write_env(project, srv.base_url))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "OK:" in out.stdout and f"model chosen: {SECOND}" in out.stdout
    assert "nvapi...1234" in out.stdout and KEY not in out.stdout + out.stderr
    assert {r["auth"] for r in srv.requests} == {f"Bearer {KEY}"}  # the key came from the file


def test_full_pipeline_run_with_spaces_in_the_paths(project, server):
    srv = server(models=[FIRST, SECOND], retired=[FIRST])
    env_file = write_env(project, srv.base_url)
    data = project / "my data"
    out = run(["pvcrew.pipeline", "--data-dir", str(data), "all", "--sample", "--limit", "3"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"model: {SECOND}" in out.stdout
    rows = list(read_jsonl(data / "extractions.jsonl"))
    assert len(rows) == 3 and all(r["prediction"] and r["error"] is None for r in rows)
    assert {r["auth"] for r in srv.requests} == {f"Bearer {KEY}"}
    assert all(r["body"]["model"] in (FIRST, SECOND) for r in srv.chat_requests)
    assert all("nvext" in r["body"] for r in srv.chat_requests if "nvext" in r["body"])


def test_resume_after_a_stop_does_not_redo_work(project, server):
    srv = server(models=[SECOND])
    env_file = write_env(project, srv.base_url)
    data = project / "data"
    d = ["pvcrew.pipeline", "--data-dir", str(data)]
    assert run(d + ["fetch", "--sample"], env_file).returncode == 0
    assert run(d + ["synth", "--offline"], env_file).returncode == 0
    assert run(d + ["extract", "--limit", "2"], env_file).returncode == 0
    before = len(srv.chat_requests)
    out = run(d + ["extract", "--resume"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    assert len(srv.chat_requests) - before == 2 + 3  # model probe + extraction test + the 3 reports not yet done
    assert len(list(read_jsonl(data / "extractions.jsonl"))) == 5


def test_real_environment_variable_beats_the_env_file(project, server):
    srv = server(models=[SECOND])
    out = run(["pvcrew.check"], write_env(project, srv.base_url), {"NVIDIA_API_KEY": "nvapi-from-real-env-9999"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert {r["auth"] for r in srv.requests} == {"Bearer nvapi-from-real-env-9999"}


def test_blank_key_fails_fast_with_instructions_and_makes_no_network_call(project):
    env_file = project / ".env"
    env_file.write_text("NVIDIA_API_KEY=\n", encoding="utf-8")
    out = run(["pvcrew.check"], env_file)
    assert out.returncode == 1
    assert "NVIDIA_API_KEY is not set" in out.stdout and str(env_file) in out.stdout and "build.nvidia.com" in out.stdout


def test_bad_key_is_explained(project, server):
    srv = server(models=[SECOND], script=[(401, {"error": "bad key"})])
    out = run(["pvcrew.check"], write_env(project, srv.base_url, key="nvapi-wrong-key-0000"))
    assert out.returncode == 1
    assert "rejected the key" in out.stdout and "nvapi-wrong-key-0000" not in out.stdout


def test_pinned_model_in_env_file_is_used_without_probing(project, server):
    srv = server(models=[SECOND])
    env_file = write_env(project, srv.base_url, extra="PV_MODEL=my/pinned-model\n")
    out = run(["pvcrew.check"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    assert {r["body"]["model"] for r in srv.chat_requests} == {"my/pinned-model"}


def test_check_discovers_a_working_model_when_every_preferred_one_is_dead(project, server):
    dead = list(config.NIM_CANDIDATES)
    srv = server(models=dead + ["acme/shiny-12b-instruct", "acme/shiny-embed-1b"], retired=dead)
    out = run(["pvcrew.check"], write_env(project, srv.base_url))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "model chosen: acme/shiny-12b-instruct" in out.stdout
    assert "PV_MODEL=acme/shiny-12b-instruct" in out.stdout  # the tip to pin it
    assert "acme/shiny-embed-1b" not in {r["body"]["model"] for r in srv.chat_requests}  # not a chat model


def test_check_failure_lists_every_model_that_was_tried_and_why(project, server):
    dead = list(config.NIM_CANDIDATES) + ["acme/other-it"]
    srv = server(models=dead, retired=dead)
    out = run(["pvcrew.check"], write_env(project, srv.base_url))
    assert out.returncode == 1
    for m in dead:
        assert f"{m}: retired or unavailable (HTTP 410)" in out.stdout
    assert "PV_MODEL" in out.stdout


def test_pipeline_extract_discovers_a_model_too(project, server):
    dead = list(config.NIM_CANDIDATES)
    srv = server(models=dead + ["acme/shiny-12b-instruct"], retired=dead)
    env_file = write_env(project, srv.base_url)
    d = ["pvcrew.pipeline", "--data-dir", str(project / "data")]
    assert run(d + ["fetch", "--sample"], env_file).returncode == 0
    assert run(d + ["synth", "--offline"], env_file).returncode == 0
    out = run(d + ["extract", "--limit", "2"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "model: acme/shiny-12b-instruct" in out.stdout and "skipped: retired or unavailable" in out.stdout


def test_terms_code_and_score_run_end_to_end(project, server):
    srv = server(models=[SECOND])
    env_file = write_env(project, srv.base_url)
    d = ["pvcrew.pipeline", "--data-dir", str(project / "my data")]
    for cmd in (["fetch", "--sample"], ["synth", "--offline"], ["extract", "--limit", "3"], ["terms", "--sample"]):
        assert run(d + cmd, env_file).returncode == 0
    out = run(d + ["code"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    rows = list(read_jsonl(project / "my data" / "coded.jsonl"))
    assert len(rows) == 3 and all("reactions_coded" in r for r in rows)
    scored = run(["pvcrew.score", str(project / "my data" / "coded.jsonl")], env_file)
    assert scored.returncode == 0 and "after coding" in scored.stdout and "as extracted" in scored.stdout


def test_code_without_a_terms_file_explains_what_to_do(project, server):
    srv = server(models=[SECOND])
    env_file = write_env(project, srv.base_url)
    d = ["pvcrew.pipeline", "--data-dir", str(project / "data")]
    run(d + ["fetch", "--sample"], env_file); run(d + ["synth", "--offline"], env_file)
    run(d + ["extract", "--limit", "2"], env_file)
    out = run(d + ["code"], env_file)
    assert out.returncode != 0 and "terms" in (out.stdout + out.stderr)


# ---------- model comparison, through the real command line ----------

def _prepare_eval_data(project, env_file):
    d = ["pvcrew.pipeline", "--data-dir", str(project / "my eval")]
    assert run(d + ["fetch", "--sample"], env_file).returncode == 0
    assert run(d + ["synth", "--offline"], env_file).returncode == 0
    assert run(d + ["terms", "--sample"], env_file).returncode == 0
    return project / "my eval"


def test_compare_models_from_the_command_line(project, server):
    other = "meta/llama-3.1-8b-instruct"
    srv = server(models=[SECOND, other])
    env_file = write_env(project, srv.base_url)
    data = _prepare_eval_data(project, env_file)
    before = len(srv.chat_requests)
    out = run(["pvcrew.compare", "--data-dir", str(data), "--models", f"{SECOND},{other}", "--limit", "3"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    asked = srv.chat_requests[before:]
    assert {r["body"]["model"] for r in asked} == {SECOND, other}  # exactly the named models, no search for others
    assert not any(r["body"]["messages"][0]["content"].startswith("Reply with") for r in asked)  # no probe calls
    assert {r["auth"] for r in asked} == {f"Bearer {KEY}"} and KEY not in out.stdout + out.stderr
    md = (data / "compare" / "compare.md").read_text(encoding="utf-8")
    assert f"`{SECOND}`" in md and f"`{other}`" in md and "3 reports" in md
    assert (data / "compare" / "compare.json").exists()
    assert len(list(read_jsonl(data / "compare" / "meta_llama-3.1-8b-instruct" / "coded.jsonl"))) == 3
    # running the same command again must not spend a single call
    before = len(srv.chat_requests)
    again = run(["pvcrew.compare", "--data-dir", str(data), "--models", f"{SECOND},{other}", "--limit", "3"], env_file)
    assert again.returncode == 0, again.stdout + again.stderr
    assert len(srv.chat_requests) == before


def test_compare_auto_picks_working_models_from_different_publishers(project, server):
    other = "meta/llama-3.1-8b-instruct"
    srv = server(models=[FIRST, SECOND, other], retired=[FIRST])
    env_file = write_env(project, srv.base_url)
    data = _prepare_eval_data(project, env_file)
    out = run(["pvcrew.compare", "--data-dir", str(data), "--auto", "2", "--limit", "2"], env_file)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"models: {SECOND}, {other}" in out.stdout  # the retired one was skipped; one model per publisher
    assert "skipped" in out.stdout


def test_compare_without_reports_says_what_to_run_first(project, server):
    srv = server(models=[SECOND])
    out = run(["pvcrew.compare", "--data-dir", str(project / "empty"), "--models", SECOND], write_env(project, srv.base_url))
    assert out.returncode != 0 and "pairs.jsonl not found" in out.stdout + out.stderr
    assert srv.chat_requests == []
