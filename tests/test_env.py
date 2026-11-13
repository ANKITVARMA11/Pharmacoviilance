"""The .env file: how it is read, who wins, and that nothing else in the project reads the environment."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pvcrew import config

ROOT = config.ROOT
CLEAN = ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MIN_INTERVAL", "PV_MODEL", "PV_PROVIDER", "PV_DATA",
         "PV_ENV_FILE", "OPENFDA_API_KEY", "OLLAMA_HOST", "PV_TEST_A", "PV_TEST_B", "PV_TEST_C", "PV_TEST_D")


def forget(monkeypatch, *names):
    """Make sure these variables are absent now AND removed again after the test."""
    for n in names:
        monkeypatch.setenv(n, "")
        monkeypatch.delenv(n)


def test_load_env_reads_plain_quoted_exported_spaced_and_commented_lines(tmp_path, monkeypatch):
    forget(monkeypatch, "PV_TEST_A", "PV_TEST_B", "PV_TEST_C", "PV_TEST_D")
    f = tmp_path / ".env"
    f.write_text('# a comment\nPV_TEST_A=plain\nPV_TEST_B="quoted value"\nexport PV_TEST_C=exported\nPV_TEST_D = spaced\n')
    assert config.load_env(f) is True
    assert os.environ["PV_TEST_A"] == "plain"
    assert os.environ["PV_TEST_B"] == "quoted value"
    assert os.environ["PV_TEST_C"] == "exported"
    assert os.environ["PV_TEST_D"] == "spaced"


def test_real_environment_variable_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_TEST_A", "from-shell")
    f = tmp_path / ".env"
    f.write_text("PV_TEST_A=from-file\n")
    config.load_env(f)
    assert os.environ["PV_TEST_A"] == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path):
    assert config.load_env(tmp_path / "nope.env") is False


def test_env_helper_strips_whitespace_and_treats_blank_as_missing(monkeypatch):
    monkeypatch.setenv("PV_TEST_A", "  nvapi-abc  ")
    monkeypatch.setenv("PV_TEST_B", "   ")
    forget(monkeypatch, "PV_TEST_C")
    assert config.env("PV_TEST_A") == "nvapi-abc"
    assert config.env("PV_TEST_B") is None and config.env("PV_TEST_B", "dflt") == "dflt"
    assert config.env("PV_TEST_C", "dflt") == "dflt"


def test_a_blank_key_line_counts_as_not_set(tmp_path, monkeypatch):
    forget(monkeypatch, "PV_TEST_A")
    f = tmp_path / ".env"
    f.write_text("PV_TEST_A=\n")
    config.load_env(f)
    assert config.env("PV_TEST_A") is None


@pytest.mark.parametrize(
    "secret,expected",
    [
        ("nvapi-SECRETSECRETSECRET1234", "nvapi...1234"),
        (None, "(not set)"),
        ("", "(not set)"),
        ("short", "set (short value)"),
    ],
)
def test_mask_never_reveals_the_middle_of_a_key(secret, expected):
    assert config.mask(secret) == expected


def run_config(env_file, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN}
    env["PV_ENV_FILE"] = str(env_file)
    env.update(extra_env or {})
    code = "from pvcrew import config; print(config.NVIDIA_API_KEY, config.NIM_MIN_INTERVAL, config.MODEL, sep='|')"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


def test_fresh_process_picks_up_settings_from_the_env_file_even_with_spaces_in_the_path(tmp_path):
    folder = tmp_path / "Covigilance Assist"
    folder.mkdir()
    f = folder / ".env"
    f.write_text("NVIDIA_API_KEY=nvapi-from-file\nNIM_MIN_INTERVAL=0.25\nPV_MODEL=my/model\n")
    out = run_config(f)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "nvapi-from-file|0.25|my/model"


def test_fresh_process_real_variable_beats_the_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("NVIDIA_API_KEY=nvapi-from-file\n")
    out = run_config(f, {"NVIDIA_API_KEY": "nvapi-from-shell"})
    assert out.stdout.split("|")[0] == "nvapi-from-shell"


def test_fresh_process_with_a_blank_key_has_none():
    out = run_config(Path(__file__).parent / "no-such.env")
    assert out.stdout.split("|")[0] == "None"


def test_only_config_py_reads_the_environment():
    """'Use the .env throughout the project' means one door: everything else goes through config."""
    offenders = []
    for path in (ROOT / "pvcrew").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name != "config.py" and ("os.getenv" in text or "os.environ" in text):
            offenders.append(path.name)
    assert offenders == []


def test_example_env_is_safe_to_share_and_env_is_git_ignored():
    example = (ROOT / ".env.example").read_text()
    assert "NVIDIA_API_KEY=\n" in example and "nvapi-" not in example
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()


def test_every_setting_config_reads_is_documented_in_the_env_template():
    import re

    wanted = set(re.findall(r'env\("([A-Z_]+)"', (ROOT / "pvcrew" / "config.py").read_text()))
    wanted.discard("PV_ENV_FILE")  # test-only switch
    template = (ROOT / ".env.example").read_text()
    missing = {name for name in wanted if name not in template}
    assert missing == set()
