"""The model comparison, driven by fake models whose answers are known."""
import json

import pytest

from pvcrew import compare, config
from pvcrew.jsonl import read_jsonl, write_jsonl
from tests.conftest import FakeAPIError

N = 20
VOCAB_RX = ["Dyspnoea", "Rash"]  # both are in the bundled vocabulary, so coding needs no model call


def truth_for(i):
    return {"patient_age": 20 + i, "patient_sex": "female" if i % 2 else "male", "suspect_drug": "Amoxicillin",
            "reactions": list(VOCAB_RX), "serious": bool(i % 2), "outcome": "recovered"}


def make_pairs(n=N, source="llm"):
    return [{"id": f"r{i}", "style": "plain", "source": source, "report": f"Report number {i}", "truth": truth_for(i)}
            for i in range(n)]


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    write_jsonl(d / "pairs.jsonl", make_pairs())
    config.SAMPLE_TERMS  # the bundled vocabulary is the fallback the tests use
    return d


@pytest.fixture(autouse=True)
def _vocabulary(monkeypatch):
    monkeypatch.setattr(config, "TERMS_PATH", config.SAMPLE_TERMS)


class ModelClient:
    """chat(model=...) answers by model. behaviour[model](i, truth) -> dict (an answer), str (raw text) or an
    exception to raise; i is the report number. Counts the calls each model got."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = {m: 0 for m in behaviour}

    def list_models(self):
        return list(self.behaviour)

    def chat(self, model, messages, format=None, options=None, **_):
        self.calls[model] += 1
        text = messages[-1]["content"]
        if len(messages) == 1:  # the one-word probe
            return {"message": {"content": "OK"}}
        i = int(text.rsplit("number ", 1)[1].split()[0]) if "Report number" in text else 0
        out = self.behaviour[model](i, truth_for(i))
        if isinstance(out, Exception):
            raise out
        return {"message": {"content": out if isinstance(out, str) else json.dumps(out)}}


def perfect(i, t):
    return t


def wrong_sex_on_even(i, t):
    return {**t, "patient_sex": "unknown"} if i % 2 == 0 else t


def junk(i, t):
    return "not json"


# ---------------- scoring one report ----------------
def evaluate(pred=None, coded=None, truth=None):
    pair = {"truth": truth or truth_for(1)}
    ext = {"prediction": pred} if pred is not None else None
    return compare.evaluate(pair, ext, coded)


def test_a_perfect_answer_gets_every_item():
    t = truth_for(1)
    e = evaluate(t, {"reactions_coded": ["Dyspnoea", "Rash"]})
    assert e["valid"] and all(e["ok"].values()) and e["rx_ext_exact"]
    assert (e["tp_cod"], e["np_cod"], e["nt"]) == (2, 2, 2)


def test_one_wrong_field_costs_exactly_that_item():
    e = evaluate({**truth_for(1), "outcome": "fatal"}, {"reactions_coded": VOCAB_RX})
    assert [k for k, v in e["ok"].items() if not v] == ["outcome"]


def test_an_unanswered_report_is_wrong_everywhere_and_its_reactions_are_missed():
    for e in (evaluate(None, None), compare.evaluate({"truth": truth_for(1)}, {"prediction": None}, None)):
        assert not e["valid"] and not any(e["ok"].values())
        assert (e["tp_cod"], e["np_cod"], e["nt"]) == (0, 0, 2)


def test_reactions_not_coded_yet_do_not_count_as_right_but_fields_do():
    e = evaluate(truth_for(1), None)
    assert e["ok"]["patient_age"] and not e["ok"]["reactions"]
    assert e["rx_ext_exact"] and e["np_cod"] == 0


def test_reaction_term_counts_ignore_case_and_count_extras():
    e = evaluate({**truth_for(1), "reactions": ["dyspnoea", "Nausea"]}, {"reactions_coded": ["DYSPNOEA", "Nausea", "Rash"]})
    assert (e["tp_ext"], e["np_ext"], e["tp_cod"], e["np_cod"]) == (1, 2, 2, 3)
    assert not e["ok"]["reactions"] and not e["rx_ext_exact"]


def test_fields_follow_the_scorers_matching_rules():
    t = {**truth_for(1), "suspect_drug": "DURAGESIC-100", "patient_age": None, "patient_sex": "unknown"}
    e = evaluate({**t, "suspect_drug": "Duragesic", "patient_sex": "Unknown"}, {"reactions_coded": VOCAB_RX}, truth=t)
    assert e["ok"]["suspect_drug"] and e["ok"]["patient_sex"] and e["ok"]["patient_age"]
    wrong = evaluate({**t, "suspect_drug": "Aspirin", "patient_age": 40}, {"reactions_coded": VOCAB_RX}, truth=t)
    assert not wrong["ok"]["suspect_drug"] and not wrong["ok"]["patient_age"]  # a made-up age is wrong, not "unknown"


def test_the_reactions_item_is_judged_after_coding_not_before():
    lay = {**truth_for(1), "reactions": ["trouble breathing", "skin rash"]}
    fixed = evaluate(lay, {"reactions_coded": VOCAB_RX})
    assert fixed["ok"]["reactions"] and not fixed["rx_ext_exact"]  # wording was wrong, coding put it right
    broken = evaluate(truth_for(1), {"reactions_coded": ["Dyspnoea"]})
    assert not broken["ok"]["reactions"] and broken["rx_ext_exact"]  # extraction was right, coding lost a term


# ---------------- statistics ----------------
def test_bootstrap_is_repeatable_and_uses_the_same_reports_for_every_key():
    v = {"a": [1.0, 0.0, 1.0, 1.0, 0.0] * 4, "b": [1.0, 0.0, 1.0, 1.0, 0.0] * 4, "c": [0.0] * 20}
    one, two = compare.bootstrap(v, runs=200), compare.bootstrap(v, runs=200)
    assert one == two
    assert one["a"] == one["b"]  # identical scores, identical resamples
    assert set(one["c"]) == {0.0}


def test_interval_of_a_constant_is_that_constant():
    assert compare.interval([0.5] * 100) == (0.5, 0.5)
    lo, hi = compare.interval(list(range(1000)))
    assert 24 <= lo <= 26 and 974 <= hi <= 976


def run_summary(behaviours, n=N, notes=None):
    """Summarize models whose per-report outcomes are given directly: behaviours[model](i) -> prediction dict."""
    pairs = make_pairs(n)
    results = {}
    for m, fn in behaviours.items():
        ext, coded = {}, {}
        for p in pairs:
            pred = fn(int(p["id"][1:]))
            ext[p["id"]] = {"id": p["id"], "prediction": pred}
            if pred:
                coded[p["id"]] = {"id": p["id"], "reactions_coded": list(pred["reactions"])}
        results[m] = (ext, coded, (notes or {}).get(m))
    return compare.summarize(list(behaviours), pairs, results, runs=300)


def test_numbers_match_a_hand_count():
    # model "a": all right except sex on 5 of 20 reports; model "b": nothing valid
    a = lambda i: {**truth_for(i), "patient_sex": "unknown"} if i < 5 else truth_for(i)  # noqa: E731
    table = run_summary({"a": a, "b": lambda i: None})
    ta, tb = table
    assert ta["model"] == "a" and tb["model"] == "b"
    assert ta["fields"]["patient_sex"] == 15 / 20 and ta["fields"]["patient_age"] == 1.0
    assert ta["items"]["value"] == pytest.approx((5 * 5 + 15 * 6) / (20 * 6))  # 5 reports have 5 right, 15 have 6
    assert ta["whole"]["value"] == 15 / 20
    assert ta["valid"] == 20 and tb["valid"] == 0
    assert tb["items"]["value"] == 0 and tb["whole"]["value"] == 0
    assert ta["reactions"] == {"exact_as_extracted": 1.0, "precision": 1.0, "recall": 1.0}
    assert tb["reactions"] == {"exact_as_extracted": 0.0, "precision": 0.0, "recall": 0.0}


def test_precision_and_recall_are_over_all_terms():
    extra = lambda i: {**truth_for(i), "reactions": ["Dyspnoea", "Nausea", "Vomiting"]}  # 1 right of 3; 1 of 2 found  # noqa: E731
    t = run_summary({"m": extra})[0]
    assert t["reactions"]["precision"] == pytest.approx(1 / 3)
    assert t["reactions"]["recall"] == pytest.approx(1 / 2)


def test_ranking_and_the_versus_top_verdict():
    good = truth_for
    bad = lambda i: {**truth_for(i), "patient_sex": "unknown", "outcome": "fatal", "patient_age": None}  # noqa: E731
    table = run_summary({"bad": bad, "good": good, "same": good})
    assert [t["model"] for t in table][:2] == ["good", "same"] and table[2]["model"] == "bad"  # ties keep name order
    assert table[0]["vs_top"] is None
    assert table[1]["vs_top"]["verdict"] == "not clearly different" and table[1]["vs_top"]["gap"] == 0
    assert table[2]["vs_top"]["verdict"] == "clearly lower" and table[2]["vs_top"]["high"] < 0
    for t in table:
        assert t["items"]["low"] <= t["items"]["value"] <= t["items"]["high"]


def test_a_small_gap_on_few_reports_is_not_called_clear():
    slightly_worse = lambda i: {**truth_for(i), "outcome": "fatal"} if i == 0 else truth_for(i)  # noqa: E731
    table = run_summary({"top": truth_for, "b": slightly_worse}, n=10)
    assert table[1]["vs_top"]["verdict"] == "not clearly different"


def test_unprocessed_reports_count_as_wrong_and_the_model_is_flagged():
    pairs = make_pairs(10)
    ext = {p["id"]: {"id": p["id"], "prediction": truth_for(int(p["id"][1:]))} for p in pairs[:4]}
    coded = {k: {"id": k, "reactions_coded": VOCAB_RX} for k in ext}
    t = compare.summarize(["m"], pairs, {"m": (ext, coded, "m: too slow or erroring")}, runs=50)[0]
    assert (t["processed"], t["valid"], t["complete"]) == (4, 4, False)
    assert t["whole"]["value"] == 0.4


def test_a_model_with_unprocessed_reports_is_incomplete_even_without_an_error():
    pairs = make_pairs(10)
    ext = {p["id"]: {"id": p["id"], "prediction": truth_for(int(p["id"][1:]))} for p in pairs[:4]}
    coded = {k: {"id": k, "reactions_coded": VOCAB_RX} for k in ext}
    t = compare.summarize(["m"], pairs, {"m": (ext, coded, None)}, runs=50)[0]
    assert t["processed"] == 4 and not t["complete"]
    full = {p["id"]: {"id": p["id"], "prediction": truth_for(int(p["id"][1:]))} for p in pairs}
    done = compare.summarize(["m"], pairs, {"m": (full, {k: {"id": k, "reactions_coded": VOCAB_RX} for k in full}, None)}, runs=50)[0]
    assert done["complete"]


def test_all_done_but_coding_missing_is_incomplete():
    pairs = make_pairs(3)
    ext = {p["id"]: {"id": p["id"], "prediction": truth_for(int(p["id"][1:]))} for p in pairs}
    t = compare.summarize(["m"], pairs, {"m": (ext, {}, None)}, runs=50)[0]
    assert t["processed"] == 3 and not t["complete"]


# ---------------- the whole run ----------------
def models_client(**extra):
    return ModelClient({"vendor/perfect": perfect, "other/half": wrong_sex_on_even, "third/junk": junk, **extra})


def test_end_to_end_ranks_models_and_writes_the_files(data):
    client = models_client()
    table = compare.compare(["vendor/perfect", "other/half", "third/junk"], data, client=client)
    assert [t["model"] for t in table] == ["vendor/perfect", "other/half", "third/junk"]
    top, half, bad = table
    assert top["whole"]["value"] == 1.0 and top["items"]["value"] == 1.0 and top["complete"]
    assert half["fields"]["patient_sex"] == 0.5 and half["whole"]["value"] == 0.5
    assert bad["valid"] == 0 and bad["whole"]["value"] == 0.0
    assert half["vs_top"]["verdict"] == "clearly lower" and bad["vs_top"]["verdict"] == "clearly lower"
    out = data / "compare"
    saved = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert saved["reports"] == N and [m["model"] for m in saved["models"]] == [t["model"] for t in table]
    md = (out / "compare.md").read_text(encoding="utf-8")
    assert "`vendor/perfect`" in md and "100.0%" in md and "20 written by a model, 0 from a template" in md
    for m in ("vendor_perfect", "other_half", "third_junk"):
        assert len(list(read_jsonl(out / m / "extractions.jsonl"))) == N
    assert len(list(read_jsonl(out / "vendor_perfect" / "coded.jsonl"))) == N


def test_every_model_gets_the_same_reports_and_is_pinned(data):
    client = models_client()
    compare.compare(["vendor/perfect", "other/half"], data, limit=7, client=client)
    assert client.calls["vendor/perfect"] == 7 and client.calls["other/half"] == 7  # no probe calls, no failover
    assert len(list(read_jsonl(data / "compare" / "vendor_perfect" / "extractions.jsonl"))) == 7
    saved = json.loads((data / "compare" / "compare.json").read_text(encoding="utf-8"))
    assert saved["reports"] == 7


def test_a_model_that_dies_is_flagged_and_the_others_still_finish(data):
    dies = {"seen": 0}

    def dies_after_five(i, t):
        dies["seen"] += 1
        return t if dies["seen"] <= 5 else FakeAPIError(404)

    client = models_client(**{"gone/dies": dies_after_five})
    table = compare.compare(["gone/dies", "vendor/perfect"], data, client=client)
    by = {t["model"]: t for t in table}
    assert by["vendor/perfect"]["complete"]
    d = by["gone/dies"]
    assert not d["complete"] and d["processed"] == 5 and d["valid"] == 5 and "gone/dies" in d["note"]
    assert d["fields"]["patient_age"] == 5 / N  # what it did answer is scored, the rest counts as wrong
    md = (data / "compare" / "compare.md").read_text(encoding="utf-8")
    assert "(incomplete)" in md and "incomplete, so not compared" in md
    assert d["vs_top"]["verdict"] == "incomplete, so not compared"  # not "clearly lower": it did not finish, which is a different thing
    assert "clearly lower" not in md
    assert "pts" not in md  # no gap figures for a model that did not finish


def test_a_crash_in_one_model_does_not_lose_the_others(data):
    client = models_client(**{"bad/crash": lambda i, t: RuntimeError("boom")})
    table = compare.compare(["bad/crash", "vendor/perfect"], data, client=client)
    by = {t["model"]: t for t in table}
    assert "RuntimeError: boom" in by["bad/crash"]["note"] and by["vendor/perfect"]["complete"]


def test_running_again_continues_and_never_pays_for_a_report_twice(data):
    seen = {"n": 0}

    def flaky(i, t):
        seen["n"] += 1
        return t if seen["n"] <= 8 else FakeAPIError(404)

    client = models_client(**{"gone/flaky": flaky})
    compare.compare(["gone/flaky"], data, client=client)
    assert client.calls["gone/flaky"] == 9  # 8 answered, the 9th hit the dead model
    client.behaviour["gone/flaky"] = perfect  # it recovers
    table = compare.compare(["gone/flaky"], data, client=client)
    assert client.calls["gone/flaky"] == 9 + (N - 8)  # only the missing reports were asked
    assert table[0]["complete"] and table[0]["whole"]["value"] == 1.0


def test_a_finished_run_repeated_costs_nothing(data):
    client = models_client()
    compare.compare(["vendor/perfect"], data, client=client)
    before = dict(client.calls)
    compare.compare(["vendor/perfect"], data, client=client)
    assert client.calls == before


def test_fresh_starts_over(data):
    client = models_client()
    compare.compare(["vendor/perfect"], data, client=client)
    compare.compare(["vendor/perfect"], data, client=client, fresh=True)
    assert client.calls["vendor/perfect"] == 2 * N


def test_a_bigger_limit_later_adds_only_the_new_reports(data):
    client = models_client()
    compare.compare(["vendor/perfect"], data, limit=5, client=client)
    table = compare.compare(["vendor/perfect"], data, limit=12, client=client)
    assert client.calls["vendor/perfect"] == 12 and table[0]["reports"] == 12


def test_missing_inputs_give_clear_messages(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="pairs.jsonl not found"):
        compare.compare(["m"], tmp_path / "nothing", client=ModelClient({"m": perfect}))
    d = tmp_path / "d"
    write_jsonl(d / "pairs.jsonl", make_pairs(2))
    monkeypatch.setattr(config, "TERMS_PATH", tmp_path / "no-terms.jsonl")
    with pytest.raises(SystemExit, match="vocabulary"):
        compare.compare(["m"], d, client=ModelClient({"m": perfect}))
    write_jsonl(tmp_path / "e" / "pairs.jsonl", [])
    with pytest.raises(SystemExit, match="no reports"):
        compare.compare(["m"], tmp_path / "e", client=ModelClient({"m": perfect}))


def test_vocabulary_next_to_the_reports_wins_over_the_default(data, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TERMS_PATH", tmp_path / "elsewhere.jsonl")  # missing
    write_jsonl(data / "terms.jsonl", [{"term": t, "count": 1} for t in ("Dyspnoea", "Rash")])
    table = compare.compare(["vendor/perfect"], data, client=models_client())
    assert table[0]["whole"]["value"] == 1.0


def test_nothing_at_all_is_an_error_not_an_empty_table(data):
    client = ModelClient({"gone/x": lambda i, t: FakeAPIError(404)})
    with pytest.raises(SystemExit, match="No model produced any results"):
        compare.compare(["gone/x"], data, client=client)


def test_template_reports_are_counted_as_such(tmp_path):
    d = tmp_path / "d"
    write_jsonl(d / "pairs.jsonl", make_pairs(4, source="offline"))
    compare.compare(["vendor/perfect"], d, client=models_client())
    assert "0 written by a model, 4 from a template" in (d / "compare" / "compare.md").read_text(encoding="utf-8")


# ---------------- choosing models ----------------
def test_auto_picks_one_working_model_per_publisher():
    client = ModelClient({"meta/llama-3.1-70b-instruct": perfect, "meta/llama-3.1-8b-instruct": perfect,
                          "google/gemma-3-12b-it": perfect, "mistralai/mistral-7b-instruct": perfect,
                          "qwen/qwen2.5-7b-instruct": perfect})
    chosen = compare.pick_models(client, 3, say=lambda *_: None)
    assert len(chosen) == 3 and len({m.split("/")[0] for m in chosen}) == 3
    assert chosen[0] == "google/gemma-3-12b-it"  # the project's preferred model comes first


def test_auto_skips_retired_models_and_models_that_cannot_extract():
    def retired(i, t):
        return FakeAPIError(410)

    client = ModelClient({"google/gemma-3-12b-it": retired, "meta/llama-3.1-8b-instruct": junk,
                          "mistralai/mistral-7b-instruct": perfect, "qwen/qwen2.5-7b-instruct": perfect})
    # the one-word probe passes for every fake; retired ones fail on the extraction test instead
    assert compare.pick_models(client, 2, say=lambda *_: None) == ["mistralai/mistral-7b-instruct", "qwen/qwen2.5-7b-instruct"]


def test_auto_returns_fewer_when_few_work_and_errors_when_none():
    one = ModelClient({"a/m-instruct": perfect, "b/m-instruct": junk})
    assert compare.pick_models(one, 3, say=lambda *_: None) == ["a/m-instruct"]
    with pytest.raises(SystemExit, match="No working model"):
        compare.pick_models(ModelClient({"b/m-instruct": junk}), 2, say=lambda *_: None)


def test_auto_stops_trying_after_the_configured_number_of_models(monkeypatch):
    monkeypatch.setattr(config, "NIM_MAX_MODEL_TRIES", 2)
    client = ModelClient({f"p{i}/m-instruct": junk for i in range(6)})
    with pytest.raises(SystemExit):
        compare.pick_models(client, 2, say=lambda *_: None)
    assert sum(client.calls.values()) <= 2 * 3 + 2  # at most two models probed, a few calls each


def test_auto_needs_a_provider_that_can_list_models():
    with pytest.raises(SystemExit, match="nim provider"):
        compare.pick_models(object(), 2)


# ---------------- command line ----------------
def test_cli_models_are_split_and_deduplicated(data, monkeypatch):
    seen = {}
    monkeypatch.setattr(compare, "get_client", lambda: "client")
    monkeypatch.setattr(compare, "compare", lambda models, d, limit, client, fresh: seen.update(
        models=models, d=d, limit=limit, fresh=fresh))
    compare.main(["--data-dir", str(data), "--models", " a/x, b/y,a/x ,", "--limit", "9", "--fresh"])
    assert seen["models"] == ["a/x", "b/y"] and seen["limit"] == 9 and seen["fresh"] and seen["d"] == data


def test_cli_needs_models_or_auto_but_not_both(data):
    for argv in ([], ["--models", "a/x", "--auto", "2"]):
        with pytest.raises(SystemExit):
            compare.main(["--data-dir", str(data), *argv])


def test_cli_auto_refuses_when_a_model_is_pinned_in_env(data, monkeypatch):
    monkeypatch.setattr(compare, "get_client", lambda: "client")
    monkeypatch.setenv("PV_MODEL", "pinned/model")
    with pytest.raises(SystemExit, match="PV_MODEL"):
        compare.main(["--data-dir", str(data), "--auto", "2"])


def test_cli_auto_runs_the_chosen_models(data, monkeypatch):
    seen = {}
    monkeypatch.setattr(compare, "get_client", lambda: "client")
    monkeypatch.setattr(compare, "pick_models", lambda client, n: ["m/one", "n/two"])
    monkeypatch.setattr(compare, "compare", lambda models, d, limit, client, fresh: seen.update(models=models))
    compare.main(["--data-dir", str(data), "--auto", "2"])
    assert seen["models"] == ["m/one", "n/two"]


def test_slug_is_a_safe_folder_name():
    assert compare.slug("meta/llama-3.1-8b-instruct") == "meta_llama-3.1-8b-instruct"
    assert "/" not in compare.slug("a/b:c d")
