import json

import pytest

from pvcrew import coding, config
from pvcrew.coding import code_reactions, exact, fetch_terms, load_terms, shortlist, similarity
from tests.conftest import FakeClient, FakeResponse

TERMS = ["Diarrhoea", "Vomiting", "Nausea", "Oedema peripheral", "Hepatic enzyme increased", "Hiccups",
         "Drug ineffective", "Treatment failure", "Dyspnoea exertional", "Rash", "Swelling"]


def reply(**kw):
    return json.dumps(kw)


def norm_reply(*pairs):
    return reply(items=[{"reported": a, "medical_term": b} for a, b in pairs])


def choose_reply(*pairs):
    return reply(choices=[{"reported": a, "term": b} for a, b in pairs])


# ---------- string matching ----------

def test_word_order_does_not_matter():
    assert exact("peripheral oedema", TERMS) == "Oedema peripheral"
    assert similarity("Peripheral Oedema", "oedema peripheral") == 1.0


def test_exact_ignores_case_and_punctuation():
    assert exact("  DIARRHOEA. ", TERMS) == "Diarrhoea" and exact("the runs", TERMS) is None and exact("", TERMS) is None


def test_shortlist_ranks_the_closest_first_and_drops_noise():
    got = [t for t, _ in shortlist("increased liver enzymes", TERMS, 3)]
    assert got[0] == "Hepatic enzyme increased" and len(got) <= 3
    assert shortlist("zzzz qqqq", TERMS) == []
    assert shortlist("rash", []) == []


def test_similarity_edge_cases():
    assert similarity("", "rash") == 0.0 and similarity("rash", "") == 0.0


# ---------- vocabulary ----------

class CountSession:
    def __init__(self, results, status=200):
        self.results, self.status, self.requests = results, status, []

    def get(self, url, params=None, timeout=None):
        self.requests.append(dict(params))
        return FakeResponse(self.status, {"results": self.results})


def test_fetch_terms_asks_openfda_for_the_count_endpoint():
    s = CountSession([{"term": "NAUSEA", "count": 5}, {"term": " ", "count": 1}, {"term": "Rash", "count": 3}])
    rows = fetch_terms(5000, session=s, api_key="k", say=lambda *_: None)
    assert rows == [{"term": "NAUSEA", "count": 5}, {"term": "Rash", "count": 3}]
    assert s.requests[0] == {"count": coding.COUNT_FIELD, "limit": 1000, "api_key": "k"}  # capped at openFDA's maximum
    s2 = CountSession([{"term": "Rash", "count": 3}])
    fetch_terms(50, session=s2, say=lambda *_: None)
    assert s2.requests[0]["limit"] == 50


def test_fetch_terms_fails_loudly_on_empty_or_http_error():
    with pytest.raises(RuntimeError):
        fetch_terms(10, session=CountSession([]), say=lambda *_: None)
    with pytest.raises(Exception):
        fetch_terms(10, session=CountSession([], status=500), say=lambda *_: None)


class RefusingCountSession:
    """openFDA answers the count query with 403 but serves plain record pages."""

    def __init__(self, pages):
        self.pages, self.requests = pages, []

    def get(self, url, params=None, timeout=None):
        self.requests.append(dict(params))
        if "count" in params:
            return FakeResponse(403)
        return FakeResponse(200, {"results": self.pages.get(params["skip"], [])})


def _rec(*terms):
    return {"patient": {"reaction": [{"reactionmeddrapt": t} for t in terms]}}


def test_403_on_the_count_query_falls_back_to_counting_a_separate_batch_of_records(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)
    pages = {5000: [_rec("Rash", "Nausea"), _rec("Rash")], 5100: [_rec("Rash", "Vomiting")], 5200: [_rec("Nausea")]}
    s = RefusingCountSession(pages)
    lines = []
    rows = fetch_terms(10, session=s, say=lines.append)
    assert rows[0] == {"term": "Rash", "count": 3} and rows[1] == {"term": "Nausea", "count": 2}
    assert {r["term"] for r in rows} == {"Rash", "Nausea", "Vomiting"}
    record_skips = [r["skip"] for r in s.requests if "skip" in r]
    assert record_skips == [5000, 5100, 5200, 5300]  # far from fetch's records at skip 0; stops at the empty page
    assert all(r["limit"] == 100 for r in s.requests if "skip" in r)  # openFDA 403s bigger pages without a key
    assert "refused the count query" in lines[0]
    assert [r.get("limit") for r in s.requests if "count" in r] == [10, None]


def test_fallback_respects_n_and_stops_at_no_matches(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)

    class Stops(RefusingCountSession):
        def get(self, url, params=None, timeout=None):
            if "count" not in params and params["skip"] >= 5100:
                return FakeResponse(404)
            return super().get(url, params, timeout)

    rows = fetch_terms(1, session=Stops({5000: [_rec("Rash", "Rash", "Nausea")]}), say=lambda *_: None)
    assert rows == [{"term": "Rash", "count": 2}]


def test_bundled_sample_terms_load():
    terms = load_terms(config.SAMPLE_TERMS)
    assert len(terms) > 50 and "Diarrhoea" in terms and len(terms) == len(set(terms))


# ---------- code_reactions ----------

def test_exact_matches_need_no_model_call():
    client = FakeClient(["unused"])
    out = code_reactions(["Nausea", "peripheral oedema"], client, "m", TERMS)
    assert [o["term"] for o in out] == ["Nausea", "Oedema peripheral"] and {o["method"] for o in out} == {"exact"}
    assert client.calls == []


def test_lay_wording_is_reworded_then_matched_without_a_second_call():
    client = FakeClient([norm_reply(("the runs", "diarrhoea"), ("throwing up", "vomiting"))])
    out = code_reactions(["the runs", "throwing up"], client, "m", TERMS)
    assert [(o["term"], o["method"]) for o in out] == [("Diarrhoea", "exact"), ("Vomiting", "exact")]
    assert [o["medical"] for o in out] == ["diarrhoea", "vomiting"] and len(client.calls) == 1


def test_model_picks_from_the_shortlist():
    client = FakeClient([norm_reply(("drug not working", "lack of efficacy")),
                         choose_reply(("drug not working", "Drug ineffective"))])
    out = code_reactions(["drug not working"], client, "m", TERMS)
    assert out[0]["term"] == "Drug ineffective" and out[0]["method"] == "model" and len(client.calls) == 2
    sent = json.loads(client.calls[1]["messages"][1]["content"])
    assert sent[0]["reported"] == "drug not working" and "Drug ineffective" in [c["term"] for c in sent[0]["candidates"]]


def test_model_cannot_invent_a_term():
    client = FakeClient([norm_reply(("hurty", "pain")), choose_reply(("hurty", "Chronic Pain Syndrome"))])
    out = code_reactions(["hurty"], client, "m", TERMS)
    assert out[0]["term"] is None and out[0]["method"] == "none"


def test_model_saying_none_is_respected_even_if_a_string_is_close():
    client = FakeClient([norm_reply(("hiccuping", "hiccuping")), choose_reply(("hiccuping", None))])
    assert code_reactions(["hiccuping"], client, "m", TERMS)[0]["term"] is None


def test_unusable_answers_fall_back_to_a_near_identical_string_only():
    client = FakeClient(["not json at all"])
    out = code_reactions(["hiccuping", "the runs"], client, "m", TERMS)
    assert out[0]["term"] == "Hiccups" and out[0]["method"] == "string"
    assert out[1]["term"] is None


def test_answers_are_matched_by_wording_even_if_reordered():
    client = FakeClient([norm_reply(("throwing up", "vomiting"), ("the runs", "diarrhoea"))])
    out = code_reactions(["the runs", "throwing up"], client, "m", TERMS)
    assert [o["term"] for o in out] == ["Diarrhoea", "Vomiting"]


def test_model_level_errors_propagate_for_failover():
    from tests.conftest import FakeAPIError
    from pvcrew.llm import ModelUnavailableError

    class Dead:
        def chat(self, **kw):
            raise ModelUnavailableError("gone", status=404)

    with pytest.raises(ModelUnavailableError):
        code_reactions(["the runs"], Dead(), "m", TERMS)


def test_empty_vocabulary_and_empty_reactions():
    client = FakeClient([norm_reply(("x", "y")), choose_reply(("x", None))])
    assert code_reactions([], client, "m", TERMS) == []
    assert code_reactions(["x"], client, "m", [])[0]["term"] is None


def test_candidates_come_from_the_medical_wording_too():
    terms = ["Abdominal pain", "Rash"]
    assert shortlist("xq", terms) == []  # the reporter's wording alone finds nothing
    client = FakeClient([norm_reply(("xq", "abdominal pains")), choose_reply(("xq", "Abdominal pain"))])
    out = code_reactions(["xq"], client, "m", terms)
    sent = json.loads(client.calls[1]["messages"][1]["content"])
    assert "Abdominal pain" in [c["term"] for c in sent[0]["candidates"]] and sent[0]["medical_wording"] == "abdominal pains"
    assert out[0]["term"] == "Abdominal pain" and out[0]["method"] == "model"


class LadderSession:
    """The count query is refused above `max_ok`; plain record pages are served."""

    def __init__(self, max_ok, results, pages=None):
        self.max_ok, self.results, self.pages, self.requests = max_ok, results, pages or {}, []

    def get(self, url, params=None, timeout=None):
        self.requests.append(dict(params))
        if "count" in params:
            lim = params.get("limit")
            return FakeResponse(200, {"results": self.results}) if lim and lim <= self.max_ok else FakeResponse(403)
        return FakeResponse(200, {"results": self.pages.get(params["skip"], [])})


def test_smaller_count_limits_are_tried_when_the_big_one_is_refused(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)
    s = LadderSession(250, [{"term": "Rash", "count": 9}])
    rows = fetch_terms(1000, session=s, say=lambda *_: None)
    assert [r.get("limit") for r in s.requests if "count" in r] == [1000, 500, 250]
    assert rows[0] == {"term": "Rash", "count": 9}


def test_short_count_results_are_topped_up_from_records_without_duplicates(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)
    s = LadderSession(1000, [{"term": "Rash", "count": 9}], pages={5000: [_rec("RASH", "Nausea", "Nausea")]})
    lines = []
    rows = fetch_terms(5, session=s, say=lines.append)
    assert [r["term"] for r in rows] == ["Rash", "Nausea"]  # count-endpoint terms first, no case-variant duplicate
    assert "adding terms counted" in lines[0]


def test_topping_up_never_exceeds_n(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)
    s = LadderSession(1000, [{"term": "Rash", "count": 9}], pages={5000: [_rec("A", "B", "C", "D")]})
    assert len(fetch_terms(3, session=s, say=lambda *_: None)) == 3


def test_everything_refused_is_an_error(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)
    with pytest.raises(Exception):
        fetch_terms(10, session=LadderSession(0, [], pages={}), say=lambda *_: None)


def test_a_failing_page_keeps_the_terms_already_read(monkeypatch):
    monkeypatch.setattr(coding.time, "sleep", lambda s: None)

    class Breaks(RefusingCountSession):
        def get(self, url, params=None, timeout=None):
            if "count" not in params and params["skip"] >= 5100:
                return FakeResponse(403)
            return super().get(url, params, timeout)

    lines = []
    rows = fetch_terms(10, session=Breaks({5000: [_rec("Rash", "Nausea")]}), say=lines.append)
    assert {r["term"] for r in rows} == {"Rash", "Nausea"} and any("stopped reading records after 1" in l for l in lines)


def test_candidates_carry_how_common_each_term_is():
    terms = ["Drug ineffective", "Treatment failure"]
    client = FakeClient([norm_reply(("not working", "lack of effect")), choose_reply(("not working", "Drug ineffective"))])
    code_reactions(["not working"], client, "m", terms, counts={"Drug ineffective": 9000, "Treatment failure": 40})
    sent = json.loads(client.calls[1]["messages"][1]["content"])
    assert {c["term"]: c["faers_reports"] for c in sent[0]["candidates"]} == {"Drug ineffective": 9000, "Treatment failure": 40}
    assert "faers_reports" in client.calls[1]["messages"][0]["content"]


def test_term_counts_load_from_the_terms_file(tmp_path):
    from pvcrew.coding import load_term_counts, save_terms
    save_terms(tmp_path / "t.jsonl", [{"term": "Rash", "count": 5}, {"term": "Nausea"}])
    assert load_term_counts(tmp_path / "t.jsonl") == {"Rash": 5, "Nausea": 0}


def test_an_unusable_reply_is_retried_once_with_the_problem_shown():
    client = FakeClient(["oops not json", norm_reply(("the runs", "diarrhoea"))])
    out = code_reactions(["the runs"], client, "m", TERMS)
    assert out[0]["term"] == "Diarrhoea" and len(client.calls) == 2
    last = client.calls[1]["messages"][-1]["content"]
    assert "not valid" in last and client.calls[1]["messages"][-2] == {"role": "assistant", "content": "oops not json"}


def test_two_unusable_replies_give_up_and_say_why():
    client = FakeClient(["nope"])
    out = code_reactions(["xyzzy"], client, "m", TERMS)
    assert out[0]["term"] is None and "rewording step gave no usable answer" in out[0]["note"] and len(client.calls) == 2


def test_reasons_are_recorded_for_uncoded_reactions():
    client = FakeClient([norm_reply(("hurty", "pain")), choose_reply(("hurty", None))])
    assert "no matching candidate" in code_reactions(["hurty"], client, "m", TERMS)[0]["note"]
    assert "no similar term" in code_reactions(["qqq"], FakeClient([norm_reply(("qqq", "zzz"))]), "m", TERMS)[0]["note"]
    ok = code_reactions(["Nausea"], FakeClient(["unused"]), "m", TERMS)[0]
    assert ok["note"] == ""


def test_choosing_step_failure_is_noted():
    client = FakeClient([norm_reply(("drug not working", "lack of efficacy")), "garbage"])
    out = code_reactions(["drug not working"], client, "m", TERMS)
    assert "choosing step gave no usable answer" in out[0]["note"]


def test_answers_with_respelled_wording_are_still_matched():
    # the model wrote "joints swollen and painful" for "joints swollen/painful", and merged nothing
    client = FakeClient([norm_reply(("joints swollen and painful", "arthralgia"), ("the runs", "diarrhoea"))])
    out = code_reactions(["joints swollen/painful", "the runs"], client, "m", TERMS + ["Arthralgia"])
    assert [o["term"] for o in out] == ["Arthralgia", "Diarrhoea"]


def test_same_count_but_unrelated_wording_is_matched_by_position():
    client = FakeClient([norm_reply(("zzz1", "vomiting"), ("zzz2", "diarrhoea"))])
    out = code_reactions(["throwing up", "the runs"], client, "m", TERMS)
    assert [o["term"] for o in out] == ["Vomiting", "Diarrhoea"]


def test_a_different_count_with_one_answer_matched_by_close_wording_only():
    client = FakeClient([norm_reply(("the runs", "diarrhoea"))])  # the model dropped the other reaction
    out = code_reactions(["throwing up", "the runs"], client, "m", TERMS)
    assert out[1]["term"] == "Diarrhoea" and out[0]["medical"] is None


def test_respelled_wording_is_matched_even_when_the_model_returns_extra_items():
    client = FakeClient([norm_reply(("joints swollen and painful", "arthralgia"), ("the runs", "diarrhoea"),
                                    ("something extra", "rash"))])
    out = code_reactions(["joints swollen/painful", "the runs"], client, "m", TERMS + ["Arthralgia"])
    assert [o["term"] for o in out] == ["Arthralgia", "Diarrhoea"]
