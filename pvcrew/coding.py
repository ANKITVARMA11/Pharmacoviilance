"""Coding step: map the reaction wording a reporter used ("the runs") to a standard term ("Diarrhoea").

Vocabulary: the most common reaction terms in FAERS, pulled from openFDA's public count endpoint
(`python -m pvcrew.pipeline terms`; if openFDA refuses that query, terms are counted from a separate batch of records). The real, licensed MedDRA dictionary would replace this file in production.
The model never invents a term: it can only pick from a short list of candidates, or say "none".

For each reaction, in order:
  1. exact match of the wording (or its medical form) against the vocabulary: no model needed
  2. one model call turns lay wording into medical wording for the rest
  3. a shortlist of similar vocabulary terms is built from both wordings
  4. one model call picks from each shortlist, or null
If a model answer is unusable, the best string match is used only when it is very close, else the term stays uncoded.
"""
import re
import time
from difflib import SequenceMatcher

import requests
from pydantic import BaseModel, ValidationError

from . import config
from .jsonl import read_jsonl, write_jsonl

COUNT_FIELD = "patient.reaction.reactionmeddrapt.exact"
SHORTLIST_SIZE = 8
CLOSE_ENOUGH = 0.9  # string similarity at which a match is trusted without the model


# ---------- vocabulary ----------

def _count_terms(session, base_url, limit, api_key):
    params = {"count": COUNT_FIELD}
    if limit:
        params["limit"] = limit
    if api_key:
        params["api_key"] = api_key
    resp = session.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    return [{"term": r["term"].strip(), "count": r.get("count", 0)} for r in resp.json().get("results", [])
            if r.get("term", "").strip()]


def _terms_from_records(session, base_url, n, api_key, pages=20, page_size=100, start_skip=5000, pause=0.3,
                        say=print):
    """Count the reaction terms in ~2,000 FAERS records. They start at skip 5000, far from the records `fetch`
    downloads (from skip 0), so the test answers do not leak into the vocabulary. Pages of 100, the size `fetch`
    already uses: openFDA answers 403 to bigger pages without an API key. Keeps what it has if a page fails."""
    counts = {}
    for i in range(pages):
        params = {"limit": page_size, "skip": start_skip + i * page_size}
        if api_key:
            params["api_key"] = api_key
        try:
            resp = session.get(base_url, params=params, timeout=60)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
        except requests.RequestException as e:
            say(f"  stopped reading records after {i} pages ({e}).")
            break
        results = resp.json().get("results", [])
        if not results:
            break
        for rec in results:
            for r in (rec.get("patient") or {}).get("reaction", []):
                t = (r.get("reactionmeddrapt") or "").strip()
                if t:
                    counts[t] = counts.get(t, 0) + 1
        if pause and i < pages - 1:
            time.sleep(pause)
    return [{"term": t, "count": c} for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]]


def fetch_terms(n: int = 2000, session=None, api_key=None, base_url: str = config.OPENFDA_URL, say=print) -> list[dict]:
    """Up to n common reaction terms, as [{"term": ..., "count": ...}].

    1. openFDA's count endpoint (global counts). It allows at most 1000 and its default is only 100, so larger limits
       are tried first; some networks and keys get HTTP 403 for them, so smaller ones follow.
    2. If that gives fewer than n terms, the rest come from counting terms in a separate batch of records.
    (A free OPENFDA_API_KEY in .env raises openFDA's limits and may allow the full 1000.)
    """
    session = session or requests.Session()
    rows, last_error = [], None
    for limit in dict.fromkeys([min(n, 1000), 500, 250, None]):
        if limit is not None and limit > n:
            continue
        try:
            rows = _count_terms(session, base_url, limit, api_key)
            break
        except requests.RequestException as e:
            last_error = e
    if last_error is not None and not rows:
        say(f"  openFDA refused the count query ({last_error}).")
    if len(rows) < n:
        say(f"  count query gave {len(rows)} terms; adding terms counted in a separate batch of records...")
        have = {norm(r["term"]) for r in rows}
        extra = _terms_from_records(session, base_url, n, api_key, say=say)
        rows += [r for r in extra if norm(r["term"]) not in have][: n - len(rows)]
    if not rows:
        raise RuntimeError("openFDA returned no reaction terms")
    return rows


def load_terms(path) -> list[str]:
    return [r["term"] for r in read_jsonl(path)]


def load_term_counts(path) -> dict[str, int]:
    """term -> how often it appears in FAERS (0 for the bundled demo list)."""
    return {r["term"]: r.get("count", 0) for r in read_jsonl(path)}


def save_terms(path, rows) -> int:
    return write_jsonl(path, rows)


# ---------- string matching ----------

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _key(text: str) -> str:
    """Word-order-free form: 'peripheral oedema' and 'oedema peripheral' share a key."""
    return " ".join(sorted(norm(text).split()))


def similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if _key(na) == _key(nb):
        return 1.0
    ta, tb = {w[:5] for w in na.split()}, {w[:5] for w in nb.split()}
    jaccard = len(ta & tb) / len(ta | tb)
    return max(SequenceMatcher(None, na, nb).ratio(), SequenceMatcher(None, _key(na), _key(nb)).ratio(),
               0.9 * jaccard)


def shortlist(phrase: str, terms: list[str], k: int = SHORTLIST_SIZE) -> list[tuple[str, float]]:
    scored = sorted(((similarity(phrase, t), i, t) for i, t in enumerate(terms)), key=lambda x: (-x[0], x[1]))
    return [(t, round(s, 3)) for s, _, t in scored[:k] if s > 0.2]


def exact(phrase: str, terms: list[str]):
    key = _key(phrase)
    return next((t for t in terms if _key(t) == key), None) if key else None


# ---------- model calls ----------

class _Norm(BaseModel):
    reported: str
    medical_term: str


class _NormReply(BaseModel):
    items: list[_Norm]


class _Choice(BaseModel):
    reported: str
    term: str | None = None


class _ChoiceReply(BaseModel):
    choices: list[_Choice]


NORMALIZE_SYSTEM = (
    "You convert informal or lay descriptions of adverse reactions into standard medical terminology, the style of "
    "MedDRA preferred terms (British spelling: oedema, diarrhoea, anaemia). One medical term per reported reaction; "
    "do not add or merge reactions. Reply with JSON only."
)
CHOOSE_SYSTEM = (
    "For each reported reaction, choose the one candidate term that means the same thing, or null if no candidate "
    "does. A term that is only related, broader or narrower in a way that changes the meaning is not a match. "
    "Choose only from the candidates given. When two candidates mean the same thing, prefer the one with the larger "
    "faers_reports count: it is the standard term. Reply with JSON only."
)


def ask_json(client, model, system, user, reply_model):
    """Ask once; if the reply is not valid JSON for the schema, show the model the problem and ask once more.
    Returns None if it is still unusable (callers then fall back to string matching). Model-level errors propagate."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        resp = client.chat(model=model, format=reply_model.model_json_schema(), messages=messages,
                           options={"temperature": 0, "timeout": config.NIM_TIMEOUT})
        content = resp["message"]["content"]
        try:
            return reply_model.model_validate_json(content)
        except (ValidationError, ValueError) as e:
            messages = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": f"That was not valid ({str(e)[:200]}). Reply with the corrected JSON only."}]
    return None


def _by_reported(items, reported):
    """Match each reaction we asked about to the model's answer for it: same wording first, then close wording,
    then by position when the counts agree. The model sometimes re-spells what it was given."""
    out, used = [None] * len(reported), set()
    lookup = {norm(getattr(i, "reported", "")): i for i in items}
    for k, r in enumerate(reported):
        it = lookup.get(norm(r))
        if it is not None and id(it) not in used:
            out[k] = it
            used.add(id(it))
    for k, r in enumerate(reported):
        if out[k] is None:
            best = max((i for i in items if id(i) not in used), key=lambda i: similarity(i.reported, r), default=None)
            if best is not None and similarity(best.reported, r) >= 0.6:
                out[k] = best
                used.add(id(best))
    if len(items) == len(reported):
        spare = iter(i for i in items if id(i) not in used)
        out = [o if o is not None else next(spare, None) for o in out]
    return out


def code_reactions(reactions: list[str], client, model: str, terms: list[str], counts: dict | None = None) -> list[dict]:
    """One dict per reaction: reported, medical (model's rewording), term (coded or None), method."""
    out = [{"reported": r, "medical": None, "term": None, "method": "none", "note": ""} for r in reactions]
    todo = []
    for o in out:
        hit = exact(o["reported"], terms)
        if hit:
            o.update(term=hit, method="exact")
        else:
            todo.append(o)
    if not todo:
        return out

    reply = ask_json(client, model, NORMALIZE_SYSTEM, "Reactions:\n" + "\n".join(f"- {o['reported']}" for o in todo), _NormReply)
    if reply:
        for o, item in zip(todo, _by_reported(reply.items, [o["reported"] for o in todo])):
            if item and item.medical_term.strip():
                o["medical"] = item.medical_term.strip()
    else:
        for o in todo:
            o["note"] = "the rewording step gave no usable answer"
    pending = []
    for o in todo:
        hit = exact(o["medical"], terms) if o["medical"] else None
        if hit:
            o.update(term=hit, method="exact")
        else:
            pending.append(o)
    if not pending:
        return out

    cands = {}
    for o in pending:
        merged = {}
        for phrase in (o["reported"], o["medical"]):
            for t, s in (shortlist(phrase, terms) if phrase else []):
                merged[t] = max(s, merged.get(t, 0))
        cands[o["reported"]] = sorted(merged, key=lambda t: -merged[t])[: SHORTLIST_SIZE + 4]
    counts = counts or {}
    ask = [{"reported": o["reported"], "medical_wording": o["medical"],
            "candidates": [{"term": t, "faers_reports": counts.get(t, 0)} for t in cands[o["reported"]]]}
           for o in pending if cands[o["reported"]]]
    picks = {}
    if ask:
        import json
        reply = ask_json(client, model, CHOOSE_SYSTEM, json.dumps(ask, ensure_ascii=False), _ChoiceReply)
        if reply:
            for o, c in zip(pending, _by_reported(reply.choices, [o["reported"] for o in pending])):
                if c:  # the model answered for this one (a term, null, or something not on the list)
                    picks[o["reported"]] = next((t for t in cands[o["reported"]] if norm(t) == norm(c.term or "")), None)
                    o["note"] = "" if picks[o["reported"]] else "the model found no matching candidate"
        else:
            for o in pending:
                o["note"] = "the choosing step gave no usable answer"
    for o in pending:
        if not cands[o["reported"]] and not o["note"]:
            o["note"] = "no similar term in the vocabulary"
    for o in pending:
        t = picks.get(o["reported"])
        if t:
            o.update(term=t, method="model")
            continue
        if o["reported"] in picks:  # the model looked and said none: respect that
            continue
        best = shortlist(o["medical"] or o["reported"], terms, 1)  # no usable answer: only a near-identical string
        if best and best[0][1] >= CLOSE_ENOUGH:
            o.update(term=best[0][0], method="string", note="")
    return out
