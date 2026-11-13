"""Compare models on accuracy:  python -m pvcrew.compare --models A,B,C --limit 50      (or: --auto 3)

Every model reads the same reports (pairs.jsonl) and goes through extraction and coding. Each answer is checked
against the FAERS record the report was written from. Results go to <data-dir>/compare/:
  compare.md / compare.json   the table (the Streamlit page shows it too)
  <model>/extractions.jsonl, coded.jsonl   one folder per model; `python -m pvcrew.score` can open these

Running the same command again continues where it stopped (finished reports are not paid for twice).
Add --fresh to throw the earlier results away and start over.
"""
import argparse
import json
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .extract import extraction_test
from .jsonl import read_jsonl
from .llm import PROBE, PROBE_OPTIONS, _skip_reason, get_client, rank_models
from .pipeline import cmd_code, cmd_extract
from .score import FIELDS, _terms, bucket

ITEMS = FIELDS + ["reactions"]  # the six things checked on every report
BOOTSTRAP_RUNS = 2000


def slug(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model)


# ---------------- scoring one report ----------------
def evaluate(pair: dict, ext: dict | None, coded: dict | None) -> dict:
    """What one model got right on one report. A report the model did not answer (or that was never processed)
    counts as wrong on everything, and its true reactions count as missed."""
    truth = pair["truth"]
    t_rx = _terms(truth.get("reactions"))
    pred = (ext or {}).get("prediction")
    out = {"valid": bool(pred), "nt": len(t_rx), "tp_ext": 0, "np_ext": 0, "tp_cod": 0, "np_cod": 0,
           "ok": {k: False for k in ITEMS}, "rx_ext_exact": False}
    if not pred:
        return out
    for f in FIELDS:
        out["ok"][f] = bucket(f, pred.get(f), truth.get(f)) == "correct"
    p_ext = _terms(pred.get("reactions"))
    out.update(tp_ext=len(p_ext & t_rx), np_ext=len(p_ext), rx_ext_exact=p_ext == t_rx)
    if coded and "reactions_coded" in coded:  # coding finished for this report
        p_cod = _terms(coded["reactions_coded"])
        out.update(tp_cod=len(p_cod & t_rx), np_cod=len(p_cod))
        out["ok"]["reactions"] = p_cod == t_rx
    return out


def _rate(num, den) -> float:
    return num / den if den else 0.0


# ---------------- statistics ----------------
def bootstrap(vectors: dict, runs: int = BOOTSTRAP_RUNS, seed: int = 0) -> dict:
    """Resample the reports with replacement. `vectors` maps a key to one score per report; the same resampled
    reports are used for every key, so differences between models are fair. Returns key -> list of resampled means."""
    rng = random.Random(seed)
    n = len(next(iter(vectors.values())))
    draws = {k: [] for k in vectors}
    for _ in range(runs):
        idx = [rng.randrange(n) for _ in range(n)]
        for k, v in vectors.items():
            draws[k].append(sum(v[i] for i in idx) / n)
    return draws


def interval(values: list, level: float = 0.95) -> tuple:
    xs = sorted(values)
    tail = (1 - level) / 2
    return xs[min(len(xs) - 1, int(tail * len(xs)))], xs[min(len(xs) - 1, int((1 - tail) * len(xs)))]


def summarize(models: list, pairs: list, results: dict, runs: int = BOOTSTRAP_RUNS) -> list:
    """One entry per model, best first. `results[model] = (ext_by_id, coded_by_id, note)`."""
    n = len(pairs)
    table = []
    for m in models:
        ext, coded, note = results[m]
        ev = [evaluate(p, ext.get(p["id"]), coded.get(p["id"])) for p in pairs]
        tp, npred, nt = sum(e["tp_cod"] for e in ev), sum(e["np_cod"] for e in ev), sum(e["nt"] for e in ev)
        done_ext = sum(p["id"] in ext for p in pairs)
        done_cod = sum(p["id"] in coded for p in pairs if (ext.get(p["id"]) or {}).get("prediction"))
        valid = sum(e["valid"] for e in ev)
        finished = done_ext == n and done_cod == valid
        table.append({
            "model": m, "note": note, "reports": n, "processed": done_ext, "complete": finished and not note,
            "valid": valid,
            "item_scores": [sum(e["ok"].values()) / len(ITEMS) for e in ev],
            "whole_scores": [float(all(e["ok"].values())) for e in ev],
            "fields": {k: _rate(sum(e["ok"][k] for e in ev), n) for k in ITEMS},
            "reactions": {"exact_as_extracted": _rate(sum(e["rx_ext_exact"] for e in ev), n),
                          "precision": _rate(tp, npred), "recall": _rate(tp, nt)},
        })
    draws = bootstrap({(t["model"], kind): t[kind + "_scores"] for t in table for kind in ("item", "whole")}, runs)
    for t in table:
        m = t["model"]
        t["items"] = _estimate(t["item_scores"], draws[(m, "item")])
        t["whole"] = _estimate(t["whole_scores"], draws[(m, "whole")])
    table.sort(key=lambda t: (-t["items"]["value"], -t["whole"]["value"], t["model"]))
    top = table[0]["model"]
    for t in table:
        if t["model"] == top:
            t["vs_top"] = None
            continue
        gap = [a - b for a, b in zip(draws[(t["model"], "item")], draws[(top, "item")])]
        lo, hi = interval(gap)
        verdict = "clearly lower" if hi < 0 else "not clearly different"
        t["vs_top"] = {"gap": t["items"]["value"] - table[0]["items"]["value"], "low": lo, "high": hi,
                       "verdict": verdict if t["complete"] else "incomplete, so not compared"}
    for t in table:
        del t["item_scores"], t["whole_scores"]
    return table


def _estimate(scores: list, draws: list) -> dict:
    lo, hi = interval(draws)
    return {"value": sum(scores) / len(scores), "low": lo, "high": hi}


# ---------------- the report ----------------
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _signed(x: float) -> str:
    return f"{100 * x:+.1f}"


def render(table: list, pairs: list, source: Path) -> str:
    written = sum(p["source"] == "llm" for p in pairs)
    made = f"{written} written by a model, {len(pairs) - written} from a template"
    lines = ["# Model comparison", "",
             f"{len(pairs)} reports from `{source}` ({made}). The right answer for each is the FAERS record the report "
             "was written from.", "",
             "## Overall", "",
             "| # | Model | Reports answered | Valid output | Items correct (95% interval) | Whole report correct (95% interval) | Versus the top model |",
             "|---|---|---|---|---|---|---|"]
    for i, t in enumerate(table, 1):
        if t["vs_top"] is None:
            vs = "top"
        else:
            v = t["vs_top"]
            vs = (f"{_signed(v['gap'])} pts ({_signed(v['low'])} to {_signed(v['high'])}): {v['verdict']}"
                  if t["complete"] else v["verdict"])
        flag = "" if t["complete"] else " **(incomplete)**"
        lines.append(f"| {i} | `{t['model']}`{flag} | {t['processed']}/{t['reports']} | {t['valid']}/{t['reports']} | "
                     f"{_pct(t['items']['value'])} ({_pct(t['items']['low'])} to {_pct(t['items']['high'])}) | "
                     f"{_pct(t['whole']['value'])} ({_pct(t['whole']['low'])} to {_pct(t['whole']['high'])}) | {vs} |")
    notes = [f"- `{t['model']}`: {t['note']}" for t in table if t["note"]]
    if notes:
        lines += ["", "Models that stopped early (run the same command again to continue them):"] + notes
    lines += ["", "## By field", "",
              "| Model | Age | Sex | Suspect drug | Serious | Outcome | Reactions, exact set (after coding) | Reactions, exact set (as extracted) | Reaction terms precision | Reaction terms recall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        f, r = t["fields"], t["reactions"]
        lines.append(f"| `{t['model']}` | " + " | ".join(_pct(f[k]) for k in ITEMS) +
                     f" | {_pct(r['exact_as_extracted'])} | {_pct(r['precision'])} | {_pct(r['recall'])} |")
    lines += ["", "## How to read this", "",
              "- **Items correct**: every report is checked on six things (age, sex, suspect drug, serious, outcome, "
              "and the set of reactions after coding to standard terms). This is the share of all those checks that were right.",
              "- **Whole report correct**: all six right on the same report.",
              "- A report the model could not answer, or that was not processed, counts as wrong on everything.",
              "- **95% interval**: from resampling the reports 2,000 times. A wide interval means too few reports to "
              "tell models apart. **Versus the top model** uses the same resampling and says whether the gap is clear.",
              "- The reports are synthetic, so these scores describe this kind of text, not real reporters' wording. "
              "If a model also wrote the reports it is scored on, it may have an edge.",
              "- The extraction prompt was developed while testing with one model, which may favour that model."]
    return "\n".join(lines) + "\n"


# ---------------- choosing and running models ----------------
def pick_models(client, count: int, say=print) -> list:
    """`count` models that work on your key, from different publishers (meta, google, mistralai, ...)."""
    if not hasattr(client, "list_models"):
        raise SystemExit("--auto works with the nim provider only. Name the models with --models.")
    accept, _ = extraction_test(client)
    chosen, publishers, tried = [], set(), 0
    for model in rank_models(client.list_models(), config.NIM_CANDIDATES):
        publisher = model.split("/")[0]
        if publisher in publishers:
            continue
        if tried >= config.NIM_MAX_MODEL_TRIES:
            break
        tried += 1
        say(f"   trying {model} ...")
        try:
            client.chat(model=model, messages=PROBE, options=PROBE_OPTIONS)
            passed = accept(model)
        except Exception as e:  # noqa: BLE001 - classified by _skip_reason
            reason = _skip_reason(e)
            if reason is None:
                raise
            say(f"      skipped: {reason}")
            continue
        if not passed:
            say("      skipped: answered, but failed the extraction test")
            continue
        say("      ok")
        chosen.append(model)
        publishers.add(publisher)
        if len(chosen) == count:
            break
    if not chosen:
        raise SystemExit("No working model found on your key. List them with `python -m pvcrew.llm` and use --models.")
    return chosen


def _read_by_id(path: Path) -> dict:
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def run_model(model: str, pairs_path: Path, folder: Path, terms: Path, limit, client, fresh: bool) -> str | None:
    """Extract and code with one pinned model. Returns why it stopped early, or None."""
    if fresh and folder.exists():
        shutil.rmtree(folder)
    ext_path, coded_path = folder / "extractions.jsonl", folder / "coded.jsonl"
    print(f"\n=== {model} ===")
    try:
        cmd_extract(pairs_path, ext_path, model, limit, client=client, resume=True)
        cmd_code(ext_path, coded_path, terms, model, limit, client=client, resume=True)
    except SystemExit as e:  # a pinned model that stopped answering
        return str(e).splitlines()[0]
    except Exception as e:  # noqa: BLE001 - one model failing must not lose the others
        return f"{type(e).__name__}: {str(e)[:150]}"
    return None


def compare(models: list, data_dir: Path, limit=None, client=None, fresh: bool = False) -> list:
    data_dir = Path(data_dir)
    pairs_path = data_dir / "pairs.jsonl"
    if not pairs_path.exists():
        raise SystemExit(f"{pairs_path} not found. Make the reports first: `python -m pvcrew.pipeline fetch`, then `synth`.")
    pairs = list(read_jsonl(pairs_path))[:limit]
    if not pairs:
        raise SystemExit(f"{pairs_path} has no reports.")
    terms = next((p for p in (data_dir / "terms.jsonl", config.TERMS_PATH) if p.exists()), None)
    if terms is None:
        raise SystemExit(f"No reaction vocabulary found. Run `python -m pvcrew.pipeline terms` first "
                         "(or `terms --sample` for the small built-in list).")
    out = data_dir / "compare"
    results = {}
    for m in models:
        note = run_model(m, pairs_path, out / slug(m), terms, limit, client, fresh)
        results[m] = (_read_by_id(out / slug(m) / "extractions.jsonl"), _read_by_id(out / slug(m) / "coded.jsonl"), note)
        if note:
            print(f"  stopped early: {note}")
    if not any(ext for ext, _, _ in results.values()):
        raise SystemExit("No model produced any results, see the messages above.")
    table = summarize(models, pairs, results)
    text = render(table, pairs, pairs_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(text, encoding="utf-8")
    payload = {"made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reports": len(pairs),
               "source": str(pairs_path), "models": table}
    (out / "compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + text)
    print(f"Saved to {out / 'compare.md'} (and compare.json, which the Streamlit page reads).")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pvcrew.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR), help="folder with pairs.jsonl (default: data)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--models", help="model ids separated by commas (list yours with `python -m pvcrew.llm instruct`)")
    which.add_argument("--auto", type=int, metavar="N", help="pick N working models from different publishers")
    ap.add_argument("--limit", type=int, help="use only the first N reports")
    ap.add_argument("--fresh", action="store_true", help="discard earlier results of these models and start over")
    args = ap.parse_args(argv)
    client = get_client()
    if args.models:
        models = list(dict.fromkeys(m.strip() for m in args.models.split(",") if m.strip()))
        if not models:
            sys.exit("--models needs at least one model id")
    else:
        if config.env("PV_MODEL"):
            sys.exit("PV_MODEL is set in .env, which pins one model. Remove it for --auto, or name the models with --models.")
        models = pick_models(client, args.auto)
        print("models:", ", ".join(models))
    compare(models, Path(args.data_dir), args.limit, client, args.fresh)


if __name__ == "__main__":
    main()
