"""Day 1 pipeline: fetch FAERS -> synthesize reports -> extract cases.

Usage:
  python -m pvcrew.pipeline fetch --n 200            # or: fetch --sample (bundled sample data)
  python -m pvcrew.pipeline fetch --n 60 --skip 2000 # a different slice of FAERS (see "Comparing models" in the README)
  python -m pvcrew.pipeline synth --offline          # or omit --offline to have the LLM write the reports
  python -m pvcrew.pipeline extract [--resume]       # --resume skips ids already in extractions.jsonl
  python -m pvcrew.pipeline terms                    # reaction vocabulary for coding (or: terms --sample)
  python -m pvcrew.pipeline code                     # map reaction wording to standard terms -> coded.jsonl
  python -m pvcrew.pipeline all --sample --offline   # fetch + synth + extract in one go
  (add --data-dir PATH before the subcommand to change where files go)
"""
import argparse
import json
import shutil
from pathlib import Path

from . import config
from .coding import code_reactions, fetch_terms, load_term_counts, load_terms, save_terms
from .extract import ExtractionError, extract_case, extraction_test
from .faers import fetch_records, load_records, save_records
from .jsonl import read_jsonl, write_jsonl
from .llm import ModelUnavailableError, _skip_reason, get_client, resolve_model
from .synth import build_pairs


def cmd_fetch(n: int, out=config.RAW_PATH, sample: bool = False, session=None, start_skip: int = 0) -> int:
    out = Path(out)
    if sample:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.SAMPLE_FAERS, out)
        count = len(load_records(out))
    else:
        records = fetch_records(n, session=session, api_key=config.OPENFDA_API_KEY, start_skip=start_skip)
        count = save_records(out, records)
    print(f"fetch: {count} records -> {out}")
    return count


def cmd_synth(raw=config.RAW_PATH, out=config.PAIRS_PATH, offline: bool = False,
              model: str | None = None, seed: int = 0, client=None) -> int:
    records = load_records(raw)
    if not offline and client is None:
        client = get_client()
    if not offline:
        model = resolve_model(client, model, say=print)
        print(f"model: {model}")
    rows = build_pairs(records, client=None if offline else client, model=model, seed=seed)
    count = write_jsonl(out, rows)
    fallbacks = sum(r["source"] == "offline_fallback" for r in rows)
    print(f"synth: {count} reports -> {out} (llm fallbacks: {fallbacks})")
    return count


class ModelSession:
    """Holds the chosen model and runs calls against it, switching models if it dies mid-run.

    Some hosted models pass a small test call, then 404 or stall on real requests. A 404 is retried once; then,
    unless PV_MODEL pins the model, the next working model is chosen and the call is repeated.
    """

    def __init__(self, client, model=None, say=print):
        self.client, self.say = client, say
        self.pinned = bool(model or config.env("PV_MODEL"))
        self.accept, _ = extraction_test(client)
        self.model = resolve_model(client, model, accept=self.accept, say=say)
        say(f"model: {self.model}")
        self.dead = set()

    def call(self, fn):
        """fn(model) -> result. Retries/fails over on model-level errors; anything else propagates."""
        retried = False
        while True:
            try:
                return fn(self.model)
            except Exception as e:  # noqa: BLE001 - only model-level failures are handled here
                if _skip_reason(e) is None:
                    raise
                if isinstance(e, ModelUnavailableError) and not retried:
                    retried = True
                    continue
                self.dead.add(self.model)
                if self.pinned:
                    raise SystemExit(f"{self.model}: {_skip_reason(e)}\n{e}\nYour finished rows are saved. Remove or "
                                     "change PV_MODEL in .env to let the project pick another model, then run the "
                                     "same command with --resume") from e
                self.say(f"  {self.model} stopped answering real requests ({_skip_reason(e)}); looking for another model...")
                self.model = resolve_model(self.client, None, accept=self.accept, say=self.say, exclude=self.dead)
                self.say(f"model: {self.model}")
                retried = False


def cmd_extract(pairs=config.PAIRS_PATH, out=config.EXTRACTIONS_PATH, model: str | None = None,
                limit: int | None = None, client=None, resume: bool = False) -> dict:
    """Extract a Case from every report. Each row is written as soon as it is done, so a crash
    (or a hit rate limit) loses nothing, and --resume continues where the last run stopped."""
    client = client or get_client()
    session = ModelSession(client, model)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(out)) if resume and out.exists() else []
    done = {r["id"] for r in rows}
    with open(out, "a" if resume else "w", encoding="utf-8") as f:
        for i, pair in enumerate(read_jsonl(pairs)):
            if limit is not None and i >= limit:
                break
            if pair["id"] in done:
                continue
            try:
                prediction, error = session.call(lambda m: extract_case(pair["report"], client, m).model_dump()), None
            except ExtractionError as e:
                prediction, error = None, str(e)
            row = {**pair, "prediction": prediction, "error": error}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"  extracted {i + 1}...")
    total = len(rows)
    valid = sum(r["prediction"] is not None for r in rows)
    summary = {
        "total": total,
        "valid": valid,
        "failed": total - valid,
        "validity_rate": round(valid / total, 3) if total else 0.0,
    }
    print(f"extract: {summary} -> {out}")
    return summary


def cmd_terms(n: int = 2000, out=config.TERMS_PATH, sample: bool = False, session=None) -> int:
    """The reaction vocabulary used by the coding step."""
    out = Path(out)
    if sample:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.SAMPLE_TERMS, out)
        count = len(load_terms(out))
    else:
        count = save_terms(out, fetch_terms(n, session=session, api_key=config.OPENFDA_API_KEY))
    print(f"terms: {count} reaction terms -> {out}")
    return count


def cmd_code(extractions=config.EXTRACTIONS_PATH, out=config.CODED_PATH, terms_path=config.TERMS_PATH,
             model: str | None = None, limit: int | None = None, client=None, resume: bool = False) -> dict:
    """Code the reactions of every extracted case. Rows are written as they finish, like extract;
    --resume keeps the rows already in coded.jsonl and only does the rest."""
    terms_path = Path(terms_path)
    if not terms_path.exists():
        raise SystemExit(f"{terms_path} not found. Run `python -m pvcrew.pipeline terms` first "
                         "(or `terms --sample` for the small built-in list).")
    counts = load_term_counts(terms_path)
    terms = list(counts)
    if len(terms) < 500:
        print(f"note: only {len(terms)} terms in {terms_path}; many reactions will stay uncoded. "
              "`terms` collects up to 2,000 (the built-in --sample list is a small demo).")
    client = client or get_client()
    session = ModelSession(client, model)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    methods = {"exact": 0, "model": 0, "string": 0, "none": 0}
    done = {r["id"] for r in read_jsonl(out)} if resume and out.exists() else set()
    n = len(done)
    with open(out, "a" if resume else "w", encoding="utf-8") as f:
        for i, row in enumerate(read_jsonl(extractions)):
            if limit is not None and i >= limit:
                break
            if row["id"] in done:
                continue
            if row.get("prediction"):
                coded = session.call(lambda m: code_reactions(row["prediction"]["reactions"], client, m, terms, counts))
                row = {**row, "coded": coded, "reactions_coded": [c["term"] or c["reported"] for c in coded]}
                for c in coded:
                    methods[c["method"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print(f"  coded {n}...")
    summary = {"rows": n, "reactions": sum(methods.values()), **methods}
    print(f"code: {summary} -> {out}")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pvcrew.pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DATA_DIR), help="where the .jsonl files are read and written")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=200)
    f.add_argument("--sample", action="store_true", help="use bundled sample records instead of openFDA")
    f.add_argument("--skip", type=int, default=0, help="start this many records into FAERS, to get reports "
                                                      "that do not overlap an earlier fetch")

    s = sub.add_parser("synth")
    s.add_argument("--offline", action="store_true", help="template reports, no model needed")
    s.add_argument("--model", default=None, help="default: auto-pick a live model")
    s.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("extract")
    e.add_argument("--model", default=None, help="default: auto-pick a live model")
    e.add_argument("--limit", type=int)
    e.add_argument("--resume", action="store_true", help="keep existing rows, only process new ids")

    t = sub.add_parser("terms")
    t.add_argument("--n", type=int, default=2000, help="how many common FAERS reaction terms to collect")
    t.add_argument("--sample", action="store_true", help="use the small built-in list instead of openFDA")

    c = sub.add_parser("code")
    c.add_argument("--model", default=None, help="default: auto-pick a live model")
    c.add_argument("--limit", type=int)
    c.add_argument("--resume", action="store_true", help="keep rows already in coded.jsonl, only code the rest")

    a = sub.add_parser("all")
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--sample", action="store_true")
    a.add_argument("--offline", action="store_true")
    a.add_argument("--model", default=None, help="default: auto-pick a live model")
    a.add_argument("--limit", type=int)
    a.add_argument("--resume", action="store_true")

    args = p.parse_args(argv)
    d = Path(args.data_dir)
    raw, pairs, extractions = d / "faers_raw.jsonl", d / "pairs.jsonl", d / "extractions.jsonl"
    terms, coded = d / "terms.jsonl", d / "coded.jsonl"

    if args.cmd == "fetch":
        cmd_fetch(args.n, raw, args.sample, start_skip=args.skip)
    elif args.cmd == "synth":
        cmd_synth(raw, pairs, args.offline, args.model, args.seed)
    elif args.cmd == "extract":
        cmd_extract(pairs, extractions, args.model, args.limit, resume=args.resume)
    elif args.cmd == "terms":
        cmd_terms(args.n, terms, args.sample)
    elif args.cmd == "code":
        cmd_code(extractions, coded, terms, args.model, args.limit, resume=args.resume)
    else:
        cmd_fetch(args.n, raw, args.sample)
        cmd_synth(raw, pairs, args.offline, args.model)
        cmd_extract(pairs, extractions, args.model, args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
