"""Score extractions against the FAERS answers:  python -m pvcrew.score [extractions.jsonl or coded.jsonl] [--show N]

Per field, every report lands in one bucket:
  correct   matches the FAERS value
  missed    model said unknown/null but FAERS has a value (also what you get when the report never states it)
  spurious  model gave a value but FAERS has none
  wrong     both have a value and they differ
Reactions are scored by exact set match, plus precision/recall over the individual terms. On a coded.jsonl file
they are scored both as extracted and after coding, so you can see what the coding step added.
"""
import argparse
import sys
from pathlib import Path

from . import config
from .jsonl import read_jsonl

EMPTY = (None, "", "unknown")
FIELDS = ["patient_age", "patient_sex", "suspect_drug", "serious", "outcome"]


def _norm(x):
    return x.strip().lower() if isinstance(x, str) else x


def bucket(field: str, pred, truth) -> str:
    p, t = _norm(pred), _norm(truth)
    p_empty, t_empty = p in EMPTY, t in EMPTY
    if field == "serious":
        return "correct" if p == t else "wrong"
    if p_empty and t_empty:
        return "correct"
    if p_empty:
        return "missed"
    if t_empty:
        return "spurious"
    if field == "suspect_drug":  # brand/strength suffixes differ a lot: "DURAGESIC-100" vs "Duragesic"
        return "correct" if p == t or p in t or t in p else "wrong"
    return "correct" if p == t else "wrong"


def _terms(items):
    return {_norm(x) for x in (items or []) if _norm(x)}


def score_rows(rows) -> dict:
    rows = list(rows)
    done = [r for r in rows if r.get("prediction")]
    out = {"total": len(rows), "scored": len(done), "invalid": len(rows) - len(done), "fields": {}, "mistakes": []}
    for f in FIELDS:
        counts = {"correct": 0, "missed": 0, "spurious": 0, "wrong": 0}
        for r in done:
            b = bucket(f, r["prediction"].get(f), r["truth"].get(f))
            counts[b] += 1
            if b != "correct":
                out["mistakes"].append((r["id"], f, b, r["prediction"].get(f), r["truth"].get(f)))
        counts["accuracy"] = round(counts["correct"] / len(done), 3) if done else 0.0
        out["fields"][f] = counts
    raw_mistakes = []
    out["reactions"] = _reactions(done, lambda r: r["prediction"].get("reactions"), "reactions", raw_mistakes)
    if any("reactions_coded" in r for r in done):  # the coding step has run: score its output too
        out["reactions_coded"] = _reactions(done, lambda r: r.get("reactions_coded", r["prediction"].get("reactions")),
                                            "reactions_coded", out["mistakes"])  # list only what is left after coding
    else:
        out["mistakes"] += raw_mistakes
    return out


def _reactions(done, get, label, mistakes) -> dict:
    exact = tp = n_pred = n_truth = 0
    for r in done:
        p, t = _terms(get(r)), _terms(r["truth"].get("reactions"))
        exact += p == t
        tp += len(p & t)
        n_pred += len(p)
        n_truth += len(t)
        if p != t:
            mistakes.append((r["id"], label, "differs", sorted(p - t), sorted(t - p)))
    return {
        "exact_set_match": round(exact / len(done), 3) if done else 0.0,
        "precision": round(tp / n_pred, 3) if n_pred else 0.0,
        "recall": round(tp / n_truth, 3) if n_truth else 0.0,
    }


def format_report(s: dict, show: int = 10) -> str:
    lines = [f"{s['scored']} scored, {s['invalid']} had no valid output", "",
             f"{'field':<14}{'correct':>8}{'missed':>8}{'spurious':>9}{'wrong':>7}{'accuracy':>10}"]
    for f, c in s["fields"].items():
        lines.append(f"{f:<14}{c['correct']:>8}{c['missed']:>8}{c['spurious']:>9}{c['wrong']:>7}{c['accuracy']:>10.3f}")
    lines.append("")
    for key, name in (("reactions", "reactions (as extracted)"), ("reactions_coded", "reactions (after coding)")):
        if key in s:
            r = s[key]
            lines.append(f"{name:<25} exact set match {r['exact_set_match']:.3f} | term precision "
                         f"{r['precision']:.3f} | term recall {r['recall']:.3f}")
    if s["mistakes"] and show:
        lines += ["", f"first {min(show, len(s['mistakes']))} of {len(s['mistakes'])} mistakes "
                      "(id, field, kind, model said, FAERS says; for reactions: extra terms, missing terms):"]
        lines += [f"  {i} | {f} | {k} | {p!r} | {t!r}" for i, f, k, p, t in s["mistakes"][:show]]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pvcrew.score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(config.EXTRACTIONS_PATH))
    ap.add_argument("--show", type=int, default=10, help="how many mistakes to list (0 = none)")
    args = ap.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m pvcrew.pipeline extract` first.")
    print(format_report(score_rows(read_jsonl(path)), args.show))


if __name__ == "__main__":
    main()
