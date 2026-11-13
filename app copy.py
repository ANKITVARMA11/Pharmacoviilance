"""Reviewer screen for the PV triage crew.   Start it with:   streamlit run app.py

Paste or pick an adverse-event report, run the AutoGen crew, check what it proposes, fix anything wrong, then
approve or reject. A second window (the Crew Monitor) opens with the page and shows the agents talking.
"""
import json

import altair as alt
import streamlit as st
import streamlit.components.v1 as components

from pvcrew import config
from pvcrew.crew import service
from pvcrew.crew.agents import CrewError

st.set_page_config(page_title="Pharmacovigilance Triage Crew", page_icon="💊", layout="wide")

rt = service.get_runtime()
monitor_url = rt.start_monitor()  # opens the monitor window the first time only

SEX = ["unknown", "female", "male"]
OUTCOME = ["unknown", "recovered", "recovering", "not_recovered", "recovered_with_sequelae", "fatal"]
TRIAGE_COLOR = {"URGENT": "red", "HIGH": "orange", "ROUTINE": "green", "REVIEW": "gray"}

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("💊 PV Triage Crew")
    st.caption("Case intake and triage with four AutoGen agents and a human reviewer.")
    reviewer = st.text_input("Your name (for the record)", key="reviewer")
    st.divider()
    if monitor_url:
        st.markdown("**Crew Monitor**")
        st.caption("A separate window that shows the agents talking, live.")
        button = f"""<button style="font:14px system-ui;padding:8px 12px;border-radius:8px;border:1px solid #8883;
            cursor:pointer;width:100%" onclick="window.open('{monitor_url}','crew_monitor',
            'width=1300,height=860,noopener')">🛰️ Open monitor window</button>"""
        if hasattr(st, "iframe"):  # newer Streamlit; components.html is being removed
            st.iframe(button, height=48)
        else:
            components.html(button, height=48)
        st.caption(f"Or open {monitor_url} in any browser tab.")
    else:
        st.caption("Monitor is switched off (PV_MONITOR=off in .env).")
    st.divider()
    st.caption(f"Model: {rt.session.model if rt.session else 'chosen on the first run'}")
    st.caption(f"Audit log: {config.AUDIT_PATH}")
    st.caption(f"Reviews: {config.REVIEWS_PATH}")

tab_review, tab_history, tab_audit, tab_compare = st.tabs(["Review", "Past reviews", "Audit log", "Model comparison"])

# ---------------- review ----------------
with tab_review:
    st.subheader("1. The report")
    mode = st.radio("Where does the report come from?", ["Pick a sample", "Paste your own"], horizontal=True,
                    label_visibility="collapsed")
    report_id, report = None, ""
    if mode == "Pick a sample":
        samples = service.load_samples()
        pick = st.selectbox("Sample report", range(len(samples)),
                            format_func=lambda i: f"{samples[i]['id']}  ·  {samples[i]['report'][:80]}…")
        report_id, report = samples[pick]["id"], samples[pick]["report"]
        st.text_area("Report text", report, height=110, disabled=True)
    else:
        report = st.text_area("Report text", height=140, placeholder="Paste the adverse-event report here…")
    run_clicked = st.button("▶ Run the crew", type="primary", disabled=not report.strip())

    if run_clicked:
        problem = None
        with st.status("The crew is working. Watch the Crew Monitor window.", expanded=False) as status:
            try:
                st.session_state["result"] = rt.run(report, report_id)
                st.session_state["saved"] = None
                status.update(label="The crew is done. Review the result below.", state="complete")
            except CrewError as e:
                st.session_state.pop("result", None)
                status.update(label="The crew could not run.", state="error")
                problem = str(e)
        if problem:  # outside the status box: inside it the message would stay hidden in a collapsed panel
            st.error(problem)

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader("2. What the crew proposes")
        if result["status"] == "failed":
            st.error(f"The run failed: {result['error']}")
        t = result.get("triage") or {}
        head = st.columns([1, 1, 2])
        head[0].markdown(f"**Triage:** :{TRIAGE_COLOR.get(t.get('priority'), 'gray')}[{t.get('priority', '-')}]")
        head[1].markdown(f"**Critic:** {result.get('verdict') or '-'}  ·  revisions: {result.get('revisions', 0)}")
        head[2].caption(t.get("why", ""))
        for note in result.get("issues", []) + result.get("notes", []):
            st.warning(note)
        if t.get("follow_up"):
            st.info("Follow-up needed: " + ", ".join(t["follow_up"]))

        case = result.get("case") or {}
        coded = result.get("coded") or [{"reported": r, "term": None, "method": "none"} for r in case.get("reactions", [])]
        left, right = st.columns([2, 3])
        left.markdown("**Source report**")
        left.info(result["report"])
        with right.form(f"review-{result['run']}"):
            st.markdown("**Proposed case** (edit anything that is wrong)")
            c1, c2, c3 = st.columns(3)
            age_unknown = c1.checkbox("Age not stated", value=case.get("patient_age") is None)
            age = c1.number_input("Age (years)", 0, 120, value=case.get("patient_age") or 0, disabled=age_unknown)
            sex = c2.selectbox("Sex", SEX, index=SEX.index(case.get("patient_sex", "unknown")))
            outcome = c3.selectbox("Outcome", OUTCOME, index=OUTCOME.index(case.get("outcome", "unknown")))
            drug = st.text_input("Suspect drug", value=case.get("suspect_drug", ""))
            serious = st.checkbox("Serious", value=bool(case.get("serious", False)))
            st.markdown("**Reactions and their standard terms**")
            kept = []
            for i, row in enumerate(coded):
                a, b, m = st.columns([1, 3, 1])
                keep = a.checkbox("keep", value=True, key=f"keep{i}-{result['run']}")
                term = b.text_input(row["reported"], value=row.get("term") or "", key=f"term{i}-{result['run']}",
                                    help="Standard term. Leave empty if there is none.")
                m.caption(row.get("method", ""))
                if keep:
                    kept.append({"reported": row["reported"], "term": term.strip() or None})
            extra = st.text_area("Add reactions the crew missed (one per line)", key=f"extra-{result['run']}", height=68)
            narrative = st.text_area("Narrative", value=result.get("narrative", ""), height=120)
            comment = st.text_input("Comment (optional)")
            b1, b2 = st.columns(2)
            approve = b1.form_submit_button("✅ Approve", type="primary", width="stretch")
            reject = b2.form_submit_button("⛔ Reject", width="stretch")

        if approve or reject:
            kept += [{"reported": x.strip(), "term": None} for x in extra.splitlines() if x.strip()]
            if approve and (not drug.strip() or not kept):
                st.error("To approve, a suspect drug and at least one reaction are needed.")
            else:
                final = {"case": {"patient_age": None if age_unknown else int(age), "patient_sex": sex,
                                  "suspect_drug": drug.strip(), "reactions": [k["reported"] for k in kept],
                                  "serious": serious, "outcome": outcome},
                         "coded": kept, "narrative": narrative}
                rec = rt.review(result, "approved" if approve else "rejected", final, reviewer, comment)
                st.session_state["saved"] = rec
        saved = st.session_state.get("saved")
        if saved and saved["run"] == result["run"]:
            msg = {"approved": "Approved", "approved_with_edits": "Approved with edits", "rejected": "Rejected"}[saved["decision"]]
            st.success(f"{msg}. Saved to {config.REVIEWS_PATH}.")
            if saved["changes"]:
                st.caption("Your edits: " + "; ".join(saved["changes"]))

# ---------------- history ----------------
with tab_history:
    rows = service.read_reviews()
    if not rows:
        st.info("No reviews yet.")
    else:
        st.dataframe([{"time": r["ts"], "report": r.get("report_id"), "decision": r["decision"], "reviewer": r["reviewer"],
                       "triage": (r.get("triage") or {}).get("priority"), "edits": "; ".join(r.get("changes", [])),
                       "comment": r.get("comment", "")} for r in reversed(rows)], width="stretch")

# ---------------- audit ----------------
with tab_audit:
    audit = service.read_audit()
    runs = list(dict.fromkeys(r["run"] for r in audit if r.get("run")))
    if not runs:
        st.info("Nothing recorded yet. Every agent input and output is written to the audit log.")
    else:
        run = st.selectbox("Run", list(reversed(runs)))
        shown = [r for r in audit if r.get("run") == run and r["kind"] in ("run_start", "called", "llm", "message", "manager", "error", "run_end", "review")]
        st.dataframe([{"time": r["clock"], "event": r["kind"], "agent": r.get("agent", ""),
                       "detail": (r.get("text") or r.get("purpose") or "")[:300] if r["kind"] != "called" else
                       "received: " + ", ".join(x["source"] for x in r.get("received", []))} for r in shown],
                     width="stretch")
        st.download_button("Download the full audit log", config.AUDIT_PATH.read_bytes(), file_name="audit.jsonl",
                           mime="application/json")

# ---------------- model comparison ----------------
with tab_compare:
    found = service.read_comparisons()
    if not found:
        st.info("No comparison yet. Run `python -m pvcrew.compare --auto 3 --limit 50` (see the README), then reload this page.")
    else:
        label = st.selectbox("Comparison", list(found)) if len(found) > 1 else next(iter(found))
        results, text = found[label]
        st.caption(f"Made {results.get('made', '')} from {results.get('reports', '?')} reports.")
        rows = [{"model": m["model"].split("/")[-1], "items": round(100 * m["items"]["value"], 1),
                 "label": f"{100 * m['items']['value']:.1f}%", "finished": "yes" if m.get("complete", True) else "no"} for m in results["models"]]
        bars = alt.Chart(alt.Data(values=rows)).encode(
            y=alt.Y("model:N", sort=[r["model"] for r in rows], title=None, axis=alt.Axis(labelLimit=420)),  # best first
            x=alt.X("items:Q", title="Items correct (%)", scale=alt.Scale(domain=[0, 100])))
        chart = bars.mark_bar().encode(color=alt.Color(  # grey for a model that did not finish every report
            "finished:N", scale=alt.Scale(domain=["yes", "no"], range=["#1f77b4", "#aaaaaa"]),
            legend=alt.Legend(title="Finished all reports") if any(r["finished"] == "no" for r in rows) else None))
        labels = bars.mark_text(align="left", dx=4).encode(text="label:N")
        st.altair_chart((chart + labels).properties(height=44 * len(rows) + 40))
        body = text.split("\n", 1)[1] if text.startswith("# ") else text  # the tab already has a title
        st.markdown(body.replace("\n## ", "\n### "))
        st.download_button("Download this comparison (compare.md)", text, file_name="compare.md", mime="text/markdown")
