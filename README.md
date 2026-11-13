# PV Crew: Day 1 (extraction + data), Day 2 (scoring + coding), Day 3 (agent crew + reviewer page)

An LLM turns messy drug-safety reports into schema-valid JSON, on a dataset where the right answers are known.
Runs on **NVIDIA NIM** (hosted API, nothing heavy on your machine). Settings and your API key live in one `.env` file.

## Quick start (Windows PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # blocked? run once:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```
macOS/Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

**1. Get a free NVIDIA API key** (about 3 minutes):
1. Go to https://build.nvidia.com and click **Login**. Create a free NVIDIA account (email + verification code).
2. Open any model page (for example search "gemma") and click **Get API Key** (or **Build with this NIM**, then **Generate API Key**).
3. NVIDIA asks you to verify a phone number by SMS the first time. Enter it and the code.
4. Click **Generate Key**, then **Copy**. The key starts with `nvapi-`. If you lose it, generate a new one from the same page.

**2. Put the key in `.env`.** (No `.env` in the folder? Copy `.env.example` to `.env`.) Open it in Notepad or VS Code and paste the key after
`NVIDIA_API_KEY=` (no quotes, no spaces, no extra line). Save the file.

**3. Check everything works** (about 2 API calls):
```powershell
python -m pvcrew.check
```
It prints the `.env` file it found, your key (masked), the models your key can use, the model it picked, and one
real test extraction. If something is wrong it says what to fix.

**4. Run the tests** (no key or internet needed):
```powershell
pytest
```

## Run the pipeline
```powershell
python -m pvcrew.pipeline all --sample --offline --limit 3   # tiny smoke run on bundled data (~4 API calls)

# the real thing. Start small: the free tier is about 1,000 credits/month (roughly 1 call each)
python -m pvcrew.pipeline fetch --n 100          # FAERS via openFDA
python -m pvcrew.pipeline synth --offline        # template reports: 0 credits. Drop --offline for LLM-written ones
python -m pvcrew.pipeline extract --limit 20     # ~20 calls; read the results before running everything
python -m pvcrew.pipeline extract --resume       # continue after a stop; finished ids are skipped
```
Files land in `data/`: `faers_raw.jsonl` -> `pairs.jsonl` (report + truth) -> `extractions.jsonl` (+ prediction).
Each extraction is saved the moment it finishes, so a crash or rate limit never loses paid-for work.

Score the results against the FAERS answers (per-field accuracy, missed/wrong/spurious, reaction precision/recall, first mistakes):
```powershell
python -m pvcrew.score                       # data/extractions.jsonl
python -m pvcrew.score data_hard\extractions.jsonl --show 20
```

See the failures (your Day 2 test cases):
```powershell
python -c "import json; [print(r['id'], r['error'], chr(10), r['report'], chr(10)) for r in map(json.loads, open('data/extractions.jsonl', encoding='utf-8')) if r['error']]"
```

## Coding step (Day 2): wording -> standard terms
The model writes reactions the way the report did ("the runs"). FAERS uses standard terms ("Diarrhoea"). This step maps one to the other.
```powershell
python -m pvcrew.pipeline terms                  # up to 2,000 common FAERS reaction terms (openFDA's count query, topped up
                                                 # from ~2,000 other records; free; add OPENFDA_API_KEY to .env for higher limits)
python -m pvcrew.pipeline code                   # data/extractions.jsonl -> data/coded.jsonl  (<= 2 model calls per report)
python -m pvcrew.pipeline code --resume          # continue after a stop; rows already coded are kept
python -m pvcrew.score data\coded.jsonl          # reactions are scored as extracted AND after coding
```
Offline demo list: `terms --sample`. For each reaction: exact match first (no model), else the model rewords it medically,
a shortlist of similar terms is built, and the model picks one or says none. It cannot invent a term. Each coded row keeps the
reporter's wording (`coded[].reported`) next to the term and how it was chosen (`exact` / `model` / `string` / `none`).
Limits: the vocabulary is the ~2,000 terms collected (`terms --n` changes it; `code` warns if there are under 500), so rarer terms cannot be reached (an upper bound on the score). The licensed MedDRA
dictionary replaces `terms.jsonl` in production.

## Day 3: the agent crew, the reviewer page and the live monitor
```powershell
pip install -r requirements.txt      # adds autogen-agentchat and streamlit
streamlit run app.py
```
The reviewer page opens in your browser. A second window, the **Crew Monitor**, opens with it so you can watch the
agents work. Your browser decides whether that is a new window or a tab; the sidebar button **Open monitor window**
always gives a separate, sized popup, and the address (default http://127.0.0.1:8765/, the next free port if that one is
busy) is shown under it.

**Use it:** pick a sample report or paste your own, press **Run the crew**, watch the monitor, then check what the crew
proposes. Fix anything wrong (drug, age, outcome, each reaction's standard term, untick wrong reactions, add missed ones)
and press **Approve** or **Reject**. The first run picks a model (a few calls). It uses the vocabulary from
`python -m pvcrew.pipeline terms` if you built it, otherwise a small demo list, and says which.

**The crew** (four AutoGen agents in a `GraphFlow` group chat):
```
report -> Intake -> Coding -> Critic -> Narrative -> you (approve / edit / reject)
                      ^         |
                      +-REVISE--+   (at most CREW_MAX_REVISIONS times, then the case is flagged ESCALATE)
```
| Agent | Job |
|---|---|
| Intake | reads the report, extracts the case fields (the Day 1 step); on a revision it also gets the Critic's list of problems |
| Coding | turns each reaction into a standard term (the Day 2 step) |
| Critic | compares the case with the report and answers APPROVE or REVISE with concrete issues. One rule is fixed in code: a fatal outcome must be marked serious |
| Narrative | writes a short case narrative and a demo triage (URGENT / HIGH / ROUTINE, plus follow-up flags) |

**The monitor** shows, live: the AutoGen workflow as a diagram that lights up as each agent is called; a WhatsApp-style
group chat of what each agent said (the Critic's REVISE goes back to Intake as a visible reply); a step-by-step list of
what AutoGen did (who the manager picked next and why, which messages each agent was handed, each model call and how
long it took); the raw objects `run_stream()` yields; and your own decision when you press Approve.

**Files written** (both are appended to, never overwritten):
- `data/audit.jsonl`: every agent input and output of every run, one JSON object per line. Also downloadable from the **Audit log** tab.
- `data/reviews.jsonl`: each decision with what the crew proposed, what you finally approved, and the list of your edits.
  Decisions are `approved`, `approved_with_edits` or `rejected`.

**Settings** (all optional, in `.env`): `CREW_MAX_REVISIONS` (default 1), `PV_MONITOR=off` (no monitor server),
`PV_MONITOR_OPEN=off` (do not auto-open the window), `PV_MONITOR_PORT` (default 8765).

**What is and is not AutoGen here.** AutoGen runs the conversation: the agents, the group chat, who speaks next, the
loop back to Intake, the stop condition. The model calls themselves go through this project's own client, which already
handles pacing, retries and switching models, so AutoGen's built-in model client is not used.

## Comparing models (accuracy)
Runs several NVIDIA models over the same reports and scores each against the FAERS answers the reports were written from.
For every report it checks six things: age, sex, suspect drug, serious, outcome, and the set of reactions after coding.
You get the share of checks that were right, the share of reports that were right in every way, a 95% interval for both
(so you can see whether two models are really different or the sample is just small), and a per-field table.

```powershell
# 1. A fresh set of reports, from a different part of FAERS than the ones used while building and tuning (--skip)
python -m pvcrew.pipeline --data-dir data_eval fetch --n 60 --skip 2000
python -m pvcrew.pipeline --data-dir data_eval synth          # ~60 calls. Or add --offline: 0 calls, but easier template reports

# 2. A 5-report trial. --auto picks 3 working models from different publishers and prints their ids
python -m pvcrew.compare --data-dir data_eval --auto 3 --limit 5
#    (or name the models yourself with --models id1,id2,id3 ; list the ones on your key with: python -m pvcrew.llm instruct)

# 3. The real run: name the models it printed, so no credits are spent finding them again. Finished reports are kept, only new ones cost
python -m pvcrew.compare --data-dir data_eval --models id1,id2,id3 --limit 50

streamlit run app.py          # the "Model comparison" tab shows the same table and a chart
```
- **Credits.** About 3 calls per report per model (one to read the report, up to two to code the reactions). 3 models x 50 reports is
  roughly 450 calls of your ~1,000 monthly free credits, plus ~60 if an LLM writes the reports and ~20 for `--auto` to find working models.
  The 5-report trial costs about 45 calls and shows which models work on your key before you spend the rest.
- **Stopping is safe.** Every answer is saved as it arrives. If a model fails (retired, rate limited), the others carry on, the table marks it
  `(incomplete)`, and running the same command again continues it. `--fresh` throws earlier results away and starts over; use it after you change
  the reports or the prompts.
- **Fair by construction.** Every model gets the same reports and is pinned: it is never swapped for another one halfway. A report a model could not
  answer counts as wrong, so a model that often returns broken JSON loses points for it.
- **If `--auto` finds nothing:** NVIDIA's model list still shows many retired models, and it only tries `NIM_MAX_MODEL_TRIES` (default 10) of them. Raise it for that
  PowerShell window with `$env:NIM_MAX_MODEL_TRIES = 40`. **Very large models can be too slow** for 50 reports (a model that times out is marked incomplete):
  give it longer with `$env:NIM_TIMEOUT = 240`, or leave it out of `--models`.
- **Needs the reaction vocabulary** from `python -m pvcrew.pipeline terms` (already there if you did the coding step). If you set `PV_MODEL` in `.env`,
  `--auto` refuses to run, because that pins one model.
- Results are in `data_eval/compare/`: `compare.md`, `compare.json`, and a folder per model. To see one model's mistakes:
  `python -m pvcrew.score data_eval\compare\<model folder>\coded.jsonl --show 20`.

Read the numbers with these limits in mind (they are also printed under the table):
- The reports are synthetic and written from the answers, so scores are optimistic for real reporters' wording. No hand-labelled reports exist yet.
- A model that also wrote the reports (`synth` without `--offline` picks one automatically) may read its own writing better. Use `synth --model` with a model you are not comparing, or `--offline`.
- The extraction prompt was developed while testing with one model, which may favour that model.
- With 50 reports the "whole report correct" interval is about plus or minus 13 points. A gap smaller than the interval is not evidence that one model is better; the "Versus the top model" column says when a gap is clear.

## Ideas for later (not built)
- **Hand-labelled reports.** Have a pharmacovigilance reviewer label 50 to 100 real (anonymised) reports. This is the biggest step up in trust, because the synthetic truth cannot show how models cope with real wording.
- **Reviewer edits as a free accuracy signal.** `data/reviews.jsonl` already stores what the model proposed and what the human changed. The share of approved-without-edits cases, and which fields get edited, measures real accuracy as the crew is used.
- **Safety-weighted scoring.** Missing a serious or fatal case is worse than getting an age wrong. A score that weights fields by consequence, or a separate recall figure for serious cases, would match how the work is judged.
- **Breakdown by report style.** `pairs.jsonl` records the writing style of each report; scoring per style shows which wording breaks which model.
- **Does the Critic work?** Plant known errors in cases and measure how many the Critic sends back. Today it is not measured.
- **Triage accuracy** against an expert's priorities, once an expert has labelled some cases.
- **Several report-writing models**, so no compared model is graded on its own writing.
- **Repeat runs and bigger samples** to see how stable a ranking is, and speed and cost per report next to accuracy.

## The .env file
`pvcrew/config.py` is the only code that reads settings; every other module goes through it, so one file controls everything.
A real environment variable with the same name wins over `.env` (handy for one-off runs).

| Setting | What it does | Default |
|---|---|---|
| `NVIDIA_API_KEY` | your NVIDIA key (required for the hosted API) | none |
| `PV_MODEL` | pin a model; otherwise a live one is picked automatically | auto |
| `NIM_TIMEOUT` | seconds to wait for one answer before a model counts as stalled | `90` |
| `NIM_MAX_MODEL_TRIES` | how many models to test before giving up | `10` |
| `NIM_MIN_INTERVAL` | seconds between calls (free tier is about 40 requests/min) | `1.5` |
| `OPENFDA_API_KEY` | higher openFDA download limits | none |
| `NIM_BASE_URL` | self-hosted NIM instead of NVIDIA's cloud (key optional) | NVIDIA cloud |
| `PV_PROVIDER` / `OLLAMA_HOST` | `ollama` to use a local model (`pip install ollama`) | `nim` |
| `PV_DATA` | where data files are written | `data` |

`.env` is in `.gitignore`. Never commit it, share it, or paste it into a chat. `.env.example` is the shareable template.

## Retired models (HTTP 410 / "model not available")
Hosted NIM retires models regularly and its model list still shows some retired ones, so no model ID is trusted.
Unless you pin `PV_MODEL`, each run does this:
1. Reads your key's model list.
2. Tries the preferred models in `config.NIM_CANDIDATES` first, then other chat/instruct models from your list
   (embedding, vision, safety, code and reasoning models are skipped; instruction-tuned models near 30B go first).
3. Tests each with a small call and, in `python -m pvcrew.check`, a real extraction. It prints `trying <id> ... skipped: <reason>`
   for each one, and stops at the first that works.
4. Gives up after `NIM_MAX_MODEL_TRIES` models (default 10) and prints every model tried with the reason.

The model is chosen by a real test extraction, not a one-word ping. If a model still fails mid-run (a 404, or no answer
within `NIM_TIMEOUT`), `extract` retries a 404 once, then switches to the next working model and carries on. A first
timeout also turns off `guided_json` and retries, because constrained decoding is a common cause of stalls. A pinned `PV_MODEL` is never swapped: it stops, keeps finished rows, and tells you to `--resume`.

When it succeeds it prints `Tip: put PV_MODEL=<id> in .env`. Do that to skip the search on later runs.
`python -m pvcrew.llm` lists your key's models (`python -m pvcrew.llm nemotron` filters).

## How structured output works on NIM
The schema is sent as `nvext.guided_json` (NIM's constrained decoding) and also written into the system prompt.
If a model rejects `guided_json` (HTTP 400), the client drops it, remembers that, and relies on the prompt plus
`extract_case`'s validate-and-retry loop.

## Privacy
`data/audit.jsonl` and `data/reviews.jsonl` contain the report text you ran, so they are kept out of git. The monitor only listens on this computer (127.0.0.1).
The hosted API sends prompts to NVIDIA's cloud. Fine for public FAERS data and synthetic reports (this project).
Do not send real patient or company case data to it. NVIDIA describes the free tier as for prototyping and research;
production needs an NVIDIA AI Enterprise licence, or a self-hosted NIM (same code, just `NIM_BASE_URL`).

## Layout
| File | Job |
|---|---|
| `.env`, `.env.example`, `.gitignore` | your settings and key; shareable template; keeps `.env` out of git |
| `pvcrew/config.py` | loads `.env`, the single place settings are read |
| `pvcrew/check.py` | `python -m pvcrew.check` preflight |
| `pvcrew/schemas.py` | `Case` model: the one record every stage uses |
| `pvcrew/faers.py` | openFDA fetch + FAERS record -> `Case` (ground truth) |
| `pvcrew/synth.py` | `Case` -> messy free-text report (LLM or offline templates) |
| `pvcrew/extract.py` | report -> validated `Case`, with retry on invalid output |
| `pvcrew/llm.py` | NIM client: pacing, 429/5xx retry, guided JSON, model discovery + auto-pick; `python -m pvcrew.llm` lists models |
| `pvcrew/coding.py` | reaction vocabulary + the coding step |
| `pvcrew/score.py` | `python -m pvcrew.score` accuracy report |
| `pvcrew/pipeline.py` | CLI: fetch (`--skip` for a different slice), synth, extract (resumable), all |
| `pvcrew/compare.py` | `python -m pvcrew.compare` runs several models on the same reports and scores them side by side |
| `app.py` | Day 3: the Streamlit reviewer page (`streamlit run app.py`), plus the Model comparison tab |
| `pvcrew/crew/agents.py` | the four agents (Intake, Coding, Critic, Narrative) |
| `pvcrew/crew/team.py` | the AutoGen `GraphFlow`, the routing rules, `run_crew` |
| `pvcrew/crew/triage.py` | demo triage rules (not regulatory advice) |
| `pvcrew/crew/events.py` | the event bus: feeds the monitor and writes the audit log |
| `pvcrew/crew/monitor.py`, `monitor.html` | the live monitor window (a small local web server and its page) |
| `pvcrew/crew/service.py` | what the page calls: run the crew, save the reviewer's decision, read samples and logs |
| `data/sample_faers.jsonl` | 6 hand-written records in FAERS format for offline use (not real reports) |
| `tests/` | About 450 tests. `fake_nim.py` is a local stand-in for NVIDIA's API; `test_e2e_cli.py` runs the real commands off a `.env` file. The Day 3 tests use a scripted fake model; `test_app.py` drives the page headlessly; `test_monitor_browser.py` draws the monitor in a real browser and is skipped unless `playwright` and Chromium are installed |

## Things to know
- FAERS reactions are coded terms ("Dyspnoea"); the model may write lay wording ("trouble breathing"). Matching them is Day 2's job.
- Synthetic reports are written from the answers, so they are easier than real ones. Early scores will be optimistic.
- Age is stored in whole years (a 6-month-old is 0). A missing age unit in FAERS gives `null` rather than a guess.
- The tests prove the request format, error handling and `.env` behaviour. They cannot prove your chosen model extracts well: run `extract --limit 20` and read the failures.
- The Critic is the same model checking its own team's work, so it catches some mistakes and not all. Treat it as a second look, not an independent one. Day 2's scorer is still how you measure accuracy.
- The narrative is not rewritten when you change case fields. If you correct the age, fix the narrative text too.
- The triage rules are a demo of the idea. Real reporting clocks depend on expectedness, jurisdiction and your company's procedure.
- One crew run at a time: the monitor shows a single conversation.
- The Day 3 tests never call a real model. How well your chosen model does as Critic and Narrative is something to look at in the audit log after a few real runs.
