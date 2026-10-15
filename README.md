# Multi-Expert Incident Triage on Nugen

A router model delegates engineering incidents to a fleet of **domain-aligned expert models**, merges their answers, and improves over time from user feedback.

**Live app:** `<YOUR PUBLIC URL>`
**Stack:** Python, FastAPI, Nugen alignment + inference APIs

## The problem

When something breaks in production, the cause often spans several domains: an API timeout might be a missing database index, a container limit, or an expired token. One general model gives broad but shallow answers. This system sends each problem to the experts whose knowledge applies and combines what they say.

## Architecture

```
            question
               |
        [ Router ]  corpus-derived lexical index, picks 1-2 experts
        /    |    \
 backend  database  cloud  security      4 aligned models (Qwen2.5-0.5B-Instruct base),
        \    |    /                      each aligned on its own corpus
   confidence check -> escalate to next-best expert if low
               |
        [ Synthesis ]  merges answers, notes disagreements
               |
   answer + route + per-expert confidence
               |
   thumbs up / down + correction  ->  logs/feedback.jsonl
               |
        improve.py <expert>  -> re-align -> compare -> promote if better
```

- **Experts:** each is aligned with the Nugen API on a plain-text corpus in `corpus/`. Nugen generates the training data and a benchmark from the documents.
- **Router:** builds a TF-IDF-style index from the expert corpora (no API call, fully inspectable). Queries that match no expert's vocabulary go to all four.
- **Escalation:** if an expert's confidence score is below `CONF_MIN` (default 0.5), the next-best unused expert is also asked.
- **Synthesis:** one model combines the answers. By default this is an expert model. Set `SYNTH_BASE_URL`, `SYNTH_API_KEY` and `SYNTH_MODEL` to use any OpenAI-compatible general model instead.
- **Improvement loop (`improve.py`):**
  1. Collects thumbs-down feedback that includes a correction.
  2. Adds the corrections to that expert's corpus and aligns a new version.
  3. Scores the old and new models on the expert's benchmark (token-overlap F1).
  4. Promotes the new version only if it scores at least as well.

## Results of the improvement loop

| Expert | Corrections used | Benchmark F1 (v1) | Benchmark F1 (v2) | Promoted |
|--------|------------------|-------------------|-------------------|----------|
| `<backend>` | `<3>` | `<0.000>` | `<0.000>` | `<yes/no>` |

**Before (v1):** `<question and the old answer>`
**After (v2):** `<same question and the new answer>`

![App screenshot](screenshot.png)

## Run it locally

```bash
pip install -r requirements.txt
export NUGEN_API_KEY=...            # never commit this

# 1. Write corpus/backend.txt, database.txt, cloud.txt, security.txt (>= 500 words each)
# 2. Align and deploy the four experts (resumable)
python setup_experts.py

# 3. Start the app
uvicorn app:app --reload            # http://127.0.0.1:8000

# 4. After collecting corrections in the UI
python improve.py backend
```

`experts_state.json` stores model and alignment IDs (no secrets) and is read by the app at startup.

## Files

| File | Purpose |
|------|---------|
| `nugen.py` | API helpers: upload, benchmark, align, deploy, chat, scoring |
| `setup_experts.py` | Aligns and deploys every expert in `experts.json` |
| `router.py` | Routing, escalation, synthesis, query logging |
| `app.py` | FastAPI web app with feedback and a stats endpoint |
| `improve.py` | The re-alignment loop for one expert |
| `corpus/` | The four expert knowledge files |

## Limitations (honest ones)

- The experts are small (0.5B parameters), so answers can be short or imprecise, especially on questions combining several facts.
- The router is lexical. It works on topic words, so a vague question like "everything is slow" falls back to asking all experts.
- The improvement score is a token-overlap F1 on a benchmark generated from the same corpus. It measures recall of the corpus, not general quality, and small changes can be noise.
- The loop rebuilds each alignment from the full corpus, because the API requires at least 500 words per document.
- Confidence-based escalation depends on the confidence value in the API response. If it is absent, escalation does not fire.
- On a free host the log files are lost on restart, so the loop was run locally and the results above come from that run.
- Nugen's own inference capture and auto-align endpoints could replace parts of this loop. I built it on the upload, align, deploy and chat endpoints that I tested.

## What I would do next

- Use embeddings from the aligned models for routing instead of lexical matching.
- Use Nugen's auto-align with a target benchmark score instead of manual re-alignment.
- Run evaluations in the platform and add a human-reviewed benchmark per expert.
- Persist logs in a database and add an admin page that lists the gaps to fix.

## Security

The API key is read from the `NUGEN_API_KEY` environment variable and is never stored in the repo. The app rate-limits each IP to 20 questions per hour to protect credits.
