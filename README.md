# Domain-Aligned Model with the Nugen API

A hands-on exercise for Nugen Intelligence: align an open-weight model to a custom domain, deploy it, and build a small app on top of it.

**Domain:** `<YOUR DOMAIN, e.g. IT runbooks for a fictional company>`
**Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Aligned model ID:** `model_01m4qwh4afzxek78`

## What's in this repo

| File | Purpose |
|------|---------|
| `corpus.txt` | Domain knowledge used for alignment (plain text) |
| `nugen_align.py` | CLI that runs the workflow: upload, benchmark, align, deploy, inference |
| `app.py` | Small FastAPI chat app backed by the aligned model |

## Workflow

Each step uses the Nugen REST API (`https://api.nugen.in/api/v3`).

1. **Upload the corpus** (`POST /documents/create`) and poll until the document is `READY`.
2. **Generate a benchmark** (`POST /benchmarks/create`) from the corpus: 10 instruction/response pairs.
3. **Review and edit the benchmark** by hand to fix wrong answers, then re-upload it (`POST /benchmarks/upload`).
4. **Create the alignment project** (`POST /alignment-projects/create`) with the document and benchmark, and poll until `READY`.
5. **Deploy** the aligned model (it was deployed automatically for evaluation) and run inference through `POST /inference/chat/completions`.

```bash
export NUGEN_API_KEY=...            # never commit your key
python nugen_align.py upload corpus.txt
python nugen_align.py benchmark 10
python nugen_align.py upload_benchmark
python nugen_align.py align
python nugen_align.py deploy
python nugen_align.py ask "your question"
```

## The app

```bash
pip install fastapi uvicorn requests
export NUGEN_API_KEY=...
uvicorn app:app --reload            # http://127.0.0.1:8000
```

The app sends each question to the aligned model and shows the answer in a simple chat page. The model ID is read from `nugen_state.json` (written by `nugen_align.py`) or the `MODEL_ID` environment variable.

![App screenshot](screenshot.png)

## Test results

I tested the deployed model with five questions: three taken from the benchmark, one reworded, and one off-topic.

| # | Type | Question | Model answer | Verdict |
|---|------|----------|--------------|---------|
| 1 | Benchmark | `<question>` | `<answer>` | `<matches the corpus? yes/partly/no>` |
| 2 | Benchmark | `<question>` | `<answer>` | `<verdict>` |
| 3 | Benchmark | `<question>` | `<answer>` | `<verdict>` |
| 4 | Reworded | `<question>` | `<answer>` | `<verdict>` |
| 5 | Off-topic | `<question>` | `<answer>` | `<verdict>` |

**Observations:** `<2 or 3 sentences: where the model was accurate, where it was wrong or vague, how it handled the off-topic question>`

## Notes and limitations

- The base model is small (0.5B parameters), so answers can be short or imprecise on questions that combine several facts.
- The developer edition accepts plain text files only, so the corpus is a `.txt` file.
- The benchmark was generated from the corpus and then edited by hand, so it measures recall of my own material, not general capability.
- `<anything you'd do next: a larger corpus, a bigger base model, adding retrieval on top, using the confidence score to route low-confidence answers>`

## Security

API keys are read from the environment and are not stored in this repo.
