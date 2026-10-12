"""
End-to-end Nugen workflow: upload docs -> benchmark -> align -> deploy -> infer.
Endpoints/fields are taken from the nugen-cookbook alignment + inference notebooks.
Run stages one at a time; state is saved to nugen_state.json so you can resume.

  export NUGEN_API_KEY=...        # never hardcode / paste the key anywhere
  python nugen_align.py upload corpus.txt   # plain text / markdown only on the developer edition
  python nugen_align.py benchmark 10
  python nugen_align.py align Qwen/Qwen2.5-0.5B-Instruct   # workflow_id is optional per the API docs
  python nugen_align.py deploy
  python nugen_align.py ask "your question"
"""
import json, os, sys, time
from pathlib import Path
import requests

BASE = "https://api.nugen.in"
H = {"Authorization": f"Bearer {os.environ['NUGEN_API_KEY']}"}
STATE = Path("nugen_state.json")
state = json.loads(STATE.read_text()) if STATE.exists() else {}


def save(**kw):
    state.update(kw)
    STATE.write_text(json.dumps(state, indent=2))


def call(method, path, **kw):
    r = requests.request(method, f"{BASE}{path}", headers=H, timeout=120, **kw)
    if not r.ok:
        sys.exit(f"{method} {path} -> {r.status_code}: {r.text}")
    return r.json()


def poll(path, done=("READY", "COMPLETED", "DONE", "SUCCESS", "DEPLOYED"),
         bad=("FAILED", "ERROR"), every=10, key="status"):
    while True:
        d = call("GET", path)
        s = str(d.get(key, "")).upper()
        print(path, "->", s)
        if s in done:
            return d
        if s in bad:
            sys.exit(f"failed: {d}")
        time.sleep(every)


def upload(file):
    with open(file, "rb") as f:
        d = call("POST", "/api/v3/documents/create",
                 data={"categories": "domain-corpus"},
                 files={"files": (Path(file).name, f, "text/plain")})
    doc_id = d["document_ids"][0]
    save(document_id=doc_id)
    poll(f"/api/v3/documents/{doc_id}/status")


def benchmark(n=10):
    d = call("POST", "/api/v3/benchmarks/create",
             json={"document_ids": [state["document_id"]], "num_questions": int(n)})
    bid = d["benchmark_id"]
    save(benchmark_id=bid)
    poll(f"/api/v3/benchmarks/{bid}/status")
    data = call("GET", f"/api/v3/benchmarks/{bid}/data")
    Path("benchmark_raw.json").write_text(json.dumps(data, indent=2))
    # The response key/field names differ from the cookbook, so find the list generically.
    items = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
    if items is None:
        sys.exit("Couldn't find samples in response; top-level keys: "
                 f"{list(data)}. See benchmark_raw.json and send me a few lines.")
    print("Sample fields:", list(items[0]))
    Path("benchmark.json").write_text(json.dumps(items, indent=2))
    print("Saved benchmark.json - review/edit it, then run: python nugen_align.py upload_benchmark")


def upload_benchmark(path="benchmark.json"):
    with open(path, "rb") as f:
        d = call("POST", "/api/v3/benchmarks/upload",
                 data={"name": "edited benchmark", "document_id": [state["document_id"]],
                       "description": "Edited benchmark"},
                 files={"file": (Path(path).name, f, "application/json")})
    save(benchmark_id=d["benchmark_id"])
    poll(f"/api/v3/benchmarks/{d['benchmark_id']}/status")


def align(base_model="Qwen/Qwen2.5-0.5B-Instruct"):
    d = call("POST", "/api/v3/alignment-projects/create", json={
        "alignment_name": "my-domain-alignment",
        "base_model_id": base_model,
        "document_ids": [state["document_id"]],
        "benchmark_id": state["benchmark_id"],
        "description": "Domain alignment for Nugen exercise",
    })
    aid = d["alignment_id"]
    save(alignment_id=aid)
    poll(f"/api/v3/alignment-projects/{aid}/status")
    models = call("GET", "/api/v3/models/aligned")["domain_aligned_models"]
    print(json.dumps(models, indent=2))
    save(model_id=models[0]["model_id"])  # verify this is the one you just trained


def deploy():
    mid = state["model_id"]
    call("POST", f"/api/v3/models/{mid}/deployment")
    poll(f"/api/v3/models/{mid}/deployment/status")


def ask(q, model=None, stream=True):
    r = requests.post(f"{BASE}/api/v3/inference/chat/completions", headers=H, stream=stream,
                      json={"model": model or state["model_id"],
                            "messages": [{"role": "user", "content": q}],
                            "max_tokens": 400, "temperature": 0.1, "stream": stream})
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            print(line.decode())


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    globals()[cmd](*args)
