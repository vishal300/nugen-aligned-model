"""Thin Nugen API helpers shared by setup, router and improve scripts."""
import json, os, re, sys, time
from pathlib import Path
import requests

BASE = os.environ.get("NUGEN_BASE", "https://api.nugen.in")
STATE_PATH = Path("experts_state.json")
DONE = {"READY", "COMPLETED", "DONE", "SUCCESS", "DEPLOYED"}
BAD = {"FAILED", "ERROR"}


def _h():
    return {"Authorization": f"Bearer {os.environ['NUGEN_API_KEY']}"}


def load_state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"experts": {}}


def save_state(s):
    STATE_PATH.write_text(json.dumps(s, indent=2))


def call(method, path, **kw):
    r = requests.request(method, f"{BASE}{path}", headers=_h(), timeout=120, **kw)
    if not r.ok:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text}")
    return r.json()


def poll(path, every=10, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = call("GET", path)
        s = str(d.get("status", "")).upper()
        print(f"  {path} -> {s}")
        if s in DONE:
            return d
        if s in BAD:
            raise RuntimeError(f"failed: {d}")
        time.sleep(every)
    raise TimeoutError(path)


def upload_doc(file):
    with open(file, "rb") as f:
        d = call("POST", "/api/v3/documents/create", data={"categories": "domain-corpus"},
                 files={"files": (Path(file).name, f, "text/plain")})
    doc_id = d["document_ids"][0]
    poll(f"/api/v3/documents/{doc_id}/status")
    return doc_id


def create_benchmark(doc_ids, n=10):
    bid = call("POST", "/api/v3/benchmarks/create",
               json={"document_ids": doc_ids, "num_questions": n})["benchmark_id"]
    poll(f"/api/v3/benchmarks/{bid}/status")
    return bid


def benchmark_samples(bid):
    data = call("GET", f"/api/v3/benchmarks/{bid}/data")
    items = data if isinstance(data, list) else next(
        v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict))
    return [{"instruction": s["instruction"], "response": s["response"]} for s in items]


def create_alignment(name, base_model, doc_ids, benchmark_id):
    aid = call("POST", "/api/v3/alignment-projects/create", json={
        "alignment_name": name, "base_model_id": base_model, "document_ids": doc_ids,
        "benchmark_id": benchmark_id, "description": f"Triage expert: {name}"})["alignment_id"]
    poll(f"/api/v3/alignment-projects/{aid}/status")
    return aid


def model_for_alignment(aid):
    models = call("GET", "/api/v3/models/aligned")["domain_aligned_models"]
    return next(m["model_id"] for m in models if m["alignment_id"] == aid)


def deploy(model_id):
    try:
        call("POST", f"/api/v3/models/{model_id}/deployment")
    except RuntimeError as e:
        if "already deployed" not in str(e):
            raise
    poll(f"/api/v3/models/{model_id}/deployment/status")


def _find_conf(obj):
    """Confidence field name isn't documented in what I could read; search for it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "confidence" in k.lower() and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            c = _find_conf(v)
            if c is not None:
                return c
    elif isinstance(obj, list):
        for v in obj:
            c = _find_conf(v)
            if c is not None:
                return c
    return None


def chat(model, messages, max_tokens=400, base_url=None, api_key=None):
    """Returns (text, confidence_or_None, raw). Works for Nugen or any OpenAI-compatible host."""
    if base_url:
        url, headers = f"{base_url}/chat/completions", {"Authorization": f"Bearer {api_key}"}
    else:
        url, headers = f"{BASE}/api/v3/inference/chat/completions", _h()
    r = requests.post(url, headers=headers, timeout=120, json={
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.1, "stream": False})
    if not r.ok:
        raise RuntimeError(f"chat {r.status_code}: {r.text}")
    raw = r.json()
    try:
        text = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = str(raw.get("content") or raw)
    return text, _find_conf(raw), raw


def tokens(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)
