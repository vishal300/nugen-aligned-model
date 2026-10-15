"""Mother/router: pick experts, call them, escalate on low confidence, synthesize, log."""
import json, math, os, time, uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nugen

CONF_MIN = float(os.environ.get("CONF_MIN", "0.5"))
LOG = Path("logs"); LOG.mkdir(exist_ok=True)
cfg = json.loads(Path("experts.json").read_text())
NAMES = [e["name"] for e in cfg["experts"]]


def _build_index():
    tf = {}
    for e in cfg["experts"]:
        p = Path(e["corpus"])
        c = Counter(nugen.tokens(p.read_text())) if p.exists() else Counter()
        n = sum(c.values()) or 1
        tf[e["name"]] = {w: v / n for w, v in c.items()}
    df = Counter(w for t in tf.values() for w in t)
    idf = {w: math.log(1 + len(tf) / d) for w, d in df.items()}
    return tf, idf


TF, IDF = _build_index()


def route(query):
    """Corpus-derived lexical router: no API call, fully inspectable."""
    toks = set(nugen.tokens(query))
    raw = {n: sum(TF[n].get(w, 0) * IDF.get(w, 0) for w in toks) for n in NAMES}
    total = sum(raw.values())
    if total == 0:
        return [(n, 0.0) for n in NAMES]  # unknown topic -> ask everyone
    ranked = sorted(((n, s / total) for n, s in raw.items()), key=lambda x: -x[1])
    picked = [r for r in ranked if r[1] >= 0.3][:2] or ranked[:1]
    return picked


def _ask_expert(name, query):
    st = nugen.load_state()["experts"].get(name, {})
    if "active_model" not in st:
        raise RuntimeError(f"Expert '{name}' isn't set up. Run: python setup_experts.py "
                           "(in the same folder as experts_state.json)")
    model = st["active_model"]
    text, conf, _ = nugen.chat(model, [{"role": "user", "content": query}])
    return {"expert": name, "answer": text, "confidence": conf, "model": model}


def _synthesize(query, answers):
    body = "\n\n".join(f"[{a['expert']}] {a['answer']}" for a in answers)
    prompt = ("You are an incident triage lead. Combine the expert answers below into one concise "
              "diagnosis with next steps. Use only what the experts said and note disagreements.\n\n"
              f"Problem: {query}\n\nExpert answers:\n{body}")
    msgs = [{"role": "user", "content": prompt}]
    if os.environ.get("SYNTH_BASE_URL"):
        return nugen.chat(os.environ["SYNTH_MODEL"], msgs, 600, os.environ["SYNTH_BASE_URL"],
                          os.environ["SYNTH_API_KEY"])[0]
    model = os.environ.get("SYNTH_MODEL") or nugen.load_state()["experts"][answers[0]["expert"]]["active_model"]
    return nugen.chat(model, msgs, 600)[0]


def answer(query):
    qid = uuid.uuid4().hex[:12]
    routed = route(query)
    names = [n for n, _ in routed]
    with ThreadPoolExecutor() as ex:
        answers = list(ex.map(lambda n: _ask_expert(n, query), names))
    escalated = False
    weak = [a for a in answers if a["confidence"] is not None and a["confidence"] < CONF_MIN]
    if weak and len(names) < len(NAMES):  # escalate to the next-best unused expert
        extra = next(n for n in NAMES if n not in names)
        answers.append(_ask_expert(extra, query)); escalated = True
    final = answers[0]["answer"] if len(answers) == 1 else _synthesize(query, answers)
    rec = {"id": qid, "ts": time.time(), "query": query, "route": routed,
           "escalated": escalated, "answers": answers, "final": final}
    with open(LOG / "queries.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
