"""Close the loop for one expert: thumbs-down + corrections -> new corpus -> re-align -> compare -> promote.
usage: python improve.py backend
"""
import json, sys
from pathlib import Path
import nugen

name = sys.argv[1]
cfg = json.loads(Path("experts.json").read_text())
exp = next(e for e in cfg["experts"] if e["name"] == name)
state = nugen.load_state(); st = state["experts"][name]

fb = [json.loads(l) for l in Path("logs/feedback.jsonl").read_text().splitlines()] \
    if Path("logs/feedback.jsonl").exists() else []
used = set(st.get("used_feedback", []))
new = [f for f in fb if f["expert"] == name and f["rating"] < 0 and f.get("correction")
       and f["id"] not in used]
if not new:
    sys.exit(f"No new corrections for '{name}'.")

q = {r["id"]: r["query"] for r in
     (json.loads(l) for l in Path("logs/queries.jsonl").read_text().splitlines())}
block = "\n\nVerified corrections from production feedback:\n" + "\n".join(
    f"Question: {f.get('query') or q.get(f['query_id'], '?')}\nCorrect answer: {f['correction']}\n" for f in new)
corpus_path = Path(exp["corpus"])
text = corpus_path.read_text()
if block.strip() not in text:  # idempotent: safe to re-run after a failure
    corpus_path.write_text(text.rstrip() + block)

# The API needs >= 500 words per document, so upload the full updated corpus
# (original knowledge + corrections) and use it in place of the old document.
print(f"Aligning {name} v{st['version'] + 1} with {len(new)} corrections")
doc_ids = [nugen.upload_doc(corpus_path)]
aid = nugen.create_alignment(f"triage-{name}-v{st['version'] + 1}", cfg["base_model"],
                             doc_ids, st["benchmark_id"])
new_model = nugen.model_for_alignment(aid)
nugen.deploy(new_model)

samples = nugen.benchmark_samples(st["benchmark_id"])
def score(model):
    return sum(nugen.f1(nugen.chat(model, [{"role": "user", "content": s["instruction"]}])[0],
                        s["response"]) for s in samples) / len(samples)
old_s, new_s = score(st["active_model"]), score(new_model)
print(f"benchmark F1  old={old_s:.3f}  new={new_s:.3f}")

st.setdefault("history", []).append({"version": st["version"] + 1, "old": old_s, "new": new_s,
                                     "promoted": new_s >= old_s, "model": new_model})
st["used_feedback"] = list(used | {f["id"] for f in new})
if new_s >= old_s:
    st.update(active_model=new_model, version=st["version"] + 1, document_ids=doc_ids)
    print("Promoted. (Undeploy the old model on the platform to free a deployment slot.)")
else:
    print("Not promoted: new model scored lower.")
nugen.save_state(state)
