"""Align + deploy every expert in experts.json. Resumable: re-run after a failure."""
import json
from pathlib import Path
import nugen

cfg = json.loads(Path("experts.json").read_text())
state = nugen.load_state()

for e in cfg["experts"]:
    name, st = e["name"], state["experts"].setdefault(e["name"], {})
    print(f"== {name}")
    if "document_ids" not in st:
        st["document_ids"] = [nugen.upload_doc(e["corpus"])]; nugen.save_state(state)
    if "benchmark_id" not in st:
        st["benchmark_id"] = nugen.create_benchmark(st["document_ids"], 10); nugen.save_state(state)
    if "alignment_id" not in st:
        st["alignment_id"] = nugen.create_alignment(
            f"triage-{name}-v1", cfg["base_model"], st["document_ids"], st["benchmark_id"])
        st["model_id"] = nugen.model_for_alignment(st["alignment_id"])
        st["active_model"], st["version"] = st["model_id"], 1
        nugen.save_state(state)
    nugen.deploy(st["active_model"])
    print(f"   ready: {st['active_model']}")
print("All experts ready.")
