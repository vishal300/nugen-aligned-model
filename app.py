"""Incident triage web app. Run: uvicorn app:app --reload  (needs NUGEN_API_KEY + experts_state.json)"""
import json, time, uuid
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import nugen, router

app = FastAPI(title="Nugen multi-expert triage")
hits = defaultdict(list)
LIMIT, WINDOW = 20, 3600  # 20 questions per IP per hour protects your credits


class Ask(BaseModel):
    question: str


class Feedback(BaseModel):
    query_id: str
    expert: str
    rating: int  # 1 or -1
    correction: str = ""


@app.post("/api/ask")
def ask(body: Ask, req: Request):
    ip, now = req.client.host, time.time()
    hits[ip] = [t for t in hits[ip] if now - t < WINDOW]
    if len(hits[ip]) >= LIMIT:
        raise HTTPException(429, "Rate limit reached, try again later.")
    hits[ip].append(now)
    q = body.question.strip()
    if not q or len(q) > 2000:
        raise HTTPException(400, "Question must be 1-2000 characters.")
    try:
        return router.answer(q)
    except Exception as e:
        raise HTTPException(502, str(e)[:300])


@app.post("/api/feedback")
def feedback(f: Feedback):
    Path("logs").mkdir(exist_ok=True)
    rec = {**f.model_dump(), "id": uuid.uuid4().hex[:12], "ts": time.time()}
    qfile = Path("logs/queries.jsonl")
    q = next((json.loads(l) for l in qfile.read_text().splitlines() if f.query_id in l), None) \
        if qfile.exists() else None
    rec["query"] = q["query"] if q else ""
    with open("logs/feedback.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return {"ok": True}


@app.get("/api/stats")
def stats():
    st = nugen.load_state()["experts"]
    fb = [json.loads(l) for l in Path("logs/feedback.jsonl").read_text().splitlines()] \
        if Path("logs/feedback.jsonl").exists() else []
    return {n: {"version": s.get("version"), "active_model": s.get("active_model"),
                "thumbs_up": sum(1 for f in fb if f["expert"] == n and f["rating"] > 0),
                "thumbs_down": sum(1 for f in fb if f["expert"] == n and f["rating"] < 0),
                "history": s.get("history", [])} for n, s in st.items()}


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Multi-expert incident triage</title>
<style>body{font-family:system-ui;max-width:780px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:90px;padding:.5rem}button{padding:.4rem .9rem;margin:.2rem}
.card{border:1px solid #ccc;border-radius:8px;padding:.7rem;margin:.7rem 0;white-space:pre-wrap}
.chip{background:#eef;border-radius:12px;padding:2px 10px;margin-right:6px;font-size:.85rem}
.final{background:#efe}small{color:#555}</style>
<h2>Incident triage: router + expert models</h2>
<p><small>A router sends your problem to domain-aligned expert models (backend, database, cloud, security),
escalates on low confidence, and merges their answers. Thumbs-down + a correction feeds the next re-alignment.</small></p>
<textarea id=q placeholder="e.g. Django API times out after we added an index and moved to Azure"></textarea>
<button onclick=go()>Triage</button><div id=out></div>
<script>
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
async function go(){const out=document.getElementById('out');out.innerHTML='Working...';
 const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});
 const j=await r.json(); if(!r.ok){out.textContent='Error: '+(j.detail||r.status);return}
 let h='<div>Routed to: '+j.route.map(x=>'<span class=chip>'+x[0]+' ('+Math.round(x[1]*100)+'%)</span>').join('')+(j.escalated?'<span class=chip>escalated: low confidence</span>':'')+'</div>';
 h+='<div class="card final"><b>Combined answer</b>\\n'+esc(j.final)+'</div>';
 j.answers.forEach(a=>{h+='<div class=card><b>'+a.expert+'</b>'+(a.confidence!=null?' <small>confidence '+a.confidence.toFixed(2)+'</small>':'')+'\\n'+esc(a.answer)+
  '\\n<button onclick="fb(\\''+j.id+'\\',\\''+a.expert+'\\',1)">👍</button><button onclick="fb(\\''+j.id+'\\',\\''+a.expert+'\\',-1)">👎</button></div>'});
 out.innerHTML=h}
async function fb(id,expert,rating){let c='';if(rating<0)c=prompt('What should the correct answer be? (optional, used to improve this expert)')||'';
 await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query_id:id,expert,rating,correction:c})});alert('Thanks, recorded.')}
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
