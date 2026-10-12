"""
Minimal chat app on your aligned Nugen model.

  pip install fastapi uvicorn requests
  export NUGEN_API_KEY=...
  uvicorn app:app --reload      # then open http://127.0.0.1:8000

Model ID is read from MODEL_ID env var, else from nugen_state.json (written by nugen_align.py).
"""
import json, os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

API = "https://api.nugen.in/api/v3/inference/chat/completions"
KEY = os.environ["NUGEN_API_KEY"]
STATE = Path("nugen_state.json")
MODEL = os.environ.get("MODEL_ID") or json.loads(STATE.read_text())["model_id"]
SYSTEM = os.environ.get("SYSTEM_PROMPT", "You are a helpful domain assistant. Answer concisely.")

app = FastAPI(title="Nugen aligned-model chat")


class Ask(BaseModel):
    question: str


@app.post("/api/chat")
def chat(body: Ask):
    r = requests.post(
        API,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"model": MODEL,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": body.question}],
              "max_tokens": 400, "temperature": 0.1, "stream": False},
        timeout=120,
    )
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    # OpenAI-compatible shape; return the raw payload too so any confidence field is visible.
    return {"answer": data["choices"][0]["message"]["content"], "raw": data}


PAGE = """<!doctype html><meta charset=utf-8><title>Aligned model chat</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}
#log div{margin:.6rem 0;padding:.6rem .8rem;border-radius:8px}
.u{background:#eef}.a{background:#efe;white-space:pre-wrap}
input{width:78%;padding:.5rem}button{padding:.5rem 1rem}</style>
<h2>Domain-aligned model</h2><div id=log></div>
<input id=q placeholder="Ask a question..." onkeydown="if(event.key=='Enter')go()">
<button onclick=go()>Ask</button>
<script>
async function go(){const q=document.getElementById('q'),log=document.getElementById('log');
const t=q.value.trim();if(!t)return;q.value='';
log.innerHTML+='<div class=u>'+t.replace(/</g,'&lt;')+'</div>';
const d=document.createElement('div');d.className='a';d.textContent='...';log.appendChild(d);
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:t})});const j=await r.json();
d.textContent=r.ok?j.answer:('Error: '+JSON.stringify(j))}catch(e){d.textContent='Error: '+e}}
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
