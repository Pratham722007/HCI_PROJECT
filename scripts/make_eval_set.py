"""Phase 10.1-10.2: blinded CBM-vs-flat explanation pairs + a standalone rater form.

Run: .venv/bin/python -E -P scripts/make_eval_set.py --tag 12ch [--n 150]
Writes reports/<tag>/human_eval/{items.json, rater_form.html} and key_PRIVATE.json (A/B -> system; gitignored, do not
share with raters). Pairs are stratified by true class and CBM-correct (as sampled in scripts/explain.py); A/B order is
randomised with a saved seed. The rater form runs offline in any browser and downloads a CSV of ratings.
"""
import argparse, html, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="12ch"); ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=20261019)
a = ap.parse_args()
src = os.path.join(ROOT, "reports", a.tag, "explanations.jsonl")
recs = [json.loads(l) for l in open(src)][: a.n]
out = os.path.join(ROOT, "reports", a.tag, "human_eval"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(a.seed)
CRITERIA = ["faithfulness", "plausibility", "clarity", "actionability"]


def scrub(t):
    """Remove wording that names the system architecture (blinding)."""
    t = re.sub(r"\bLevel \d,?\s*", "", t)
    return re.sub(r"(?i)a neural network mapped the EEG embedding directly to emotion classes\.\s*", "", t).strip()


items, key = [], {}
for i, r in enumerate(recs):
    cbm = dict(facts=scrub(r["facts"]["cbm"]), text=scrub(r["texts"]["cbm"]), emotion=r["cbm_emotion"])
    flat = dict(facts=scrub(r["facts"]["flat"]), text=scrub(r["texts"]["flat"]), emotion=r["flat_emotion"])
    swap = bool(rng.integers(2))
    A, B = (flat, cbm) if swap else (cbm, flat)
    items.append(dict(id=i, A=A, B=B))
    key[i] = dict(A="flat" if swap else "cbm", B="cbm" if swap else "flat", dataset=r["dataset"], subject=r["subject"],
                  true_emotion=r["true_emotion"], cbm_correct=(r["cbm_emotion"] == r["true_emotion"]) if r["true_emotion"] else None,
                  flat_correct=(r["flat_emotion"] == r["true_emotion"]) if r["true_emotion"] else None)
json.dump(items, open(os.path.join(out, "items.json"), "w"), indent=1)
json.dump(dict(seed=a.seed, source=os.path.relpath(src, ROOT), key=key), open(os.path.join(out, "key_PRIVATE.json"), "w"), indent=1)
print("A=cbm share:", np.mean([k["A"] == "cbm" for k in key.values()]))

DEFS = {"faithfulness": "The explanation only states things supported by the evidence shown, and its reasoning matches the evidence.",
        "plausibility": "The reasoning is psychologically/physiologically sensible.",
        "clarity": "Easy to read and understand.",
        "actionability": "It would help a clinician decide whether to trust or check this prediction."}
page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explanation Rating</title><style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--card:#ffffff;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--line:#3a3936;--card:#232321;--acc:#3987e5}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin-inline:auto}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:720px){.pair{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
pre{white-space:pre-wrap;font:13px/1.4 ui-monospace,monospace;color:var(--mut);margin:0 0 8px}
table{border-collapse:collapse;width:100%}td{padding:4px 2px;border-top:1px solid var(--line)}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:15px;cursor:pointer}
.nav{display:flex;gap:8px;align-items:center;margin:12px 0}.mut{color:var(--mut)}
</style></head><body>
<h1>Explanation rating</h1>
<p class="mut">Each screen shows two explanations (A, B) of an EEG emotion prediction, each with the evidence it was given. Rate each from
1 (poor) to 5 (excellent). Ratings stay in this browser until you press <b>Download CSV</b> and send the file back.</p>
<details><summary>Criteria</summary><ul>__DEFS__</ul></details>
<p><label>Your rater ID: <input id="rid" size="10"></label></p>
<div class="nav"><button id="prev">&larr; Prev</button><span id="pos"></span><button id="next">Next &rarr;</button>
<span style="flex:1"></span><button id="dl">Download CSV</button></div>
<div id="item"></div>
<script>
const ITEMS=__ITEMS__, CRIT=__CRIT__;
let i=0, R={};
try{R=JSON.parse(localStorage.getItem("ratings")||"{}");document.getElementById("rid").value=localStorage.getItem("rid")||""}catch(e){}
function save(){try{localStorage.setItem("ratings",JSON.stringify(R));localStorage.setItem("rid",document.getElementById("rid").value)}catch(e){}}
function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML}
function side(it,s){let h=`<div class="card"><h3>${s}</h3><pre>${esc(it[s].facts)}</pre><p>${esc(it[s].text)}</p><table>`;
 for(const c of CRIT){h+=`<tr><td>${c}</td><td>`;for(let v=1;v<=5;v++){const k=`${it.id}|${s}|${c}`;
 h+=`<label><input type="radio" name="${k}" value="${v}" ${R[k]==v?"checked":""} onchange="R['${k}']=${v};save()"> ${v}</label> `}h+="</td></tr>"}
 return h+"</table></div>"}
function show(){const it=ITEMS[i];document.getElementById("item").innerHTML=`<div class="pair">${side(it,"A")}${side(it,"B")}</div>
 <p><label>Comment (optional)<br><textarea rows="2" style="width:100%" oninput="R['${it.id}|comment']=this.value;save()">${esc(R[it.id+"|comment"]||"")}</textarea></label></p>`;
 const done=Object.keys(R).filter(k=>!k.endsWith("comment")).length;document.getElementById("pos").textContent=`Item ${i+1} / ${ITEMS.length} (${done} ratings)`}
document.getElementById("prev").onclick=()=>{if(i>0){i--;show()}};document.getElementById("next").onclick=()=>{if(i<ITEMS.length-1){i++;show()}};
document.getElementById("dl").onclick=()=>{const rid=document.getElementById("rid").value||"anon";let csv="rater_id,item_id,side,criterion,score,comment\\n";
 for(const it of ITEMS){for(const s of ["A","B"])for(const c of CRIT){const v=R[`${it.id}|${s}|${c}`];if(v)csv+=`${rid},${it.id},${s},${c},${v},\\n`}
 const cm=R[it.id+"|comment"];if(cm)csv+=`${rid},${it.id},,comment,,"${cm.replace(/"/g,'""')}"\\n`}
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download=`ratings_${rid}.csv`;a.click()};
show();
</script></body></html>"""
page = (page.replace("__ITEMS__", json.dumps(items)).replace("__CRIT__", json.dumps(CRITERIA))
        .replace("__DEFS__", "".join(f"<li><b>{k}</b>: {html.escape(v)}</li>" for k, v in DEFS.items())))
open(os.path.join(out, "rater_form.html"), "w").write(page)
print("wrote", out, len(items), "items")
