#!/usr/bin/env bash
# Rebuild h-cbm-eeg.pdf (two passes so the contents page numbers are correct).
set -e
cd "$(dirname "$0")/../.."
PY=.venv-report/bin/python
[ -x "$PY" ] || { python3 -m venv .venv-report && .venv-report/bin/pip install -q reportlab pypdf pillow; }
[ -f docs/report/progression.png ] || .venv/bin/python -E -P docs/report/make_chart.py
$PY docs/report/make_report.py h-cbm-eeg.pdf >/dev/null
P=$($PY -c "
from pypdf import PdfReader; import json
r=PdfReader('h-cbm-eeg.pdf'); n=len(r.pages); st={}
for i,p in enumerate(r.pages):
    t=p.extract_text()
    if 'OVERVIEW' in t and 'O' not in st: st['O']=i+1
    for k in range(1,6):
        if f'PART {k} OF 5' in t: st[str(k)]=i+1
order=['O','1','2','3','4','5']; st['end']=n+1; nxt=dict(zip(order, order[1:]+['end']))
print(json.dumps({k:(f'{st[k]}' if st[nxt[k]]-1==st[k] else f'{st[k]}–{st[nxt[k]]-1}') for k in order}))")
$PY docs/report/make_report.py h-cbm-eeg.pdf "$P"
echo "pages: $P"
