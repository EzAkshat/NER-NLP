"""The single-page UI served by ner_app.web."""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ner-app</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#1b1f27;--muted:#667085;--line:#e4e7ec;--accent:#3b5bdb;--accent-text:#fff;--warn:#8a5a00;--warn-bg:#fff6e0;--err:#b42318;--err-bg:#fdecea}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--panel:#1a1d24;--text:#e8eaef;--muted:#9aa3b2;--line:#2b303b;--accent:#7c93ff;--accent-text:#0d1020;--warn:#f0c36b;--warn-bg:#2a2314;--err:#ff8a80;--err-bg:#2c1816}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:2rem 1rem 4rem}
h1{font-size:1.5rem;margin:0}
.sub{color:var(--muted);margin:.25rem 0 1.5rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.25rem;margin-bottom:1rem}
#drop{border:2px dashed var(--line);border-radius:12px;padding:2rem 1rem;text-align:center;cursor:pointer;transition:.15s}
#drop:hover,#drop.over{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,transparent)}
#drop strong{display:block;font-size:1.05rem}
#drop span{color:var(--muted);font-size:.9rem}
.row{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin-top:1rem}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--text);padding:.45rem .9rem}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:var(--accent-text);border-color:var(--accent);font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
#status{color:var(--muted)} #status.err{color:var(--err)}
.banner{border-radius:8px;padding:.6rem .9rem;margin-bottom:1rem;white-space:pre-wrap}
.banner.warn{background:var(--warn-bg);color:var(--warn)}
.stats{display:flex;gap:1.5rem;flex-wrap:wrap;margin:.25rem 0 1rem}
.stat b{display:block;font-size:1.3rem} .stat span{color:var(--muted);font-size:.85rem}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1rem}
.chip{border-radius:999px;padding:.2rem .7rem;font-size:.85rem;border:1px solid hsl(var(--h) 60% 50%);background:hsl(var(--h) 80% 55% / .2)}
.chip.off{opacity:.4;background:transparent;text-decoration:line-through}
#text{white-space:pre-wrap;word-wrap:break-word;max-height:480px;overflow:auto;padding:1rem;border:1px solid var(--line);border-radius:8px;font:14px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace}
mark{color:inherit;background:hsl(var(--h) 80% 55% / .28);border-bottom:2px solid hsl(var(--h) 70% 50%);border-radius:3px;padding:0 1px}
mark.flash{outline:3px solid var(--accent)}
.tools{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 .5rem;align-items:center}
input[type=search]{font:inherit;padding:.4rem .7rem;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--text);min-width:220px}
.tablewrap{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%}
th,td{padding:.4rem .7rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--panel);font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
tbody tr{cursor:pointer} tbody tr:hover{background:color-mix(in srgb,var(--accent) 8%,transparent)}
td.t{white-space:normal}
.tag{border-radius:6px;padding:.05rem .45rem;font-size:.8rem;background:hsl(var(--h) 80% 55% / .25)}
.low{color:var(--warn);font-size:.75rem;margin-left:.4rem}
.note{color:var(--muted);font-size:.85rem}
.hidden{display:none!important}
</style></head><body><main>
<h1>ner-app</h1>
<p class="sub">Local named entity recognition. Files stay on this machine and are not stored.</p>

<div class="card">
 <div id="drop" tabindex="0">
  <strong id="dropTitle">Drop a file here, or click to choose</strong>
  <span>.txt  .md  .html  .docx  .pdf</span>
 </div>
 <input type="file" id="file" class="hidden" accept=".txt,.md,.html,.htm,.docx,.pdf">
 <div class="row">
  <button id="go" class="primary" disabled>Analyze</button>
  <span id="status"></span>
 </div>
 <p class="note" style="margin-bottom:0">On CPU, expect roughly 10 seconds per 6,000 characters. Long files can take minutes.</p>
</div>

<div id="out" class="hidden">
 <div class="card">
  <h2 id="title" style="margin-top:0;font-size:1.1rem"></h2>
  <div id="warnings" class="banner warn hidden"></div>
  <div class="stats" id="stats"></div>
  <div class="note" style="margin-bottom:.4rem">Click a label to show or hide it.</div>
  <div class="chips" id="chips"></div>
  <div id="text"></div>
 </div>
 <div class="card">
  <div class="tools">
   <input type="search" id="q" placeholder="Filter mentions...">
   <span style="flex:1"></span>
   <button id="copy">Copy JSON</button>
   <button id="download">Download JSON</button>
  </div>
  <div class="tablewrap"><table>
   <thead><tr><th>#</th><th>Label</th><th>Text</th><th>Offsets</th><th>Score</th></tr></thead>
   <tbody id="rows"></tbody>
  </table></div>
  <p class="note" style="margin-bottom:0">Scores are model confidence estimates, not probabilities, and can be high for wrong predictions. Offsets refer to the extracted, normalized text, not the original file.</p>
 </div>
</div>

<script>
const $ = id => document.getElementById(id);
const state = {data: null, hidden: new Set(), query: '', file: null};
let timer = null;

function hue(label){ let h = 0; for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360; return h; }
function el(tag, cls, text){ const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

// ---- file selection (all document text is inserted via textContent, never innerHTML)
function setFile(f){
  state.file = f || null;
  $('dropTitle').textContent = f ? f.name : 'Drop a file here, or click to choose';
  $('go').disabled = !f;
  setStatus('', '');
}
function setStatus(msg, cls){ $('status').textContent = msg; $('status').className = cls || ''; }
$('drop').onclick = () => $('file').click();
$('drop').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') $('file').click(); };
$('file').onchange = () => setFile($('file').files[0]);
['dragenter','dragover'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave','drop'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', e => { if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });

// ---- analyze
function stopTimer(){ clearInterval(timer); $('go').disabled = !state.file; }
$('go').onclick = async () => {
  if (!state.file) return;
  $('out').classList.add('hidden'); $('go').disabled = true;
  const t0 = Date.now();
  const tick = () => setStatus(`Analyzing... ${Math.round((Date.now() - t0) / 1000)}s`, '');
  tick(); timer = setInterval(tick, 1000);
  const fd = new FormData(); fd.append('file', state.file);
  let resp;
  try { resp = await fetch('/analyze', {method: 'POST', body: fd}); }
  catch (e) { stopTimer(); setStatus('Could not reach the server.', 'err'); return; }
  const data = await resp.json().catch(() => ({}));
  stopTimer();
  if (!resp.ok) { setStatus(data.detail || ('Error ' + resp.status), 'err'); return; }
  state.data = data; state.hidden = new Set(); state.query = ''; $('q').value = '';
  setStatus(`Done in ${Math.round((Date.now() - t0) / 1000)}s.`, '');
  render();
};

// ---- rendering
function render(){
  const r = state.data.result, doc = r.document;
  $('title').textContent = doc.source_name;
  const w = $('warnings');
  w.textContent = doc.warnings.map(x => `[${x.code}] ${x.message}`).join('\n');
  w.classList.toggle('hidden', doc.warnings.length === 0);

  const stats = $('stats'); stats.replaceChildren();
  for (const [value, label] of [[r.total_mentions, 'entities'], [Object.keys(r.counts_by_label).length, 'labels'],
                                [doc.char_count.toLocaleString(), 'characters'], [doc.detected_format, 'format']]) {
    const s = el('div', 'stat'); s.append(el('b', '', String(value)), el('span', '', label)); stats.append(s);
  }
  renderChips(); renderText(); renderRows();
  $('out').classList.remove('hidden');
}

function renderChips(){
  const box = $('chips'); box.replaceChildren();
  for (const [label, n] of Object.entries(state.data.result.counts_by_label)) {
    const b = el('button', 'chip' + (state.hidden.has(label) ? ' off' : ''), `${label}: ${n}`);
    b.style.setProperty('--h', hue(label));
    b.onclick = () => { state.hidden.has(label) ? state.hidden.delete(label) : state.hidden.add(label); renderChips(); renderText(); renderRows(); };
    box.append(b);
  }
}

function renderText(){
  // Python offsets are code points; JS strings are UTF-16, so slice by code point.
  const cps = Array.from(state.data.text), frag = document.createDocumentFragment();
  let pos = 0;
  for (const m of [...state.data.result.mentions].sort((a, b) => a.start - b.start)) {
    if (m.start < pos) continue;
    frag.append(cps.slice(pos, m.start).join(''));
    const piece = cps.slice(m.start, m.end).join('');
    if (state.hidden.has(m.label)) { frag.append(piece); }
    else {
      const mk = el('mark', '', piece);
      mk.style.setProperty('--h', hue(m.label)); mk.title = `${m.label} #${m.id}`; mk.dataset.id = m.id;
      frag.append(mk);
    }
    pos = m.end;
  }
  frag.append(cps.slice(pos).join(''));
  $('text').replaceChildren(frag);
}

function renderRows(){
  const q = state.query.toLowerCase(), rows = $('rows'); rows.replaceChildren();
  for (const m of state.data.result.mentions) {
    if (state.hidden.has(m.label)) continue;
    if (q && !(m.text.toLowerCase().includes(q) || m.label.toLowerCase().includes(q))) continue;
    const tr = el('tr');
    tr.append(el('td', '', m.id));
    const tl = el('td'); const tag = el('span', 'tag', m.label); tag.style.setProperty('--h', hue(m.label)); tl.append(tag); tr.append(tl);
    tr.append(el('td', 't', m.text), el('td', '', `${m.start}:${m.end}`));
    const sc = el('td', '', m.score == null ? 'n/a' : m.score.toFixed(3));
    if (m.score != null && m.score < 0.5) sc.append(el('span', 'low', 'low'));
    tr.append(sc);
    tr.onclick = () => {
      const mk = document.querySelector(`mark[data-id="${m.id}"]`);
      if (!mk) return;
      mk.scrollIntoView({block: 'center', behavior: 'smooth'});
      mk.classList.add('flash'); setTimeout(() => mk.classList.remove('flash'), 1500);
    };
    rows.append(tr);
  }
}
$('q').oninput = e => { state.query = e.target.value; renderRows(); };

// ---- export
const json = () => JSON.stringify(state.data.result, null, 2);
$('download').onclick = () => {
  const a = el('a'); a.href = URL.createObjectURL(new Blob([json()], {type: 'application/json'}));
  a.download = state.data.result.document.source_name + '.ner.json'; a.click(); URL.revokeObjectURL(a.href);
};
$('copy').onclick = async () => {
  try { await navigator.clipboard.writeText(json()); $('copy').textContent = 'Copied'; }
  catch (e) { $('copy').textContent = 'Copy failed'; }
  setTimeout(() => $('copy').textContent = 'Copy JSON', 1500);
};
</script></main></body></html>
"""