/* app.js: Incremental Circle Fit web app (page logic and Plotly figures).
 * All maths runs in Python (modal_core.py via web/api.py) in web/worker.js. */
'use strict';

const VERSION = '1';
const COL = {
  measured: '#9aa0aa', residual: '#7c4dbe', sum: '#e65100', smooth: '#b35c00',
  after: '#2e7d32', theory: '#1565c0', fr: '#c62828', mode: '#e65100',
  data: '#1565c0', strongest: '#2e7d32', poor: '#c62828', phase: '#8e24aa',
};
const PCFG = {
  responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
};
const $ = (id) => document.getElementById(id);

// ─── Python worker ─────────────────────────────────────────────────────────
const worker = new Worker('web/worker.js?v=' + VERSION);
const pending = new Map();
let seq = 0, busy = 0;

worker.onmessage = (e) => {
  const d = e.data;
  if (d.type === 'progress') { $('load-title').textContent = d.msg; return; }
  if (d.type === 'ready') { onReady(); return; }
  if (d.type === 'fatal') {
    document.querySelector('#loading .spin').style.display = 'none';
    $('load-title').textContent = 'Python failed to load';
    document.querySelector('#loading .load-sub').textContent = d.error;
    return;
  }
  const p = pending.get(d.id);
  if (!p) return;
  pending.delete(d.id);
  setBusy(-1);
  d.ok ? p.resolve(d.result) : p.reject(new Error(d.error));
};

function rpc(msg, transfer) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    setBusy(+1);
    worker.postMessage({ id, ...msg }, transfer || []);
  });
}
async function call(fn, ...args) {
  const r = await rpc({ fn, args: args.map(a => (a === undefined ? null : a)) });
  return r == null ? null : JSON.parse(r);
}
let busyTimer = null;
function setBusy(d) {
  busy += d;
  clearTimeout(busyTimer);
  if (busy > 0) busyTimer = setTimeout(() => { document.body.style.cursor = 'progress'; updateStatusRight(); }, 250);
  else { document.body.style.cursor = ''; updateStatusRight(); }
}
/* Run an async refresh so that calls made while it runs collapse into one
   more run afterwards (sliders and point scrubbing fire many events). */
function coalesce(fn) {
  let running = false, again = false;
  const run = async () => {
    if (running) { again = true; return; }
    running = true;
    try { do { again = false; await fn(); } while (again); }
    catch (err) { status(err.message, 'err'); }
    finally { running = false; }
  };
  return run;
}

// ─── State ─────────────────────────────────────────────────────────────────
const st = {
  ready: false, info: null, folder: null, f: null, measured: null, top: null,
  step: null,            // null = latest; else number of modes shown
  view: null,            // {which:'proposal'} | {which:'mode', n} | null
  point: null,           // point index for the circle plot (null = strongest)
  fit: null,             // last fit_view result
  specs: null, sliders: {}, free: new Set(['E', 'k']), plate: null,
  theory: null, theoryPreview: null, theoryOn: true, band: null,
  opts: { fmin: 20, fmax: 7000, flip: false, refine: true, pad: 0 },
  showMeasured: true,
};

function status(msg, kind) {
  const el = $('status');
  el.className = 'status-txt' + (kind ? ' ' + kind : '');
  el.textContent = msg;
}
function updateStatusRight() {
  const i = st.info;
  const parts = [];
  if (busy > 0) parts.push('working…');
  if (i) parts.push(`${i.n_points} points · Δf ${i.df.toFixed(4)} Hz`
    + (i.pad_s ? ` (${i.record_s.toFixed(2)} s + ${i.added_s.toFixed(1)} s of zeros)` : '')
    + ` · ${st.top ? st.top.total : 0} modes` + (st.opts.flip ? ' · sign flipped' : '')
    + (i.raw && i.raw.clipped_response.length + i.raw.clipped_hammer.length
       ? ` · ⚠ ${i.raw.clipped_response.length + i.raw.clipped_hammer.length} clipped recordings` : ''));
  $('status-right').textContent = parts.join('  ·  ');
}

// ─── Startup ───────────────────────────────────────────────────────────────
async function onReady() {
  st.ready = true;
  $('loading').classList.add('gone');
  st.specs = await call('specs');
  loadPrefs();
  buildSliders();
  await previewTheory();
}

function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem('icf:prefs') || '{}');
    if (p.sliders) st.sliders = p.sliders;
    if (p.free) st.free = new Set(p.free);
    if (p.opts) Object.assign(st.opts, p.opts);
    if (typeof p.theoryOn === 'boolean') st.theoryOn = p.theoryOn;
    if (p.tuneOpen) $('tune').classList.remove('hidden');
  } catch (e) { /* storage unavailable: defaults */ }
  for (const s of st.specs) if (!(s.key in st.sliders)) st.sliders[s.key] = s.value;
  $('theory-on').checked = st.theoryOn;
  $('opt-fmin').value = st.opts.fmin; $('opt-fmax').value = st.opts.fmax;
  $('opt-flip').checked = st.opts.flip; $('opt-refine').checked = st.opts.refine;
  $('opt-pad').value = st.opts.pad || 0;
  syncTuneButton();
}
function savePrefs() {
  try {
    localStorage.setItem('icf:prefs', JSON.stringify({
      sliders: st.sliders, free: [...st.free], opts: st.opts, theoryOn: st.theoryOn,
      tuneOpen: !$('tune').classList.contains('hidden'),
    }));
  } catch (e) { /* ignore */ }
}

// ─── Opening a test folder ─────────────────────────────────────────────────
$('dir-input').addEventListener('change', async (ev) => {
  const files = [...ev.target.files];
  ev.target.value = '';
  ev.target.blur();
  if (!st.ready) { status('Python is still loading; try again in a moment.', 'err'); return; }
  const tests = new Map();
  const get = (d) => { if (!tests.has(d)) tests.set(d, { trf: [], raw: [], tpl: null }); return tests.get(d); };
  for (const f of files) {
    const p = f.webkitRelativePath || f.name;
    let m = p.match(/^(.*)\/TRF\/([^/]+\.trf)$/i);
    if (m) { get(m[1]).trf.push({ file: f, name: m[2] }); continue; }
    m = p.match(/^(.*)\/raw\/([^/]+\.wav)$/i);
    if (m) { get(m[1]).raw.push({ file: f, name: m[2] }); continue; }
    m = p.match(/^(.*)\/template\.json$/i);
    if (m) get(m[1]).tpl = f;
  }
  const list = [...tests].filter(([, t]) => t.trf.length).sort((a, b) => a[0].localeCompare(b[0]));
  if (!list.length) { status('No TRF/*.trf files in that folder. Pick a test folder (it contains TRF/).', 'err'); return; }
  if (list.length === 1) { openTest(...list[0]); return; }
  const box = $('pick-list');
  box.innerHTML = '';
  for (const [dir, t] of list) {
    const b = document.createElement('button');
    b.className = 'pick-item';
    b.innerHTML = `${esc(dir.split('/').pop())}<small>${esc(dir)} · ${t.trf.length} points${t.raw.length ? ` · ${t.raw.length} recordings` : ''}</small>`;
    b.onclick = () => { closeModals(); openTest(dir, t); };
    box.appendChild(b);
  }
  openModal('pick-modal');
});

async function openTest(dir, t) {
  const name = dir.split('/').pop();
  status(`Reading ${name}…`);
  try {
    const files = [];
    for (const x of t.trf) files.push({ path: 'TRF/' + x.name, buffer: await x.file.arrayBuffer() });
    if (t.tpl) files.push({ path: 'template.json', buffer: await t.tpl.arrayBuffer() });
    for (const x of t.raw) files.push({ path: 'raw/' + x.name, buffer: await x.file.arrayBuffer() });
    await rpc({ fn: '__write', name, files }, files.map(f => f.buffer));
    st.folder = name;
    await loadSession(null);
  } catch (err) { status(err.message, 'err'); }
}

async function loadSession(replay) {
  const plate = st.theoryOn ? await plateFromSliders() : null;
  if (st.opts.pad > 0) status(`Rebuilding the FRFs with ${st.opts.pad} s of zeros added…`);
  const info = await call('load', '/data/' + st.folder, st.opts.fmin, st.opts.fmax,
    st.opts.refine, st.opts.flip, plate ? JSON.stringify(plate) : null, st.opts.pad);
  st.info = info;
  st.step = null; st.view = null; st.point = null; st.fit = null;
  st.theory = info.theory; st.theoryPreview = null;
  $('dataset-name').textContent = info.name;
  $('welcome').classList.add('hidden');
  $('workspace').classList.remove('hidden');
  const sl = $('point-slider');
  sl.max = info.n_points - 1;
  const t = await call('top', null, true, true);
  st.f = t.f; st.measured = t.measured;
  let msg = null;
  if (replay) {
    const r = await call('replay', replay);
    msg = `Re-fitted ${r.done} modes` + (r.failed.length ? `; could not fit ${r.failed.join(', ')} Hz` : '') + '.';
  }
  await refreshTop();
  clearFit();
  if (st.top.total) { st.view = { which: 'mode', n: st.top.total }; await refreshFit(); }
  await refreshPairs();
  // only now offer to restore, so a quick click can't race the loading above
  if (msg) status(msg, 'ok');
  else {
    status(`Loaded ${info.name}: ${info.n_points} points. Click a peak or press n.` + clipNote());
    offerRestore();
  }
}

function offerRestore() {
  let saved = '';
  try { saved = localStorage.getItem('icf:modes:' + st.info.name) || ''; } catch (e) { /* ignore */ }
  const n = saved ? saved.split(',').filter(Boolean).length : 0;
  if (!n) return;
  const el = $('status');
  el.className = 'status-txt';
  el.textContent = `Loaded ${st.info.name}. Your previous session here had ${n} modes.` + clipNote();
  const b = document.createElement('button');
  b.className = 'tb-btn accent';
  b.textContent = `Restore ${n} modes`;
  b.onclick = async () => {
    const r = await call('replay', saved);
    status(`Restored ${r.done} modes.`, 'ok');
    await refreshTop();
    st.view = st.top.total ? { which: 'mode', n: st.top.total } : null;
    await refreshFit();
    await refreshPairs();
  };
  el.appendChild(b);
}
function clipNote() {
  const r = st.info && st.info.raw;
  if (!r) return '';
  const parts = [];
  if (r.clipped_response.length) parts.push(`${r.clipped_response.length} of ${r.n} recordings clip on the accelerometer channel`);
  if (r.clipped_hammer.length) parts.push(`${r.clipped_hammer.length} of ${r.n} clip on the hammer channel`);
  return parts.length ? `  ⚠ ${parts.join('; ')} (reach full scale), which distorts those FRFs.` : '';
}
function saveReplay() {
  if (!st.info || !st.top) return;
  const s = st.top.modes.map(m => `${m.peak_hz.toFixed(4)}:${m.side}`).join(',');
  try { localStorage.setItem('icf:modes:' + st.info.name, s); } catch (e) { /* ignore */ }
}

// ─── Plot helpers ──────────────────────────────────────────────────────────
function layout(title, xl, yl, extra = {}) {
  const { xaxis = {}, yaxis = {}, ...rest } = extra;
  const ax = { gridcolor: '#e3e6ea', zerolinecolor: '#c0c4cc', tickfont: { size: 9 },
               titlefont: { size: 10 }, automargin: true };
  return {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { size: 10, family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif', color: '#1a1a1a' },
    title: { text: title, font: { size: 11 }, x: 0.02, xanchor: 'left', y: 0.98, yanchor: 'top' },
    xaxis: { title: { text: xl }, ...ax, ...xaxis },
    yaxis: { title: { text: yl }, ...ax, ...yaxis },
    margin: { l: 52, r: 10, t: 34, b: 38 },
    showlegend: true,
    legend: { font: { size: 9 }, bgcolor: 'rgba(255,255,255,0.75)', x: 0.01, xanchor: 'left', y: 0.99, yanchor: 'top' },
    hoverlabel: { font: { size: 10 } },
    ...rest,
  };
}
const vline = (x, color, dash = 'solid', width = 1, opacity = 1) => ({
  type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1,
  line: { color, dash, width }, opacity, layer: 'above',
});
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function theoryShapes(th, numbers) {
  const shapes = [], ann = [];
  if (!th) return { shapes, ann };
  let last = null;
  th.forEach((f, i) => {
    if (f == null) return;
    shapes.push(vline(f, COL.theory, 'dash', 1, 0.4));
    if (numbers && (last == null || Math.log10(f / last) > 0.03)) {   // keep numbers legible
      last = f;
      ann.push({ x: Math.log10(f), xref: 'x', y: 1, yref: 'paper', yanchor: 'bottom', text: String(i + 1),
                 showarrow: false, font: { size: 8, color: COL.theory } });
    }
  });
  return { shapes, ann };
}

// ─── Top FRF plot ──────────────────────────────────────────────────────────
const refreshTop = coalesce(async () => {
  if (!st.info) return;
  st.top = await call('top', st.step, true, false);
  renderTop();
  renderModes();
  updateStatusRight();
});

function shownTheory() {
  if (!st.theoryOn) return null;
  return st.theoryPreview || st.theory;
}

function topShapes() {
  const t = st.top, shapes = [], ann = [];
  t.modes.forEach((m, i) => {
    if (i < t.n) shapes.push(vline(m.fr, COL.mode, 'solid', 1, 0.45));
    else shapes.push(vline(m.fr, '#7f7f7f', 'dot', 1, 0.8));
  });
  if (st.step != null && t.n) shapes.push(vline(t.modes[t.n - 1].fr, COL.fr, 'dash', 1.5));
  if (t.proposal) {
    shapes.push(vline(t.proposal.fr, COL.fr, 'dash', 1.2));
    shapes.push(vline(t.proposal.peak_hz, '#444', 'dot', 1));
  }
  if (st.band) shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: st.band[0], x1: st.band[1], y0: 0, y1: 1,
                             fillcolor: COL.after, opacity: 0.07, line: { width: 0 }, layer: 'below' });
  const th = theoryShapes(shownTheory(), true);
  return { shapes: shapes.concat(th.shapes), annotations: ann.concat(th.ann) };
}

function topTitle() {
  const t = st.top, i = st.info;
  const res = `residual ${t.residual_db >= 0 ? '+' : ''}${t.residual_db.toFixed(2)} dB re measured (mean ${i.fmin}–${i.fmax} Hz)`;
  if (st.step != null) {
    const m = t.n ? `after mode ${t.n}, ${t.modes[t.n - 1].fr.toFixed(2)} Hz` : 'no modes removed';
    return `<span style="color:${COL.fr}"><b>Viewing step ${t.n} of ${t.total}</b> (${m}) · → or Latest to return · ${res}</span>`;
  }
  return `<b>${esc(i.name)}</b> · ${t.total} mode${t.total === 1 ? '' : 's'} accepted · ${res}`;
}

function renderTop() {
  const t = st.top, tr = [];
  if (st.showMeasured) tr.push({ x: st.f, y: st.measured, type: 'scattergl', mode: 'lines', name: 'measured',
            line: { color: COL.measured, width: 1 }, hovertemplate: '%{y:.1f} dB' });
  if (t.sum) tr.push({ x: st.f, y: t.sum, type: 'scattergl', mode: 'lines', name: `sum of ${t.n} modes (at bins)`,
                       line: { color: COL.sum, width: 1 }, visible: 'legendonly', hovertemplate: '%{y:.1f} dB' });
  if (t.smooth) tr.push({ x: t.smooth.f, y: t.smooth.db, type: 'scattergl', mode: 'lines',
                          name: `fitted model, ${t.n} modes (smooth)`, line: { color: COL.smooth, width: 1.3 },
                          hovertemplate: '%{y:.1f} dB' });
  tr.push({ x: st.f, y: t.residual, type: 'scattergl', mode: 'lines', name: `residual after ${t.n} modes`,
            line: { color: COL.residual, width: 1 }, hovertemplate: '%{y:.1f} dB' });
  if (t.proposal) tr.push({ x: t.proposal.f, y: t.proposal.after, type: 'scatter', mode: 'lines',
                            name: 'residual after proposed mode', line: { color: COL.after, width: 2 },
                            hovertemplate: '%{y:.1f} dB' });
  if (shownTheory()) tr.push({ x: [null], y: [null], type: 'scatter', mode: 'lines', name: 'analytical plate',
                               line: { color: COL.theory, dash: 'dash', width: 1 }, hoverinfo: 'skip' });
  const { shapes, annotations } = topShapes();
  const lay = layout(topTitle(), 'Frequency (Hz)', 'rms |H| over points (dB)', {
    xaxis: { type: 'log', hoverformat: '.2f' }, hovermode: 'x unified', dragmode: 'zoom',
    uirevision: 'top-' + st.info.name, shapes, annotations, margin: { l: 52, r: 10, t: 40, b: 36 },
  });
  Plotly.react('plot-top', tr, lay, PCFG);
  wireTopClick();
}

function topRelayoutTheory() {
  if (!st.top) return;
  const { shapes, annotations } = topShapes();
  Plotly.relayout('plot-top', { shapes, annotations });
}

function wireTopClick() {
  const gd = $('plot-top');
  if (gd._wired) return;
  gd._wired = true;
  gd.on('plotly_click', (d) => {
    // frequency under the mouse (Plotly's hover point can be stale, e.g. with shift held)
    const xa = gd._fullLayout.xaxis, ev = d.event;
    let x = d.points && d.points.length ? d.points[0].x : null;
    if (ev && xa && xa.p2l) {
      const px = ev.clientX - gd.getBoundingClientRect().left - xa._offset;
      if (px >= 0 && px <= xa._length) x = 10 ** xa.p2l(px);
    }
    if (x == null) return;
    proposeAt(x, !(ev && ev.shiftKey));
  });
}

// ─── Modes sidebar ─────────────────────────────────────────────────────────
function renderModes() {
  const box = $('mode-list'), t = st.top;
  if (!t || !t.modes.length) { box.innerHTML = '<div class="list-empty">None yet</div>'; return; }
  const sel = st.view && st.view.which === 'mode' ? st.view.n : null;
  box.innerHTML = t.modes.map((m, i) => `
    <div class="mode-item${m.n === sel ? ' selected' : ''}${i >= t.n ? ' later' : ''}" data-n="${m.n}" title="${esc(m.title)}">
      <span class="n">${m.n}</span><span class="fr">${m.fr.toFixed(2)} Hz</span>
      <span class="q">Q ${m.Q.toFixed(0)}</span>
      <span></span><span class="q">η ${m.eta.toFixed(5)}</span><span class="q">${m.gain.toFixed(1)} dB</span>
    </div>`).join('');
  box.querySelectorAll('.mode-item').forEach(el => el.onclick = () => browseTo(+el.dataset.n));
  const s = box.querySelector('.selected');
  if (s) s.scrollIntoView({ block: 'nearest' });
}

// ─── Bottom panels ─────────────────────────────────────────────────────────
function clearFit(msg) {
  st.fit = null;
  for (const id of ['plot-nyq', 'plot-est', 'plot-zoom', 'plot-shape']) Plotly.purge(id);
  const lay = layout('', '', '', { xaxis: { visible: false }, yaxis: { visible: false },
    annotations: [{ text: msg || 'Click a peak in the top plot (or press n) to fit a mode.', showarrow: false,
                    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, font: { size: 11, color: '#5a5f6a' } }] });
  Plotly.react('plot-zoom', [], lay, { ...PCFG, displayModeBar: false });
  $('shape-caption').textContent = '';
  $('point-label').textContent = '';
}

const refreshFit = coalesce(async () => {
  if (!st.info || !st.view) return;
  const v = st.view;
  const r = await call('fit_view', v.which, v.which === 'mode' ? v.n : null, st.point == null ? -1 : st.point);
  if (!r) { clearFit(); return; }
  st.fit = r;
  renderNyquist(r); renderEstimates(r); renderZoom(r); renderShape(r);
  wireBottomClicks();
  const sl = $('point-slider');
  sl.value = r.nyquist.point;
  $('point-label').textContent = r.nyquist.label;
});
const refreshPoints = coalesce(async () => {   // only what changes with the point
  if (!st.info || !st.view) return;
  const v = st.view;
  const r = await call('fit_view', v.which, v.which === 'mode' ? v.n : null, st.point == null ? -1 : st.point);
  if (!r) return;
  st.fit = r;
  renderNyquist(r); renderEstimates(r); renderShape(r);
  $('point-label').textContent = r.nyquist.label;
});

function uirev(r, extra = '') { return `${r.mode.uid}${extra}`; }

function renderNyquist(r) {
  const q = r.nyquist;
  const tag = st.point == null ? 'strongest' : (q.point === q.strongest ? 'your pick, strongest' : 'your pick');
  const tr = [
    { x: q.full[0], y: q.full[1], type: 'scatter', mode: 'lines', name: 'full circle',
      line: { color: COL.sum, width: 1, dash: 'dot' }, hoverinfo: 'skip' },
    { x: q.arc[0], y: q.arc[1], type: 'scatter', mode: 'lines', name: 'fit (band)',
      line: { color: COL.sum, width: 2.2 }, hoverinfo: 'skip' },
    { x: q.data[0], y: q.data[1], type: 'scatter', mode: 'lines+markers', name: 'data',
      line: { color: COL.data, width: 0.8 }, marker: { size: 5, color: COL.data }, customdata: q.f,
      hovertemplate: '%{customdata:.2f} Hz<extra></extra>' },
    { x: [q.star[0]], y: [q.star[1]], type: 'scatter', mode: 'markers', name: 'f_r',
      marker: { symbol: 'star', size: 13, color: COL.fr }, hovertemplate: `f_r ${r.mode.fr.toFixed(3)} Hz<extra></extra>` },
    { x: [q.offset[0]], y: [q.offset[1]], type: 'scatter', mode: 'markers', name: 'offset',
      marker: { symbol: 'cross-thin', size: 11, line: { width: 2, color: '#000' } }, hoverinfo: 'skip' },
    { x: [0], y: [0], type: 'scatter', mode: 'markers+text', name: 'origin (0,0)', text: ['(0,0)'],
      textposition: 'top right', textfont: { size: 9 },
      marker: { symbol: 'circle-open', size: 9, color: '#000', line: { width: 1.5 } }, hoverinfo: 'skip' },
  ];
  const lay = layout(`<b>Point ${q.label}</b> (${tag}, ${q.status})<br><span style="font-size:9px;color:#5a5f6a">${q.describe}</span>`,
    'Re H/(−ω²)', 'Im H/(−ω²)', {
      xaxis: { exponentformat: 'power', zeroline: true, zerolinecolor: '#9aa0aa' },
      yaxis: { exponentformat: 'power', zeroline: true, zerolinecolor: '#9aa0aa', scaleanchor: 'x', scaleratio: 1 },
      legend: { font: { size: 8 }, orientation: 'h', x: 0, xanchor: 'left', y: -0.16, yanchor: 'top' },
      uirevision: uirev(r, '-' + q.point), margin: { l: 52, r: 8, t: 44, b: 62 },
    });
  Plotly.react('plot-nyq', tr, lay, PCFG);
}

function renderEstimates(r) {
  const e = r.estimates, sel = r.nyquist.point;
  const used = e.status.map((s, i) => (s === 0 ? Math.abs(e.dfr[i]) : 0));
  const lim = Math.max(1.4 * Math.max(0, ...used), 2 * e.df);
  const groups = [
    { s: 0, name: 'used', marker: (i) => ({ symbol: 'circle', size: 4 + 16 * e.R[i] ** 2, color: COL.data }) },
    { s: 1, name: 'small circle', marker: () => ({ symbol: 'circle-open', size: 7, color: '#8a8f99' }) },
    { s: 2, name: 'poor circle', marker: () => ({ symbol: 'x-thin', size: 8, color: COL.poor, line: { width: 1.5, color: COL.poor } }) },
    { s: 3, name: 'no circle', marker: () => ({ symbol: 'line-ns', size: 8, color: '#555', line: { width: 1.5, color: '#555' } }) },
  ];
  const tr = [];
  for (const g of groups) {
    const idx = e.status.map((s, i) => (s === g.s ? i : -1)).filter(i => i >= 0);
    if (!idx.length) continue;
    const ys = idx.map(i => (g.s === 3 || e.dfr[i] == null ? 0 : Math.max(-lim, Math.min(lim, e.dfr[i]))));
    const ms = idx.map(i => g.marker(i));
    tr.push({
      x: idx.map(i => e.labels[i]), y: ys, type: 'scatter', mode: 'markers', name: `${g.name} (${idx.length})`,
      customdata: idx, text: idx.map(i => (e.dfr[i] == null ? 'no circle' : `${e.dfr[i] >= 0 ? '+' : ''}${e.dfr[i].toFixed(3)} Hz`)),
      hovertemplate: 'point %{x}: %{text}<extra></extra>',
      marker: { symbol: ms.map(m => m.symbol), size: ms.map(m => m.size), color: ms[0].color,
                line: ms[0].line || { width: 1, color: ms[0].color } },
    });
  }
  tr.push({ x: [e.labels[sel]], y: [Math.max(-lim, Math.min(lim, e.dfr[sel] ?? 0))], type: 'scatter', mode: 'markers',
            name: 'selected', marker: { symbol: 'square-open', size: 13, color: '#000', line: { width: 2 } }, hoverinfo: 'skip' });
  tr.push({ x: [null], y: [null], type: 'scatter', mode: 'lines', name: `final ${e.fr.toFixed(2)}`,
            line: { color: COL.sum, width: 1.5 } });
  tr.push({ x: [null], y: [null], type: 'scatter', mode: 'lines', name: `circle-only ${e.fr_circle.toFixed(2)}`,
            line: { color: COL.sum, width: 1, dash: 'dash' } });
  const nUsed = e.status.filter(s => s === 0).length;
  const lay = layout(`<b>Each point's own f_r</b>: ${nUsed} of ${e.labels.length} used`,
    'Point (marker size = weight; hollow/× = excluded)', 'estimate − final f_r (Hz)', {
      yaxis: { range: [-lim * 1.05, lim * 1.05], zeroline: false },
      shapes: [
        { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: COL.sum, width: 1.5 } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: e.circle_only, y1: e.circle_only, line: { color: COL.sum, width: 1, dash: 'dash' } },
      ],
      legend: { font: { size: 8 }, orientation: 'h', x: 0, xanchor: 'left', y: -0.16, yanchor: 'top' },
      margin: { l: 52, r: 10, t: 34, b: 66 }, uirevision: uirev(r), hovermode: 'closest',
    });
  Plotly.react('plot-est', tr, lay, PCFG);
}

function renderZoom(r) {
  const z = r.zoom, m = r.mode, isProp = r.which === 'proposal';
  const tr = [
    { x: z.f, y: z.measured, type: 'scatter', mode: 'lines', name: 'measured', line: { color: COL.measured, width: 1 } },
    { x: z.f, y: z.before, type: 'scatter', mode: 'lines+markers', name: 'residual before (bins)',
      line: { color: COL.residual, width: 1.3 }, marker: { size: 3 } },
    { x: z.smooth_f, y: z.smooth, type: 'scatter', mode: 'lines', name: 'this mode (smooth)', line: { color: COL.sum, width: 1.3 } },
    { x: z.f, y: z.after, type: 'scatter', mode: 'lines', name: 'residual after', line: { color: COL.after, width: 1.6 } },
    { x: [z.fr], y: [z.fr_db], type: 'scatter', mode: 'markers', name: 'f_r', showlegend: false,
      marker: { symbol: 'star', size: 12, color: COL.fr }, hovertemplate: `f_r ${z.fr.toFixed(3)} Hz<extra></extra>` },
  ];
  const shapes = [
    { type: 'rect', xref: 'x', yref: 'paper', x0: z.band[0], x1: z.band[1], y0: 0, y1: 1, fillcolor: COL.sum,
      opacity: 0.1, line: { width: 0 }, layer: 'below' },
    vline(z.fr, COL.fr, 'dash', 1),
  ];
  const ann = [];
  if (st.theoryOn) for (const [i, f] of z.theory) {
    shapes.push(vline(f, COL.theory, 'dash', 1, 0.7));
    ann.push({ x: f, y: 0.02, xref: 'x', yref: 'paper', text: `analytical #${i}: ${f.toFixed(1)} Hz`, textangle: -90,
               showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 8, color: COL.theory } });
  }
  const col = isProp ? COL.fr : COL.data;
  const lay = layout(`<b style="color:${col}">${isProp ? 'PROPOSED' : 'MODE ' + m.n}</b>: ${m.fr.toFixed(3)} Hz · η ${m.eta.toFixed(5)} · ${m.bins} bins · ${m.gain.toFixed(1)} dB better`,
    'Frequency (Hz)', 'rms |H| (dB)', { shapes, annotations: ann, uirevision: uirev(r), hovermode: 'x unified',
      legend: { font: { size: 8 }, bgcolor: 'rgba(255,255,255,0.8)', x: 0.99, xanchor: 'right', y: 0.99 } });
  Plotly.react('plot-zoom', tr, lay, PCFG);
}

function renderShape(r) {
  const s = r.shape, sel = s.selected, tr = [];
  const names = { strongest: 'strongest', poor_fit: 'poor circle fit', phase: 'not cleanly in/out of phase' };
  const colors = { strongest: COL.strongest, poor_fit: COL.poor, phase: COL.phase };
  const sizes = { strongest: 15, poor_fit: 20, phase: 25 };
  let lay;
  if (s.x) {
    if (s.grid) {
      tr.push({ type: 'heatmap', x: s.grid.x, y: s.grid.y, z: s.grid.z, zmin: -1, zmax: 1, zsmooth: 'best',
                colorscale: 'RdBu', reversescale: true, showscale: false, hoverinfo: 'skip' });
      tr.push({ type: 'contour', x: s.grid.x, y: s.grid.y, z: s.grid.z, showscale: false, hoverinfo: 'skip',
                contours: { start: 0, end: 0, size: 1, coloring: 'none' }, line: { color: '#000', width: 1.5 } });
    }
    tr.push({ x: s.x, y: s.y, type: 'scatter', mode: 'markers', customdata: s.labels.map((_, i) => i),
              text: s.labels.map((l, i) => `point ${l}: ${s.values[i].toFixed(2)}`),
              hovertemplate: '%{text}<extra></extra>',
              marker: s.grid ? { size: 4, color: '#000' }
                             : { size: 12, color: s.values, colorscale: 'RdBu', reversescale: true, cmin: -1, cmax: 1, line: { width: 0.5, color: '#000' } } });
    for (const k of ['strongest', 'poor_fit', 'phase']) {
      const idx = s.picks[k];
      if (!idx.length) continue;
      tr.push({ x: idx.map(i => s.x[i]), y: idx.map(i => s.y[i]), type: 'scatter', mode: 'markers+text',
                text: idx.map(i => s.labels[i]), textposition: 'top right',
                textfont: { size: 9, color: colors[k] }, hoverinfo: 'skip',
                marker: { symbol: 'circle-open', size: sizes[k], color: colors[k], line: { width: 2 } } });
    }
    tr.push({ x: [s.x[sel]], y: [s.y[sel]], type: 'scatter', mode: 'markers', hoverinfo: 'skip',
              marker: { symbol: 'square-open', size: 14, color: '#000', line: { width: 2 } } });
    lay = layout('<b>Mode shape</b> (real part) · □ selected point', 'x (mm)', 'y (mm)', {
      yaxis: { scaleanchor: 'x', scaleratio: 1, zeroline: false }, xaxis: { zeroline: false },
      showlegend: false, uirevision: uirev(r), hovermode: 'closest' });
  } else {
    tr.push({ x: s.labels, y: s.values, type: 'bar', marker: { color: s.values.map(v => (v >= 0 ? '#b2182b' : '#2166ac')) },
              customdata: s.labels.map((_, i) => i), hovertemplate: 'point %{x}: %{y:.2f}<extra></extra>' });
    lay = layout('<b>Mode shape</b> (real part)', 'Point', 'shape', { showlegend: false, uirevision: uirev(r) });
  }
  Plotly.react('plot-shape', tr, lay, PCFG);
  const lab = (k) => (s.picks[k].length ? s.picks[k].map(i => s.labels[i]).join(', ') : 'none');
  $('shape-caption').innerHTML = ['strongest', 'poor_fit', 'phase']
    .map(k => `<b style="color:${colors[k]}">○ ${names[k]}</b>: ${lab(k)}`).join(' · ');
}

function wireBottomClicks() {
  const est = $('plot-est'), shp = $('plot-shape');
  if (!est._wired) {
    est._wired = true;
    est.on('plotly_click', (d) => {
      const p = d.points && d.points.find(q => q.customdata != null);
      if (p) selectPoint(p.customdata);
    });
  }
  if (!shp._wired) {
    shp._wired = true;
    shp.on('plotly_click', (d) => {
      if (!st.fit || !d.points || !d.points.length) return;
      const s = st.fit.shape, q = d.points[0];
      if (!s.x) { if (q.customdata != null) selectPoint(q.customdata); return; }
      let best = 0, bd = Infinity;
      s.x.forEach((x, i) => { const dd = (x - q.x) ** 2 + (s.y[i] - q.y) ** 2; if (dd < bd) { bd = dd; best = i; } });
      selectPoint(best);
    });
  }
}

function selectPoint(i) {
  st.point = i;
  $('point-slider').value = i;
  refreshPoints();
}
$('point-slider').addEventListener('input', (e) => selectPoint(+e.target.value));
$('point-strongest').addEventListener('click', () => { st.point = null; refreshPoints(); });

// ─── Actions ───────────────────────────────────────────────────────────────
async function afterPropose(r) {
  if (!r.ok) { status(r.msg, 'err'); return; }
  st.view = { which: 'proposal' };
  status('Proposed: ' + r.mode.title);
  await refreshTop();
  await refreshFit();
}
async function proposeAt(f, snap) {
  if (!st.info) return;
  st.step = null;
  try { await afterPropose(await call('propose', f, -1, snap)); } catch (err) { status(err.message, 'err'); }
}
async function browseTo(n) {
  if (!st.top) return;
  const N = st.top.total;
  n = Math.max(0, Math.min(N, n));
  if (st.top.proposal) await call('discard');
  st.step = n === N ? null : n;
  st.view = n ? { which: 'mode', n } : null;
  await refreshTop();
  if (n) await refreshFit(); else clearFit('Step 0: the measured data with no modes removed.');
}
function shown() { return st.top ? (st.step == null ? st.top.total : st.step) : 0; }

async function act(a) {
  if (a === 'help') return openModal('help-modal');
  if (a === 'options') return openModal('options-modal');
  if (a === 'measured') {
    st.showMeasured = !st.showMeasured;
    const b = $('btn-measured');
    b.classList.toggle('active', !st.showMeasured);
    b.innerHTML = (st.showMeasured ? 'Hide measured' : 'Show measured') + ' <kbd>m</kbd>';
    if (st.top) renderTop();
    return;
  }
  if (a === 'tune') {
    $('tune').classList.toggle('hidden');
    syncTuneButton(); savePrefs();
    window.dispatchEvent(new Event('resize'));
    return;
  }
  if (!st.info) { status('Open a test folder first.', 'err'); return; }
  try {
    switch (a) {
      case 'next': {
        st.step = null;
        const f = await call('next_peak');
        if (f == null) status('No further peaks above the last mode.');
        else await proposeAt(f, true);
        break;
      }
      case 'accept': {
        const r = await call('accept');
        if (!r.ok) { status(r.msg, 'err'); return; }
        st.view = { which: 'mode', n: r.mode.n };
        status(`Accepted mode ${r.mode.n}: ${r.mode.fr.toFixed(3)} Hz, η ${r.mode.eta.toFixed(5)}. Click the next peak or press n.`, 'ok');
        await refreshTop(); saveReplay();
        await refreshFit(); await refreshPairs();
        break;
      }
      case 'discard':
        await call('discard');
        st.view = null; clearFit('Proposal discarded. Click a peak or press n.');
        await refreshTop();
        break;
      case 'band-': case 'band+':
        await afterPropose(await call('band', a === 'band+' ? 1 : -1)); break;
      case 'left': case 'right':
        await afterPropose(await call('nudge', a === 'right' ? 1 : -1)); break;
      case 'undo': {
        const r = await call('undo');
        st.step = null;
        if (r.ok) status(`Undid mode ${r.mode.n} (${r.mode.fr.toFixed(2)} Hz).`);
        await refreshTop(); saveReplay();
        st.view = st.top.total ? { which: 'mode', n: st.top.total } : null;
        if (st.view) await refreshFit(); else clearFit();
        await refreshPairs();
        break;
      }
      case 'prev': await browseTo(shown() - 1); break;
      case 'nextstep': await browseTo(shown() + 1); break;
      case 'latest': await browseTo(st.top.total); break;
      case 'export': await openExport(); break;
    }
  } catch (err) { status(err.message, 'err'); }
}
document.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => act(b.dataset.act)));

document.addEventListener('keydown', (e) => {
  const tgt = e.target;
  const typing = tgt.tagName === 'TEXTAREA' || tgt.tagName === 'SELECT' ||
    (tgt.tagName === 'INPUT' && ['text', 'number', 'range'].includes(tgt.type));
  if (typing) {                 // let text boxes and sliders have their keys
    if (e.key === 'Escape') tgt.blur();
    return;
  }
  if (document.querySelector('.modal-overlay.active')) { if (e.key === 'Escape') closeModals(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const map = { n: 'next', a: 'accept', Enter: 'accept', x: 'discard', Escape: 'discard', '[': 'band-', ']': 'band+',
                ',': 'left', '.': 'right', u: 'undo', m: 'measured', ArrowLeft: 'prev', ArrowRight: 'nextstep' };
  if (e.key === 'Enter' && e.target.tagName === 'BUTTON') return;
  if (map[e.key]) { e.preventDefault(); act(map[e.key]); return; }
  if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && st.fit) {
    e.preventDefault();
    const n = st.info.n_points, cur = st.fit.nyquist.point;
    selectPoint(Math.max(0, Math.min(n - 1, cur + (e.key === 'ArrowUp' ? 1 : -1))));
  }
});

// ─── Modals ────────────────────────────────────────────────────────────────
function openModal(id) { $(id).classList.add('active'); }
function closeModals() { document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); }
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', closeModals));
document.querySelectorAll('.modal-overlay').forEach(o => o.addEventListener('click', (e) => { if (e.target === o) closeModals(); }));

$('opt-apply').addEventListener('click', async () => {
  const fmin = parseFloat($('opt-fmin').value), fmax = parseFloat($('opt-fmax').value);
  if (!(fmin > 0 && fmax > fmin)) { alert('Enter a valid frequency range.'); return; }
  const pad = Math.max(0, parseFloat($('opt-pad').value) || 0);
  const binsChange = pad !== st.opts.pad;
  st.opts = { fmin, fmax, flip: $('opt-flip').checked, refine: $('opt-refine').checked, pad };
  savePrefs(); closeModals();
  if (st.info) {
    // fit bands are stored in bins; when the bin spacing changes, let each band re-size itself
    const replay = st.top ? st.top.modes.map(m => (binsChange ? m.peak_hz.toFixed(4)
                                                               : `${m.peak_hz.toFixed(4)}:${m.side}`)).join(',') : '';
    try { await loadSession(replay || null); } catch (err) { status(err.message, 'err'); }
  }
});

async function openExport() {
  const ex = await call('exports');
  st.exports = ex;
  $('replay-text').value = ex.replay;
  $('export-msg').textContent = '';
  openModal('export-modal');
}
function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function flash(id, msg) { const el = $(id); el.textContent = msg; setTimeout(() => { el.textContent = ''; }, 2500); }
$('dl-modes').addEventListener('click', () => download(`${st.exports.name}_modes.csv`, st.exports.modes_csv));
$('dl-shapes').addEventListener('click', () => download(`${st.exports.name}_mode_shapes.csv`, st.exports.shapes_csv));
$('replay-copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('replay-text').value); flash('export-msg', 'Copied'); }
  catch (e) { $('replay-text').select(); flash('export-msg', 'Press ⌘C to copy'); }
});
$('replay-run').addEventListener('click', async () => {
  const txt = $('replay-text').value.trim();
  if (!txt) return;
  closeModals();
  const r = await call('replay', txt);
  status(`Replayed ${r.done} modes` + (r.failed.length ? `; could not fit ${r.failed.join(', ')} Hz` : '') + '.', 'ok');
  await refreshTop(); saveReplay();
  st.view = st.top.total ? { which: 'mode', n: st.top.total } : null;
  if (st.view) await refreshFit();
  await refreshPairs();
});

// ─── Analytical plate panel ────────────────────────────────────────────────
function syncTuneButton() {
  const open = !$('tune').classList.contains('hidden');
  const b = document.querySelector('[data-act="tune"]');
  b.textContent = open ? 'Analytical ◂' : 'Analytical ▸';
  b.classList.toggle('active', open);
}

function fmtSlider(key, v) {
  if (key === 'k') return v >= 5.99 ? 'rigid' : (10 ** v).toPrecision(3).replace(/\.?0+e/, 'e').replace(/\.0+$/, '');
  if (key === 'nu') return v.toFixed(3);
  if (key === 't') return v.toFixed(3);
  return (Math.round(v * 100) / 100).toString();
}
function parseSlider(key, text) {
  const t = String(text).trim().toLowerCase();
  if (key === 'k') {
    if (t === 'rigid' || t === '') return 6;
    const k = parseFloat(t);
    return k > 0 ? Math.min(6, Math.max(0, Math.log10(k))) : null;
  }
  const v = parseFloat(t);
  return isFinite(v) ? v : null;
}

function buildSliders() {
  const box = $('sliders');
  box.innerHTML = '';
  for (const sp of st.specs) {
    const row = document.createElement('div');
    row.className = 'slider';
    const step = (sp.hi - sp.lo) / 1000;
    row.innerHTML = `
      <input type="checkbox" title="vary when optimizing" ${st.free.has(sp.key) ? 'checked' : ''}>
      <span class="lab">${esc(sp.label)}</span>
      <input type="range" min="${sp.lo}" max="${sp.hi}" step="${step}" value="${st.sliders[sp.key]}">
      <input type="text" class="tb-inp val" value="${fmtSlider(sp.key, st.sliders[sp.key])}">`;
    const [chk, rng, txt] = row.querySelectorAll('input');
    chk.addEventListener('change', () => { chk.checked ? st.free.add(sp.key) : st.free.delete(sp.key); savePrefs(); });
    rng.addEventListener('input', () => { st.sliders[sp.key] = +rng.value; txt.value = fmtSlider(sp.key, +rng.value); previewTheory(); });
    rng.addEventListener('change', () => applyTheory());
    txt.addEventListener('change', () => {
      const v = parseSlider(sp.key, txt.value);
      if (v == null) { txt.value = fmtSlider(sp.key, st.sliders[sp.key]); return; }
      st.sliders[sp.key] = Math.min(sp.hi, Math.max(sp.lo, v));
      rng.value = st.sliders[sp.key]; txt.value = fmtSlider(sp.key, st.sliders[sp.key]);
      applyTheory();
    });
    row._set = (v) => { rng.value = v; txt.value = fmtSlider(sp.key, v); };
    row.dataset.key = sp.key;
    box.appendChild(row);
  }
}
function setSliderValues(vals) {
  Object.assign(st.sliders, vals);
  $('sliders').querySelectorAll('.slider').forEach(r => r._set(st.sliders[r.dataset.key]));
}

async function plateFromSliders() {
  const r = await call('theory_from_sliders', JSON.stringify(st.sliders), true);
  st.plate = r.plate; st.plateCli = r.cli;
  return r.plate;
}

/* Live preview while dragging: light solve, only the lines and the table. */
const previewTheory = coalesce(async () => {
  if (!st.info) { await plateFromSliders(); return; }
  const r = await call('theory_from_sliders', JSON.stringify(st.sliders), true);
  st.plate = r.plate; st.plateCli = r.cli;
  st.theoryPreview = st.theoryOn ? r.theory : null;
  topRelayoutTheory();
  await refreshPairs();
});

/* On release: full solve, used everywhere (top plot, zoom panel, table). */
const applyTheory = coalesce(async () => {
  savePrefs();
  if (!st.info) { await plateFromSliders(); return; }
  const plate = await plateFromSliders();
  const r = await call('set_theory', st.theoryOn ? JSON.stringify(plate) : null);
  st.theory = r.theory; st.theoryPreview = null;
  topRelayoutTheory();
  if (st.top) renderTop();
  if (st.view) await refreshFit();
  await refreshPairs();
});

$('theory-on').addEventListener('change', (e) => { st.theoryOn = e.target.checked; applyTheory(); });

const refreshPairs = coalesce(async () => {
  const body = $('pairs-body'), score = $('tune-score');
  const th = shownTheory();
  if (!st.info || !th) { body.innerHTML = ''; score.textContent = st.info ? 'Analytical lines are off.' : ''; return; }
  const [lo, hi] = st.band || [null, null];
  const r = await call('match', JSON.stringify(th), lo, hi);
  if (!r.pairs.length) {
    body.innerHTML = '';
    score.textContent = r.n_all ? 'No accepted modes in the band.' : 'Accept some modes to compare with the analytical plate.';
    return;
  }
  const band = st.band ? ` in ${st.band[0].toFixed(0)}–${st.band[1].toFixed(0)} Hz` : '';
  score.textContent = `${r.pairs.length} peaks${band}: mean ${r.mean >= 0 ? '+' : ''}${r.mean.toFixed(1)}%, rms ${r.rms.toFixed(1)}%`;
  body.innerHTML = r.pairs.map(p => `<tr><td>${p.peak.toFixed(1)}</td><td>${p.idx}</td><td>${p.analytical.toFixed(1)}</td>
    <td class="${Math.abs(p.diff) > 5 ? 'bad' : ''}">${p.diff >= 0 ? '+' : ''}${p.diff.toFixed(1)}%</td></tr>`).join('');
});

function setBand(b) {
  st.band = b;
  $('band-lo').value = b ? b[0].toFixed(1) : '';
  $('band-hi').value = b ? b[1].toFixed(1) : '';
  topRelayoutTheory();
  refreshPairs();
}
function bandFromInputs() {
  const lo = parseFloat($('band-lo').value), hi = parseFloat($('band-hi').value);
  if (isFinite(lo) && isFinite(hi) && hi > lo) setBand([lo, hi]);
  else if (!$('band-lo').value && !$('band-hi').value) setBand(null);
}
$('band-lo').addEventListener('change', bandFromInputs);
$('band-hi').addEventListener('change', bandFromInputs);
$('band-all').addEventListener('click', () => setBand(null));
$('band-view').addEventListener('click', () => {
  const gd = $('plot-top');
  if (!gd.layout || !gd.layout.xaxis || !gd.layout.xaxis.range) return;
  const [a, b] = gd.layout.xaxis.range;
  setBand([10 ** a, 10 ** b]);
});

$('tune-opt').addEventListener('click', async () => {
  if (!st.info) { tuneMsg('Open a test and accept some modes first.', true); return; }
  if (!st.free.size) { tuneMsg('Tick at least one parameter to vary.', true); return; }
  tuneMsg('Optimizing…');
  const plate = await plateFromSliders();
  const [lo, hi] = st.band || [null, null];
  const r = await call('optimize', JSON.stringify(plate), JSON.stringify([...st.free]), lo, hi);
  if (!r.ok) { tuneMsg(r.msg, true); return; }
  const vals = await call('plate_to_sliders', JSON.stringify(r.plate));
  setSliderValues(vals);
  await applyTheory();
  let msg = `rms ${r.before.toFixed(1)}% → ${r.after.toFixed(1)}% over ${r.n_peaks} peaks`;
  const tbr = ['t', 'E', 'rho'].filter(k => st.free.has(k));
  if (tbr.length > 1) msg += '\nThickness, E and density only matter together, so their split is arbitrary.';
  if (r.at_limit.length) msg += `\nAt the slider limit: ${r.at_limit.join(', ')} (the best value may lie beyond it).`;
  tuneMsg(msg);
});
function tuneMsg(msg, err) { const el = $('tune-msg'); el.textContent = msg; el.className = 'tune-msg' + (err ? ' err' : ''); }
$('tune-reset').addEventListener('click', () => {
  const vals = {};
  for (const sp of st.specs) vals[sp.key] = sp.value;
  setSliderValues(vals); tuneMsg(''); applyTheory();
});
$('tune-copy').addEventListener('click', async () => {
  await plateFromSliders();
  try { await navigator.clipboard.writeText(st.plateCli); tuneMsg('Copied: ' + st.plateCli); }
  catch (e) { tuneMsg(st.plateCli); }
});

// Python takes a while on first load; say so if it is very slow
setTimeout(() => {
  if (!st.ready) document.querySelector('#loading .load-sub').textContent =
    'Still loading… this needs an internet connection the first time.';
}, 45000);
