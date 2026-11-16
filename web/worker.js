/* worker.js: runs Python (Pyodide + NumPy + SciPy) off the main thread so the
 * page stays responsive while fits and optimizations run. It loads the same
 * modal_core.py the desktop script uses, plus web/api.py, then answers
 * {id, fn, args} messages with the JSON string the Python function returns. */
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(PYODIDE + 'pyodide.js');

const VERSION = new URL(self.location).searchParams.get('v') || '';
let py = null, api = null;

const ready = (async () => {
  postMessage({ type: 'progress', msg: 'Loading Python…' });
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: 'progress', msg: 'Loading NumPy + SciPy…' });
  await py.loadPackage(['numpy', 'scipy']);
  postMessage({ type: 'progress', msg: 'Loading the circle-fit code…' });
  const root = new URL('../', self.location).href;
  for (const [src, dest] of [['modal_core.py', 'modal_core.py'], ['web/api.py', 'api.py']]) {
    const r = await fetch(root + src + '?v=' + VERSION, { cache: 'no-cache' });
    if (!r.ok) throw new Error('Could not fetch ' + src + ' (' + r.status + ')');
    py.FS.writeFile('/home/pyodide/' + dest, await r.text());
  }
  py.runPython('import sys; sys.path.insert(0, "/home/pyodide")');
  api = py.pyimport('api');
  postMessage({ type: 'ready' });
})().catch(err => postMessage({ type: 'fatal', error: String(err && err.message || err) }));

function writeFolder(name, files) {
  // Replace /data with this test folder: /data/<name>/TRF/*.trf (+ template.json)
  py.runPython('import shutil, os; shutil.rmtree("/data", ignore_errors=True)');
  py.FS.mkdirTree('/data/' + name + '/TRF');
  for (const f of files) py.FS.writeFile('/data/' + name + '/' + f.path, new Uint8Array(f.buffer));
}

onmessage = async (e) => {
  const { id, fn, args = [], files, name } = e.data;
  try {
    await ready;
    if (!api) throw new Error('Python did not load');
    let result = null;
    if (fn === '__write') writeFolder(name, files);
    else {
      const f = api[fn];
      if (!f) throw new Error('Unknown function ' + fn);
      result = f(...args);
    }
    postMessage({ id, ok: true, result });
  } catch (err) {
    // keep the last line of a Python traceback: the actual error
    const msg = String(err && err.message || err).trim().split('\n').filter(Boolean).pop();
    postMessage({ id, ok: false, error: msg });
  }
};
