import { loadModel as onnxLoad, inferFrame, setClassNames, clearSession } from './inference.js';

// ════════════════════════════════════════
// STATE
// ════════════════════════════════════════
const S = {
  session: null, loaded: false, fmt: null,
  inputType: 'image', task: 'detect',
  mediaEl: null, stream: null,
  running: false, loopId: null,
  frame: 0, runs: 0, lats: [],
  colors: {}, classes: [],
  imgW: 0, imgH: 0, zoom: 1,
  vfCount: 0, vfTotal: 0, vfPeak: 0,
};

const PAL = [
  '#ffb400','#00e676','#4fc3f7','#ff6b6b','#ce93d8','#80cbc4',
  '#ffcc02','#ef9a9a','#80deea','#a5d6a7','#ffab91','#90caf9',
  '#f48fb1','#ffe082','#bcaaa4','#c5e1a5','#b39ddb','#ff8a65',
];

// ════════════════════════════════════════
// BOOT
// ════════════════════════════════════════
window.addEventListener('load', () => {
  drawPerf([]);
  toast('Upload your YOLO26 .onnx model to begin', 'inf');
  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) processMedia(f);
  });
});

// ════════════════════════════════════════
// MODEL LOAD
// ════════════════════════════════════════
async function loadModel(e) {
  const file = e.target.files[0];
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  S.fmt = ext;

  const card = document.getElementById('modelCard');
  card.classList.add('show');
  document.getElementById('mcFile').textContent =
    file.name.length > 20 ? file.name.slice(0, 18) + '…' : file.name;
  document.getElementById('mcFmt').textContent = ext.toUpperCase();
  document.getElementById('mcSz').textContent  = (file.size / 1024 / 1024).toFixed(1) + ' MB';
  setStat('busy', 'LOADING…');
  setMcStat('Loading…', 'warn');

  if (ext === 'pt' || ext === 'torchscript') {
    document.getElementById('ptModal').classList.add('on');
    S.loaded = true;
    clearSession();
    setStat('ready', 'DEMO MODE');
    setMcStat('Demo (simulation)', 'warn');
    document.getElementById('mcBack').textContent = 'Demo / simulation';
    document.getElementById('iBack').textContent  = 'Demo';
    document.getElementById('iMdl').textContent   = file.name.slice(0, 14);
    updSteps();
    return;
  }

  if (ext !== 'onnx') {
    toast('Use .onnx or .pt format', 'err');
    setStat('error', 'FORMAT ERROR');
    return;
  }

  const pw = document.getElementById('progWrap');
  const pb = document.getElementById('progBar');
  pw.classList.add('show');
  let prog = 0;
  const iv = setInterval(() => {
    prog = Math.min(prog + Math.random() * 12, 88);
    pb.style.width = prog + '%';
  }, 120);

  try {
    S.session = await onnxLoad(file);
    clearInterval(iv);
    pb.style.width = '100%';
    setTimeout(() => pw.classList.remove('show'), 700);

    S.loaded = true;
    setMcStat('✓ Ready', 'ok');
    document.getElementById('mcBack').textContent = 'YOLO26 / ONNX (WASM)';
    document.getElementById('iBack').textContent  = 'ONNX/WASM';
    document.getElementById('iMdl').textContent   = file.name.slice(0, 14);
    setStat('ready', 'MODEL READY');
    updSteps();
    toast(`✓ ${file.name} loaded`, 'ok');
  } catch (err) {
    clearInterval(iv);
    console.error(err);
    setMcStat('✗ Failed', '');
    document.getElementById('mcStat').style.color = 'var(--red)';
    setStat('error', 'LOAD ERROR');
    toast('❌ ONNX load failed: ' + err.message, 'err');
  }
}

function setMcStat(txt, cls) {
  const el = document.getElementById('mcStat');
  el.textContent = txt;
  el.className   = 'mc-v' + (cls ? ' ' + cls : '');
}

// ════════════════════════════════════════
// INPUT TYPE
// ════════════════════════════════════════
function setInput(type) {
  S.inputType = type;
  ['image', 'video', 'camera', 'rtsp'].forEach(t =>
    document.getElementById('ib-' + t).classList.toggle('on', t === type)
  );
  document.getElementById('fileSection').style.display =
    ['image', 'video'].includes(type) ? 'block' : 'none';
  document.getElementById('rtspBox').classList.toggle('show', type === 'rtsp');
  const mf = document.getElementById('mediaFile');
  if (type === 'image') mf.accept = 'image/*';
  else if (type === 'video') mf.accept = 'video/*';
  if (type === 'camera') toast('📷 Click RUN INFERENCE to start webcam', 'inf');
  if (type === 'rtsp')   toast('📡 RTSP: browser uses webcam as fallback', 'inf');
  stopInference();
}

// ════════════════════════════════════════
// MEDIA
// ════════════════════════════════════════
function handleMedia(e) {
  const f = e.target.files[0];
  if (f) processMedia(f);
}

function processMedia(file) {
  stopInference();
  const isImg = file.type.startsWith('image/');
  const isVid = file.type.startsWith('video/');
  if (!isImg && !isVid) { toast('Unsupported media type', 'err'); return; }
  const url = URL.createObjectURL(file);
  if (isImg) { loadImg(url); document.getElementById('vidBar').classList.remove('on'); }
  else       { loadVid(url); document.getElementById('vidBar').classList.add('on'); }
  toast('Loaded: ' + file.name, 'ok');
}

function loadImg(url) {
  const img = new Image();
  img.onload = () => {
    S.mediaEl = img;
    S.imgW    = img.naturalWidth;
    S.imgH    = img.naturalHeight;
    showCanvas();
    drawMedia();
    updSteps();
  };
  img.src = url;
}

function loadVid(url) {
  const vid = document.createElement('video');
  vid.src = url; vid.muted = true; vid.crossOrigin = 'anonymous';
  vid.onloadedmetadata = () => {
    S.mediaEl = vid;
    S.imgW    = vid.videoWidth;
    S.imgH    = vid.videoHeight;
    document.getElementById('vsD').textContent = fmt(vid.duration);
    showCanvas();
    updSteps();
  };
  vid.ontimeupdate = () => {
    if (!vid.duration) return;
    document.getElementById('tlFill').style.width =
      (vid.currentTime / vid.duration * 100) + '%';
    document.getElementById('tsLbl').textContent =
      fmt(vid.currentTime) + ' / ' + fmt(vid.duration);
  };
}

// ════════════════════════════════════════
// CANVAS
// ════════════════════════════════════════
function showCanvas() {
  const wrap = document.querySelector('.canvas-wrap');
  const sc   = Math.min(
    (wrap.clientWidth  - 40) / S.imgW,
    (wrap.clientHeight - 40) / S.imgH,
    1
  );
  const ic = document.getElementById('inputCanvas');
  const oc = document.getElementById('outputCanvas');
  ic.width = S.imgW; ic.height = S.imgH;
  oc.width = S.imgW; oc.height = S.imgH;
  ic.style.width  = oc.style.width  = (S.imgW * sc) + 'px';
  ic.style.height = oc.style.height = (S.imgH * sc) + 'px';
  document.getElementById('placeholder').style.display  = 'none';
  document.getElementById('canvasBox').style.display    = 'block';
}

function drawMedia() {
  if (!S.mediaEl) return;
  const ic = document.getElementById('inputCanvas');
  ic.getContext('2d').drawImage(S.mediaEl, 0, 0, ic.width, ic.height);
}

function doZoom(f) {
  if (f === 0) { S.zoom = 1; document.getElementById('canvasBox').style.transform = ''; return; }
  S.zoom = Math.max(0.25, Math.min(S.zoom * f, 4));
  document.getElementById('canvasBox').style.transform = `scale(${S.zoom})`;
}

// ════════════════════════════════════════
// RUN / STOP
// ════════════════════════════════════════
async function runInference() {
  if (S.running) return;
  if (!S.loaded) { toast('Load a model first', 'err'); return; }
  S.task    = document.getElementById('taskType').value;
  S.classes = document.getElementById('classNames').value
    .split(',').map(s => s.trim()).filter(Boolean);
  setClassNames(S.classes);

  S.running = true;
  document.getElementById('runBtn').style.display  = 'none';
  document.getElementById('stopBtn').style.display = 'block';
  setStat('busy', 'RUNNING');

  if (S.inputType === 'image') {
    if (!S.mediaEl) { toast('Load an image first', 'err'); stopInference(); return; }
    await oneShot();
  } else if (S.inputType === 'video') {
    if (!S.mediaEl) { toast('Load a video first', 'err'); stopInference(); return; }
    await vidLoop();
  } else {
    await startCam();
  }
}

function stopInference() {
  S.running = false;
  if (S.loopId) clearTimeout(S.loopId);
  if (S.stream) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; }
  document.getElementById('camLive').classList.remove('on');
  document.getElementById('runBtn').style.display  = 'block';
  document.getElementById('stopBtn').style.display = 'none';
  showProc(false);
  setStat(S.loaded ? 'ready' : 'off', S.loaded ? 'MODEL READY' : 'NO MODEL');
}

async function inferCanvas() {
  const ic   = document.getElementById('inputCanvas');
  const sz   = parseInt(document.getElementById('imgsize').value);
  const conf = parseFloat(document.getElementById('conf').value);
  if (S.session) return await inferFrame(ic, sz, conf);
  return demo();
}

async function oneShot() {
  showProc(true, 'INFERRING');
  const t0  = performance.now();
  const res = await inferCanvas();
  const lat = performance.now() - t0;
  showProc(false);
  if (res) { draw(res, lat); S.runs++; updInfo(); }
  toast(`✓ Done · ${Math.round(lat)} ms`, 'ok');
  stopInference();
}

async function vidLoop() {
  const vid = S.mediaEl;
  vid.currentTime = 0;
  vid.play();
  S.vfCount = 0; S.vfTotal = 0; S.vfPeak = 0;
  const step = async () => {
    if (!S.running || vid.ended) { stopInference(); return; }
    drawMedia();
    const t0  = performance.now();
    const res = await inferCanvas();
    const lat = performance.now() - t0;
    if (res) {
      draw(res, lat);
      const n = res.predictions.length;
      S.vfTotal += n; S.vfPeak = Math.max(S.vfPeak, n); S.vfCount++;
      S.runs++;
      logEntry(vid.currentTime, n, res);
      document.getElementById('vsF').textContent = S.vfCount;
      document.getElementById('vsP').textContent = S.vfPeak;
      document.getElementById('vsA').textContent = (S.vfTotal / S.vfCount).toFixed(1);
      updInfo();
    }
    if (!vid.ended && S.running) S.loopId = setTimeout(step, 80);
  };
  S.loopId = setTimeout(step, 300);
}

async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    S.stream = stream;
    const vid = document.createElement('video');
    vid.srcObject = stream; vid.muted = true;
    await vid.play();
    S.mediaEl = vid; S.imgW = 640; S.imgH = 480;
    showCanvas();
    document.getElementById('camLive').classList.add('on');
    toast('🎥 Camera live', 'ok');
    const rate = parseInt(document.getElementById('rtspRate').value) || 100;
    const step = async () => {
      if (!S.running) return;
      drawMedia();
      const t0  = performance.now();
      const res = await inferCanvas();
      const lat = performance.now() - t0;
      if (res) {
        draw(res, lat);
        logEntry(Date.now() / 1000, res.predictions.length, res);
        S.runs++;
        updInfo();
      }
      if (S.running) S.loopId = setTimeout(step, rate);
    };
    S.loopId = setTimeout(step, 400);
  } catch (e) {
    toast('❌ Camera: ' + e.message, 'err');
    stopInference();
  }
}

// ════════════════════════════════════════
// DEMO MODE  (.pt uploads or no ONNX session)
// ════════════════════════════════════════
function demo() {
  const bases = {
    detect:   ['person', 'helmet', 'vest', 'car', 'forklift', 'pallet', 'box', 'bottle'],
    segment:  ['item_A', 'item_B', 'defect', 'surface'],
    classify: ['Grade_A', 'Grade_B', 'Grade_C', 'Reject'],
    pose:     ['worker'],
    obb:      ['package', 'crate', 'bin', 'tray'],
  };
  const pool = [...(bases[S.task] || ['object']), ...S.classes];
  const ic   = document.getElementById('inputCanvas');
  const W    = ic.width || 640, H = ic.height || 480;
  const ct   = parseFloat(document.getElementById('conf').value);
  const n    = Math.floor(Math.random() * 7) + 1;

  const preds = Array.from({ length: n }, () => {
    const name = pool[Math.floor(Math.random() * pool.length)];
    const x  = Math.random() * W * 0.7 + W * 0.08;
    const y  = Math.random() * H * 0.7 + H * 0.08;
    const bw = Math.random() * W * 0.18 + W * 0.07;
    const bh = Math.random() * H * 0.18 + H * 0.07;
    const conf = ct + Math.random() * (1 - ct) * 0.97;
    const p = { name, confidence: conf, box: { x1: x-bw/2, y1: y-bh/2, x2: x+bw/2, y2: y+bh/2 } };
    if (S.task === 'segment') {
      p.segments = { x: [], y: [] };
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        p.segments.x.push((x + Math.cos(a) * bw / 2) / W);
        p.segments.y.push((y + Math.sin(a) * bh / 2) / H);
      }
    }
    if (S.task === 'pose') {
      p.kp = { x: [], y: [], v: [] };
      for (let i = 0; i < 17; i++) {
        p.kp.x.push((x + (Math.random() - 0.5) * bw) / W);
        p.kp.y.push((y + (Math.random() - 0.5) * bh) / H);
        p.kp.v.push(Math.random() > 0.3 ? 1 : 0);
      }
    }
    return p;
  });

  return { predictions: preds, _demo: true };
}

// ════════════════════════════════════════
// DRAW ANNOTATIONS
// ════════════════════════════════════════
function color(name) {
  if (!S.colors[name]) S.colors[name] = PAL[Object.keys(S.colors).length % PAL.length];
  return S.colors[name];
}

function rgba(h, a) {
  const r = parseInt(h.slice(1,3), 16);
  const g = parseInt(h.slice(3,5), 16);
  const b = parseInt(h.slice(5,7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}

function draw(result, lat) {
  const preds = result.predictions || [];
  const oc    = document.getElementById('outputCanvas');
  const ctx   = oc.getContext('2d');
  ctx.clearRect(0, 0, oc.width, oc.height);
  const W = oc.width, H = oc.height;

  preds.forEach(p => {
    const col = color(p.name);
    const cp  = (p.confidence * 100).toFixed(1);

    if (S.task === 'classify') {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, H - 64, W, 64);
      ctx.fillStyle = col;
      ctx.font = `bold ${Math.min(W / 8, 40)}px 'Barlow Condensed',sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(p.name.toUpperCase(), W / 2, H - 30);
      ctx.font = `${Math.min(W / 12, 20)}px 'IBM Plex Mono',monospace`;
      ctx.fillStyle = 'rgba(255,255,255,.7)';
      ctx.fillText(`${cp}% confidence`, W / 2, H - 9);
      return;
    }

    if (!p.box) return;
    const { x1, y1, x2, y2 } = p.box;
    const bw = x2 - x1, bh = y2 - y1;

    if (p.segments) {
      if (p.segments.mask) {
        const { mask, ph, pw } = p.segments;
        const [r, g, b] = hexToRgb(col);
        const mc    = document.createElement('canvas');
        mc.width = pw; mc.height = ph;
        const mCtx  = mc.getContext('2d');
        const mData = mCtx.createImageData(pw, ph);
        for (let i = 0; i < ph * pw; i++) {
          if (mask[i] > 0.5) {
            mData.data[i*4]   = r;
            mData.data[i*4+1] = g;
            mData.data[i*4+2] = b;
            mData.data[i*4+3] = 128;
          }
        }
        mCtx.putImageData(mData, 0, 0);
        ctx.drawImage(mc, 0, 0, pw, ph, x1, y1, bw, bh);
      } else if (p.segments.x) {
        const xs = p.segments.x, ys = p.segments.y;
        ctx.beginPath();
        ctx.moveTo(xs[0] * W, ys[0] * H);
        xs.forEach((x, i) => ctx.lineTo(x * W, ys[i] * H));
        ctx.closePath();
        ctx.fillStyle = rgba(col, 0.25); ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }

    if (p.kp) {
      const SKEL = [
        [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6],
        [5,7],[6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],
      ];
      SKEL.forEach(([a, b]) => {
        if (p.kp.v[a] && p.kp.v[b]) {
          ctx.beginPath();
          ctx.moveTo(p.kp.x[a] * W, p.kp.y[a] * H);
          ctx.lineTo(p.kp.x[b] * W, p.kp.y[b] * H);
          ctx.strokeStyle = rgba(col, 0.7); ctx.lineWidth = 2; ctx.stroke();
        }
      });
      p.kp.x.forEach((x, i) => {
        if (p.kp.v[i]) {
          ctx.beginPath();
          ctx.arc(x * W, p.kp.y[i] * H, 4, 0, Math.PI * 2);
          ctx.fillStyle = col; ctx.fill();
        }
      });
    }

    const cs = Math.min(bw, bh) * 0.17;
    ctx.strokeStyle = rgba(col, 0.4); ctx.lineWidth = 1;
    ctx.strokeRect(x1, y1, bw, bh);
    ctx.strokeStyle = col; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(x1, y1+cs); ctx.lineTo(x1, y1); ctx.lineTo(x1+cs, y1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2-cs, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1+cs); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y2-cs); ctx.lineTo(x1, y2); ctx.lineTo(x1+cs, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2-cs, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2-cs); ctx.stroke();

    const lbl = `${p.name}  ${cp}%`;
    ctx.font = "bold 10px 'IBM Plex Mono',monospace";
    const tw = ctx.measureText(lbl).width;
    const lx = x1, ly = y1 > 22 ? y1 - 20 : y1 + 2;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.roundRect(lx, ly, tw + 12, 18, 2); ctx.fill();
    ctx.fillStyle = '#000'; ctx.textAlign = 'left';
    ctx.fillText(lbl, lx + 6, ly + 13);
  });

  updateMetrics(preds, lat);
}

// ════════════════════════════════════════
// METRICS
// ════════════════════════════════════════
function updateMetrics(preds, lat) {
  const n   = preds.length;
  const avg = n ? (preds.reduce((a, p) => a + p.confidence, 0) / n * 100).toFixed(1) : '—';

  document.getElementById('mObj').textContent   = n;
  document.getElementById('mConf').textContent  = avg;
  document.getElementById('mTime').textContent  = Math.round(lat);
  document.getElementById('tFrame').textContent = ++S.frame;
  document.getElementById('tLat').textContent   = Math.round(lat) + 'ms';
  document.getElementById('tFps').textContent   = lat > 0 ? (1000 / lat).toFixed(1) : '—';
  document.getElementById('tObj').textContent   = n;

  S.lats.push(Math.round(lat));
  if (S.lats.length > 50) S.lats.shift();
  drawPerf(S.lats);

  const byC = {};
  preds.forEach(p => {
    if (!byC[p.name]) byC[p.name] = { c: 0, s: 0 };
    byC[p.name].c++; byC[p.name].s += p.confidence;
  });
  const body  = document.getElementById('detBody');
  const empty = document.getElementById('detEmpty');
  const table = document.getElementById('detTable');
  body.innerHTML = '';
  if (!n) {
    empty.style.display = 'block'; table.style.display = 'none';
  } else {
    empty.style.display = 'none'; table.style.display = 'table';
    Object.entries(byC).sort((a, b) => b[1].c - a[1].c).forEach(([name, d]) => {
      const col = color(name);
      const ac  = (d.s / d.c * 100).toFixed(0);
      const tr  = document.createElement('tr');
      tr.innerHTML = `
        <td><div class="cls-name">
          <div class="cls-dot" style="background:${col}"></div>${name}
        </div></td>
        <td><span class="cnt">${d.c}</span></td>
        <td><div style="display:flex;align-items:center;gap:5px;">
          <div class="cbar-wrap"><div class="cbar" style="width:${ac}%;background:${col}"></div></div>
          <span style="font-size:8px;color:var(--dim)">${ac}%</span>
        </div></td>`;
      body.appendChild(tr);
    });
  }

  const leg = document.getElementById('legend');
  leg.innerHTML = '';
  Object.entries(S.colors).forEach(([name, c]) => {
    const d = document.createElement('div');
    d.style.cssText = 'display:flex;align-items:center;gap:4px;font-family:var(--mono);font-size:8px;color:var(--dim);';
    d.innerHTML = `<div style="width:8px;height:8px;border-radius:1px;background:${c};flex-shrink:0"></div>${name}`;
    leg.appendChild(d);
  });
}

function logEntry(time, count, result) {
  const log     = document.getElementById('log');
  const isEmpty = log.querySelector('.empty');
  if (isEmpty) isEmpty.remove();
  const classes = [...new Set((result.predictions || []).map(p => p.name))];
  const row = document.createElement('div');
  row.className = 'log-row';
  row.innerHTML = `<span class="log-ts">${fmt(time)}</span>
    <span class="log-msg">${count} obj · ${classes.slice(0,4).join(', ')}${classes.length > 4 ? '…' : ''}</span>`;
  log.insertBefore(row, log.firstChild);
  if (log.children.length > 60) log.removeChild(log.lastChild);
}

// ════════════════════════════════════════
// PERFORMANCE CHART
// ════════════════════════════════════════
function drawPerf(data) {
  const c = document.getElementById('perfCanvas');
  const W = c.offsetWidth || 270, H = 56;
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(255,255,255,.03)';
  ctx.fillRect(0, 0, W, H);

  if (data.length < 2) {
    ctx.fillStyle = '#454d66';
    ctx.font = "9px 'IBM Plex Mono'";
    ctx.textAlign = 'center';
    ctx.fillText('Run inference to see chart', W / 2, H / 2 + 3);
    return;
  }

  const max  = Math.max(...data, 1);
  const step = W / (data.length - 1);
  [0.25, 0.5, 0.75].forEach(f => {
    ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H * f); ctx.lineTo(W, H * f); ctx.stroke();
  });

  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, 'rgba(255,180,0,.3)');
  g.addColorStop(1, 'rgba(255,180,0,0)');
  ctx.beginPath();
  ctx.moveTo(0, H);
  data.forEach((v, i) => ctx.lineTo(i * step, H - (v / max) * H * 0.9));
  ctx.lineTo((data.length - 1) * step, H);
  ctx.closePath();
  ctx.fillStyle = g; ctx.fill();

  ctx.beginPath();
  data.forEach((v, i) => {
    i === 0
      ? ctx.moveTo(0, H - (v / max) * H * 0.9)
      : ctx.lineTo(i * step, H - (v / max) * H * 0.9);
  });
  ctx.strokeStyle = '#ffb400'; ctx.lineWidth = 1.5; ctx.stroke();

  const lx = (data.length - 1) * step;
  const ly = H - (data[data.length - 1] / max) * H * 0.9;
  ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#ffb400'; ctx.fill();
}

// ════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════
function fmt(s) {
  const m = Math.floor(s / 60);
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

function seekVid(e) {
  const v = S.mediaEl;
  if (!v || !v.duration) return;
  const r = e.currentTarget.getBoundingClientRect();
  v.currentTime = ((e.clientX - r.left) / r.width) * v.duration;
}

function setStat(state, txt) {
  const d = document.getElementById('sDot');
  const t = document.getElementById('sTxt');
  d.className   = 'dot ' + ({ ready:'ready', busy:'busy', error:'err', off:'' }[state] || '');
  t.textContent = txt;
}

function showProc(on, txt) {
  document.getElementById('procOverlay').classList.toggle('on', on);
  if (txt) document.getElementById('procTxt').textContent = txt;
}

function swTab(name, btn) {
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('on'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('tp-' + name).classList.add('on');
  if (name === 'perf') setTimeout(() => drawPerf(S.lats), 40);
}

function updInfo() {
  const t = { detect:'DETECT', segment:'SEGMENT', classify:'CLASSIFY', pose:'POSE', obb:'OBB' };
  document.getElementById('iTask').textContent = t[S.task] || S.task.toUpperCase();
  document.getElementById('iSz').textContent   = document.getElementById('imgsize').value + 'px';
  document.getElementById('iRuns').textContent = S.runs;
  if (S.lats.length) {
    document.getElementById('iAvg').textContent =
      (S.lats.reduce((a, v) => a + v, 0) / S.lats.length).toFixed(0) + ' ms';
  }
}

function updSteps() {
  const sn = (id, st) => {
    const e = document.getElementById(id);
    if (!e) return;
    e.className   = 'sn' + (st === 'done' ? ' done' : st === 'active' ? ' active' : '');
    e.textContent = st === 'done' ? '✓' : id.slice(2);
  };
  const hm = S.loaded;
  const hd = !!S.mediaEl || S.inputType === 'camera' || S.inputType === 'rtsp';
  sn('sn1', hm ? 'done' : 'active');
  sn('sn2', hm ? 'done' : '');
  sn('sn3', hm && hd ? 'done' : hm ? 'active' : '');
  sn('sn4', hm && hd ? 'active' : '');
}

function resetAll() {
  stopInference();
  S.frame = 0; S.runs = 0; S.lats = []; S.colors = {}; S.mediaEl = null;
  S.vfCount = 0; S.vfTotal = 0; S.vfPeak = 0;
  document.getElementById('placeholder').style.display  = 'flex';
  document.getElementById('canvasBox').style.display    = 'none';
  document.getElementById('vidBar').classList.remove('on');
  document.getElementById('detEmpty').style.display     = 'block';
  document.getElementById('detTable').style.display     = 'none';
  document.getElementById('detBody').innerHTML          = '';
  document.getElementById('legend').innerHTML =
    '<span style="font-family:var(--mono);font-size:8px;color:var(--muted);">—</span>';
  document.getElementById('log').innerHTML =
    '<div class="empty">Use video or camera<br>to populate timeline.</div>';
  ['mObj','mConf','mTime','tFrame','tFps','tLat','tObj','vsF','vsP','vsA']
    .forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('mObj').textContent = '0';
  drawPerf([]); updInfo(); updSteps();
  document.getElementById('mediaFile').value = '';
  toast('↺ Reset', 'inf');
}

function exportJSON() {
  const d = {
    timestamp:  new Date().toISOString(),
    app:        'MiraSight Vision Inference',
    task:       S.task,
    runs:       S.runs,
    avgLatency: S.lats.length
      ? (S.lats.reduce((a, v) => a + v, 0) / S.lats.length).toFixed(1)
      : null,
    latencies: S.lats,
    classes:   Object.keys(S.colors),
  };
  const a = document.createElement('a');
  a.href     = URL.createObjectURL(
    new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' })
  );
  a.download = `mirasight-${Date.now()}.json`;
  a.click();
  toast('✓ Exported', 'ok');
}

function snapshot() {
  const ic = document.getElementById('inputCanvas');
  const oc = document.getElementById('outputCanvas');
  if (!ic.width) { toast('No frame to capture', 'err'); return; }
  const snap = document.createElement('canvas');
  snap.width = ic.width; snap.height = ic.height;
  const ctx = snap.getContext('2d');
  ctx.drawImage(ic, 0, 0);
  ctx.drawImage(oc, 0, 0);
  const a = document.createElement('a');
  a.href     = snap.toDataURL('image/png');
  a.download = `snap-${Date.now()}.png`;
  a.click();
  toast('✓ Snapshot saved', 'ok');
}

function toast(msg, type = 'inf') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span style="opacity:.6">${{ ok:'✓', err:'✕', inf:'◈' }[type] || '◈'}</span> ${msg}`;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// Expose to window for inline HTML onclick= handlers
Object.assign(window, {
  loadModel, setInput, handleMedia, runInference, stopInference,
  resetAll, exportJSON, snapshot, doZoom, swTab, seekVid,
});
