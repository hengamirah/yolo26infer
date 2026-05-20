import * as ort from 'onnxruntime-web';

let session = null;
let classNames = [];

export function setClassNames(names) { classNames = names; }
export function getSession()         { return session; }
export function clearSession()       { session = null; }

function cls(i) {
  return classNames.length > i ? classNames[i] : `class_${i}`;
}

// Load an ONNX model from a File object or a URL string
export async function loadModel(fileOrUrl) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;

  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };

  if (fileOrUrl instanceof File || fileOrUrl instanceof Blob) {
    const buffer = await fileOrUrl.arrayBuffer();
    session = await ort.InferenceSession.create(buffer, opts);
  } else {
    session = await ort.InferenceSession.create(fileOrUrl, opts);
  }
  return session;
}

// Letterbox-resize source into a sz×sz offscreen canvas and return a CHW Float32 tensor
// plus the transform state needed to map model coords back to original image coords.
function letterboxTensor(source, sz) {
  const srcW = source.videoWidth  || source.naturalWidth  || source.width;
  const srcH = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(sz / srcW, sz / srcH);
  const newW  = Math.round(srcW * scale);
  const newH  = Math.round(srcH * scale);
  const padX  = Math.floor((sz - newW) / 2);
  const padY  = Math.floor((sz - newH) / 2);

  const off = document.createElement('canvas');
  off.width = sz; off.height = sz;
  const ctx = off.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, sz, sz);
  ctx.drawImage(source, padX, padY, newW, newH);

  const px = ctx.getImageData(0, 0, sz, sz).data;
  const n  = sz * sz;
  const t  = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    t[i]         = px[i * 4]     / 255;
    t[n + i]     = px[i * 4 + 1] / 255;
    t[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return {
    tensor: new ort.Tensor('float32', t, [1, 3, sz, sz]),
    scale, padX, padY, srcW, srcH,
  };
}

// Parse YOLO26 end-to-end detection output.
// tensor.dims = [1, max_det, D]  D >= 6
// Each row: [x1, y1, x2, y2, score, class_id, …extra]
// Coordinates are in model input pixel space (0..sz); they are sorted descending by score.
function parseYolo26(tensor, sz, imgW, imgH, scale, padX, padY, confT) {
  const data   = tensor.data;
  const maxDet = tensor.dims[1];
  const D      = tensor.dims[2];
  const invSc  = 1 / scale;
  const preds  = [];

  for (let i = 0; i < maxDet; i++) {
    const b     = i * D;
    const score = data[b + 4];
    if (score < confT) break; // YOLO26 outputs are sorted — safe to break early

    const classId = Math.round(data[b + 5]);

    // Undo letterbox padding and scale back to source image coordinates
    const x1 = Math.max(0,    (data[b + 0] - padX) * invSc);
    const y1 = Math.max(0,    (data[b + 1] - padY) * invSc);
    const x2 = Math.min(imgW, (data[b + 2] - padX) * invSc);
    const y2 = Math.min(imgH, (data[b + 3] - padY) * invSc);

    const pred = { name: cls(classId), confidence: score, box: { x1, y1, x2, y2 } };

    // Pose: D === 6 + 17*3 = 57  →  extract COCO 17 keypoints
    if (D === 57) {
      pred.kp = { x: [], y: [], v: [] };
      for (let k = 0; k < 17; k++) {
        const o = b + 6 + k * 3;
        pred.kp.x.push((data[o]     - padX) * invSc / imgW);
        pred.kp.y.push((data[o + 1] - padY) * invSc / imgH);
        pred.kp.v.push(data[o + 2] > 0.5 ? 1 : 0);
      }
    }

    preds.push(pred);
  }
  return preds;
}

// Segmentation variant: second output is a prototype tensor [1, nm, ph, pw].
// Detection rows have D = 6 + nm mask coefficients after the base 6 values.
function parseYolo26Seg(detTensor, protoTensor, sz, imgW, imgH, scale, padX, padY, confT) {
  const preds = parseYolo26(detTensor, sz, imgW, imgH, scale, padX, padY, confT);
  const D       = detTensor.dims[2];
  const nm      = D - 6;
  const data    = detTensor.data;
  const PROTO_H = protoTensor.dims[2];
  const PROTO_W = protoTensor.dims[3];
  const proto   = protoTensor.data;

  preds.forEach((pred, idx) => {
    const b      = idx * D;
    const coeffs = new Float32Array(nm);
    for (let m = 0; m < nm; m++) coeffs[m] = data[b + 6 + m];

    const mask = new Float32Array(PROTO_H * PROTO_W);
    for (let py = 0; py < PROTO_H; py++) {
      for (let px = 0; px < PROTO_W; px++) {
        let val = 0;
        for (let m = 0; m < nm; m++) {
          val += coeffs[m] * proto[m * PROTO_H * PROTO_W + py * PROTO_W + px];
        }
        mask[py * PROTO_W + px] = 1 / (1 + Math.exp(-val));
      }
    }
    pred.segments = { mask, ph: PROTO_H, pw: PROTO_W };
  });

  return preds;
}

// Main inference entry point.
// source   — canvas, video, or image element to run inference on
// inputSz  — square input size in px (e.g. 640); read from the UI slider
// confT    — confidence threshold
export async function inferFrame(source, inputSz, confT) {
  if (!session) return null;

  const { tensor, scale, padX, padY, srcW, srcH } =
    letterboxTensor(source, inputSz);

  const feeds   = { [session.inputNames[0]]: tensor };
  const out     = await session.run(feeds);
  const outNames = session.outputNames;
  const o0      = out[outNames[0]];

  // YOLO26 end-to-end: output shape [1, max_det, D] where D >= 6
  if (o0.dims.length === 3 && o0.dims[2] >= 6) {
    if (outNames.length > 1) {
      // Segmentation — second tensor is the prototype map
      const proto = out[outNames[1]];
      const preds = parseYolo26Seg(o0, proto, inputSz, srcW, srcH, scale, padX, padY, confT);
      return { predictions: preds };
    }
    const preds = parseYolo26(o0, inputSz, srcW, srcH, scale, padX, padY, confT);
    return { predictions: preds };
  }

  console.warn('Unrecognized YOLO26 output format:', o0.dims, '— export with end2end=True');
  return { predictions: [] };
}
