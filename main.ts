// png-squeeze: a lossless PNG re-encoder for 1280x720 RGBA screenshots.
//
// Strategy: decode the input PNG, reduce the colour type where the image
// allows it (palette for <=256 colours, grey / grey+alpha, RGB, or RGBA, with
// a tRNS key when a single fully-transparent value exists), pick per-row PNG
// filters by screening (fixed, minimum-sum, fewest-pairs, brute-force by
// zlib, and a static-Huffman cost model), then encode the filtered bytes with
// the smallest of several DEFLATE streams: zlib levels 6 / 9 (default and
// filtered strategies) and an own optimal-parsing DEFLATE (lazy hash-chain
// matching, two cost-model passes, dynamic Huffman with block splitting).
// Everything is deterministic: fixed work budgets, no clock, no randomness,
// no network, no files outside /input, /output and /bot.
//
// Run as: node /bot/main.ts   (Node 22; plain JavaScript, no dependencies)
// Reads /input/<id>/image.png (or $MAKEFUN_INPUT_DIR), writes
// /output/<id>/image.png (or $MAKEFUN_OUTPUT_DIR).

import { inflateSync, deflateSync, constants } from 'node:zlib';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------- CRC / Adler

const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(bytes, start = 0, end = bytes.length) {
  let c = -1;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function adler32(bytes) {
  let a = 1;
  let b = 0;
  for (let i = 0; i < bytes.length;) {
    const end = Math.min(bytes.length, i + 5552);
    for (; i < end; i++) {
      a += bytes[i];
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

const SIGNATURE = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
const ascii = (s) => Uint8Array.from(s, (ch) => ch.charCodeAt(0));

function pngOf(chunks) {
  let size = SIGNATURE.length;
  for (const [, data] of chunks) size += 12 + data.length;
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  out.set(SIGNATURE, 0);
  let at = SIGNATURE.length;
  for (const [type, data] of chunks) {
    view.setUint32(at, data.length);
    out.set(ascii(type), at + 4);
    out.set(data, at + 8);
    view.setUint32(at + 8 + data.length, crc32(out, at + 4, at + 8 + data.length));
    at += 12 + data.length;
  }
  return out;
}

function ihdrOf(width, height, colourType) {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  d[8] = 8;
  d[9] = colourType;
  return d;
}

// ---------------------------------------------------------------- PNG decode

const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = p > a ? p - a : a - p;
  const pb = p > b ? p - b : b - p;
  const pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

// Decode a bit-depth-8, non-interlaced PNG of colour type 0/2/3/4/6 to RGBA.
function decodePng(bytes) {
  if (bytes.length < 8 || SIGNATURE.some((b, i) => bytes[i] !== b)) throw new Error('bad PNG signature');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 8;
  let header = null;
  let palette = null;
  let trns = null;
  const idat = [];
  let ended = false;
  while (at < bytes.length) {
    if (ended) throw new Error('bytes after IEND');
    const len = view.getUint32(at);
    if (at + 12 + len > bytes.length) throw new Error('chunk cut short');
    const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
    if (crc32(bytes, at + 4, at + 8 + len) !== view.getUint32(at + 8 + len)) throw new Error(`${type} CRC`);
    const data = bytes.subarray(at + 8, at + 8 + len);
    at += 12 + len;
    if (type === 'IHDR') {
      const dv = new DataView(data.buffer, data.byteOffset, 13);
      header = { width: dv.getUint32(0), height: dv.getUint32(4), depth: data[8], colourType: data[9], interlace: data[12] };
      if (header.depth !== 8 || !(header.colourType in CHANNELS) || header.interlace !== 0) throw new Error('unsupported IHDR');
    } else if (type === 'PLTE') palette = data;
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') ended = true;
    else throw new Error(`unexpected chunk ${type}`);
  }
  if (!ended || idat.length === 0) throw new Error('no IEND/IDAT');
  const { width, height, colourType } = header;
  const bpp = CHANNELS[colourType];
  const stride = width * bpp;
  const expected = height * (stride + 1);
  const raw = inflateSync(idat.length === 1 ? idat[0] : Buffer.concat(idat), { maxOutputLength: expected + 1 });
  if (raw.length !== expected) throw new Error('IDAT size mismatch');
  const samples = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    if (ft > 4) throw new Error('bad filter type');
    const src = y * (stride + 1) + 1;
    const o = y * stride;
    const p = o - stride;
    if (ft === 0) samples.set(raw.subarray(src, src + stride), o);
    else if (ft === 1) {
      for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (i >= bpp ? samples[o + i - bpp] : 0);
    } else if (ft === 2) {
      for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (y > 0 ? samples[p + i] : 0);
    } else if (ft === 3) {
      for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (((i >= bpp ? samples[o + i - bpp] : 0) + (y > 0 ? samples[p + i] : 0)) >> 1);
    } else {
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? samples[o + i - bpp] : 0;
        const b = y > 0 ? samples[p + i] : 0;
        const c = y > 0 && i >= bpp ? samples[p + i - bpp] : 0;
        samples[o + i] = raw[src + i] + paeth(a, b, c);
      }
    }
  }
  const n = width * height;
  const rgba = new Uint8Array(n * 4);
  if (colourType === 6) rgba.set(samples);
  else if (colourType === 2) {
    const key = trns === null ? -1 : (trns[1] << 16) | (trns[3] << 8) | trns[5];
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 3, d += 4) {
      rgba[d] = samples[s];
      rgba[d + 1] = samples[s + 1];
      rgba[d + 2] = samples[s + 2];
      rgba[d + 3] = ((samples[s] << 16) | (samples[s + 1] << 8) | samples[s + 2]) === key ? 0 : 255;
    }
  } else if (colourType === 0) {
    const key = trns === null ? -1 : trns[1];
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const g = samples[i];
      rgba[d] = rgba[d + 1] = rgba[d + 2] = g;
      rgba[d + 3] = g === key ? 0 : 255;
    }
  } else if (colourType === 4) {
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 2, d += 4) {
      rgba[d] = rgba[d + 1] = rgba[d + 2] = samples[s];
      rgba[d + 3] = samples[s + 1];
    }
  } else {
    const entries = palette.length / 3;
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const e = samples[i];
      if (e >= entries) throw new Error('palette index out of range');
      rgba[d] = palette[e * 3];
      rgba[d + 1] = palette[e * 3 + 1];
      rgba[d + 2] = palette[e * 3 + 2];
      rgba[d + 3] = trns !== null && e < trns.length ? trns[e] : 255;
    }
  }
  return { width, height, rgba };
}

// ---------------------------------------------------------------- Image analysis

function wordsOf(rgba) {
  const at = rgba.byteOffset % 4 === 0 ? rgba : rgba.slice();
  return new Uint32Array(at.buffer, at.byteOffset, at.length >>> 2);
}

// Whether every pixel is opaque, whether every pixel is grey, and the colour
// histogram when the image holds at most 256 distinct RGBA values.
function analyse(rgba) {
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  let opaque = true;
  let grey = true;
  const counts = new Map();
  let many = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!many) {
      const c = u32[i];
      const k = counts.get(c);
      if (k === undefined) {
        if (counts.size === 256) many = true;
        else counts.set(c, 1);
      } else counts.set(c, k + 1);
    }
  }
  let colours = null;
  let firstSeen = null;
  if (!many) {
    colours = [...counts.keys()].map((c) => {
      const b = new Uint8Array(Uint32Array.of(c).buffer);
      return { rgba: [b[0], b[1], b[2], b[3]], word: c, count: counts.get(c) };
    });
    // First-appearance order, for a palette order that follows scan order.
    firstSeen = new Map();
    for (let i = 0; i < n; i++) {
      const c = u32[i];
      if (!firstSeen.has(c)) firstSeen.set(c, i);
    }
  }
  return { opaque, grey, colours, firstSeen };
}

// The one fully transparent value (grey sample or RGB triple) the image may
// key by tRNS instead of carrying an alpha channel, or null.
function transparentKey(rgba, grey) {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}

function samplesOf(rgba, ct) {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 3];
    } else {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 1];
      out[s + 2] = rgba[d + 2];
    }
  }
  return out;
}

// Order a palette's entries: transparent entries first (short tRNS), then by
// the named order. 'greedy' walks nearest-neighbour through colour space so
// neighbouring indices hold similar colours and Sub filtering stays small.
function paletteOrder(colours, order, firstSeen) {
  const lum = (c) => 299 * c.rgba[0] + 587 * c.rgba[1] + 114 * c.rgba[2];
  const transparentFirst = (x, y) => {
    const ta = x.rgba[3] === 255 ? 1 : 0;
    const tb = y.rgba[3] === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
    return 0;
  };
  if (order === 'greedy') {
    const rest = [...colours].sort((x, y) => transparentFirst(x, y) || y.count - x.count || x.word - y.word);
    const out = [];
    let cur = rest.shift();
    out.push(cur);
    while (rest.length > 0) {
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < rest.length; i++) {
        const c = rest[i];
        const dr = c.rgba[0] - cur.rgba[0];
        const dg = c.rgba[1] - cur.rgba[1];
        const db = c.rgba[2] - cur.rgba[2];
        const da = (c.rgba[3] - cur.rgba[3]) * 4;
        const d = dr * dr + dg * dg + db * db + da * da - c.count * 0.001;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      cur = rest.splice(best, 1)[0];
      out.push(cur);
    }
    return out;
  }
  return [...colours].sort((x, y) => {
    const t = transparentFirst(x, y);
    if (t !== 0) return t;
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    if (order === 'first') return firstSeen.get(x.word) - firstSeen.get(y.word);
    if (order === 'rgb') return x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
}

function paletteForm(rgba, colours, order, firstSeen) {
  const entries = paletteOrder(colours, order, firstSeen);
  const index = new Map(entries.map((c, i) => [c.word, i]));
  const plte = new Uint8Array(entries.length * 3);
  let transparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.rgba[0];
    plte[i * 3 + 1] = c.rgba[1];
    plte[i * 3 + 2] = c.rgba[2];
    if (c.rgba[3] !== 255) transparent = i + 1;
  });
  const trns = transparent === 0 ? null : Uint8Array.from(entries.slice(0, transparent), (c) => c.rgba[3]);
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  const indices = new Uint8Array(n);
  for (let i = 0; i < n; i++) indices[i] = index.get(u32[i]);
  return { plte, trns, indices };
}

// Every lossless form the image can take: palettes first (a palette of a
// non-grey image is never beaten by its RGB/RGBA form at a third of the bytes
// or less), then the channel form.
function formsOf(rgba) {
  const { opaque, grey, colours, firstSeen } = analyse(rgba);
  const forms = [];
  if (colours !== null) {
    for (const order of ['count', 'luminance', 'first', 'greedy', 'rgb']) {
      const p = paletteForm(rgba, colours, order, firstSeen);
      forms.push({ colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, filters: ['zero', 'minsum', 'pairs', 'brute'] });
    }
    if (!grey) return forms;
  }
  const key = opaque ? null : transparentKey(rgba, grey);
  let ct;
  let trns = null;
  if (grey) {
    ct = opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    ct = opaque || key !== null ? 2 : 6;
    if (key !== null) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[ct];
  forms.push({ colourType: ct, bpp, samples: samplesOf(rgba, ct), plte: null, trns, filters: ['zero', 'one', 'two', 'four', 'minsum', 'pairs', 'model', 'brute'] });
  return forms;
}

// ---------------------------------------------------------------- Filters

function filterRow(type, cur, prev, bpp, out) {
  const n = cur.length;
  if (type === 0) out.set(cur);
  else if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (i >= bpp ? cur[i - bpp] : 0);
  } else if (type === 2) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (prev === null ? 0 : prev[i]);
  } else if (type === 3) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (((i >= bpp ? cur[i - bpp] : 0) + (prev === null ? 0 : prev[i])) >> 1);
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = cur[i] - paeth(a, b, c);
    }
  }
}

function signedSum(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    s += v < 128 ? v : 256 - v;
  }
  return s;
}

function pairsOf(row, seen, stamp) {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const pair = (row[i - 1] << 8) | row[i];
    if (seen[pair] !== stamp) {
      seen[pair] = stamp;
      n++;
    }
  }
  return n;
}

// A static-Huffman cost model for filter choice: byte costs from the
// frequencies of a first (minimum-sum) filtering, scaled by 256.
function modelCosts(samples, width, height, bpp) {
  const first = filtered(samples, width, height, bpp, 'minsum');
  const freq = new Float64Array(256);
  for (let i = 0; i < first.length; i++) freq[first[i]]++;
  const n = first.length;
  const cost = new Float64Array(256);
  for (let b = 0; b < 256; b++) cost[b] = -Math.log2((freq[b] + 1) / (n + 256)) * 256;
  return cost;
}

const BRUTE_CONTEXT = 32768;
const BRUTE_LEVEL = 4;

// Filter `samples` by `choice` into a fresh buffer: a fixed type 0-4, or
// 'minsum' / 'pairs' / 'model' (per-row best by that cost) or 'brute'
// (per-row best by zlib at level 4 over the 32 KiB written before it).
// The named filter constants below ('zero', 'one', ...) map to types.
function filtered(samples, width, height, bpp, choice, modelCost = null) {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial = [];
  for (let t = 0; t < 5; t++) {
    const row = new Uint8Array(stride + 1);
    row[0] = t;
    trial.push(row);
  }
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    if (choice === 'zero' || choice === 'one' || choice === 'two' || choice === 'three' || choice === 'four') {
      const type = { zero: 0, one: 1, two: 2, three: 3, four: 4 }[choice];
      raw[at] = type;
      filterRow(type, cur, prev, bpp, raw.subarray(at + 1, at + 1 + stride));
      continue;
    }
    const before = raw.subarray(Math.max(0, at - BRUTE_CONTEXT), at);
    let type = 0;
    let best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t].subarray(1));
      let cost;
      if (choice === 'minsum') cost = signedSum(trial[t].subarray(1));
      else if (choice === 'pairs') cost = pairsOf(trial[t].subarray(1), seen, ++stamp);
      else if (choice === 'model') {
        let s = 0;
        const row = trial[t].subarray(1);
        for (let i = 0; i < row.length; i++) s += modelCost[row[i]];
        cost = s;
      } else {
        cost = deflateSync(trial[t], before.length === 0 ? { level: BRUTE_LEVEL, memLevel: 9 } : { level: BRUTE_LEVEL, memLevel: 9, dictionary: before }).length;
      }
      if (cost < best) {
        best = cost;
        type = t;
      }
    }
    raw.set(trial[type], at);
  }
  return raw;
}

const SCREEN = Object.freeze({ level: 6, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY });
const chunkBytes = (form) => (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);

// Screen every form and filter choice at zlib level 6; return the winner
// (smallest stream + PLTE/tRNS overhead) and the runner-up form for a second
// final encoding when it is a different colour type.
function screen(rgba, width, height) {
  const forms = formsOf(rgba);
  let best = null;
  let second = null;
  for (const form of forms) {
    const size = height * (width * form.bpp + 1);
    const chunks = chunkBytes(form);
    const mc = form.filters.includes('model') ? modelCosts(form.samples, width, height, form.bpp) : null;
    for (const choice of form.filters) {
      const data = filtered(form.samples, width, height, form.bpp, choice, mc);
      const zlib = deflateSync(data, SCREEN);
      const bytes = zlib.length + chunks;
      const cand = { form, choice, data, bytes };
      if (best === null || bytes < best.bytes) {
        second = best;
        best = cand;
      } else if (second === null || bytes < second.bytes) second = cand;
    }
  }
  return { best, second };
}

// ---------------------------------------------------------------- DEFLATE tables

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

const LEN_CODE_BY_LEN = (() => {
  const t = new Uint8Array(259);
  for (let len = 3; len <= 258; len++) {
    let c = 28;
    while (len < LEN_BASE[c]) c--;
    t[len] = c;
  }
  return t;
})();

const DIST_CODE_BY_DIST = (() => {
  const t = new Uint8Array(32769);
  for (let d = 1; d <= 32768; d++) {
    let c = 29;
    while (d < DIST_BASE[c]) c--;
    t[d] = c;
  }
  return t;
})();

// ---------------------------------------------------------------- Huffman codes

// Length-limited Huffman code lengths for `freq` (counts per symbol), max
// length 15 (7 for a code-length code): build a Huffman tree, cap depths,
// fix the overflow the zlib way, then hand the shortest lengths to the most
// frequent symbols. Returns a Uint8Array of lengths (0 = unused).
function buildLengths(freq, maxLen) {
  const n = freq.length;
  const lengths = new Uint8Array(n);
  const syms = [];
  for (let s = 0; s < n; s++) if (freq[s] > 0) syms.push(s);
  if (syms.length === 0) return lengths;
  if (syms.length === 1) {
    lengths[syms[0]] = 1;
    for (let s = 0; s < n; s++) {
      if (s !== syms[0]) {
        lengths[s] = 1;
        break;
      }
    }
    return lengths;
  }
  // Binary heap of node ids, smallest (freq, id) first.
  const nodeFreq = [];
  const nodeLeft = [];
  const nodeRight = [];
  const nodeSym = [];
  const addNode = (f, sym, l, r) => {
    nodeFreq.push(f);
    nodeSym.push(sym);
    nodeLeft.push(l);
    nodeRight.push(r);
    return nodeFreq.length - 1;
  };
  const heap = [];
  const less = (a, b) => nodeFreq[a] < nodeFreq[b] || (nodeFreq[a] === nodeFreq[b] && a < b);
  const push = (id) => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (less(heap[i], heap[p])) {
        const t = heap[i];
        heap[i] = heap[p];
        heap[p] = t;
        i = p;
      } else break;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        const t = heap[i];
        heap[i] = heap[m];
        heap[m] = t;
        i = m;
      }
    }
    return top;
  };
  for (const s of syms) push(addNode(freq[s], s, -1, -1));
  while (heap.length > 1) {
    const a = pop();
    const b = pop();
    push(addNode(nodeFreq[a] + nodeFreq[b], -1, a, b));
  }
  const root = heap[0];
  // Depths, capped at maxLen, counting the overflow.
  const blCount = new Uint32Array(maxLen + 2);
  let overflow = 0;
  let maxDepth = 0;
  const stack = [[root, 0]];
  while (stack.length > 0) {
    const [id, d] = stack.pop();
    if (nodeSym[id] >= 0) {
      let depth = d;
      if (depth > maxLen) {
        depth = maxLen;
        overflow++;
      }
      blCount[depth]++;
      if (depth > maxDepth) maxDepth = depth;
    } else {
      stack.push([nodeLeft[id], d + 1]);
      stack.push([nodeRight[id], d + 1]);
    }
  }
  if (overflow > 0) {
    do {
      let bits = maxLen - 1;
      while (blCount[bits] === 0) bits--;
      blCount[bits]--;
      blCount[bits + 1] += 2;
      blCount[maxLen]--;
      overflow -= 2;
    } while (overflow > 0);
  }
  // Hand the shortest lengths to the most frequent symbols.
  const order = [...syms].sort((a, b) => freq[b] - freq[a] || a - b);
  let k = 0;
  for (let len = 1; len <= maxLen; len++) {
    for (let j = 0; j < blCount[len]; j++) lengths[order[k++]] = len;
  }
  return lengths;
}

// Canonical Huffman codes from lengths: { codes, lengths } with codes as
// bit-reversed values ready to write LSB-first.
function canonicalCodes(lengths) {
  const n = lengths.length;
  const codes = new Uint32Array(n);
  const blCount = new Uint32Array(16);
  for (let s = 0; s < n; s++) if (lengths[s] > 0) blCount[lengths[s]]++;
  const nextCode = new Uint32Array(16);
  let code = 0;
  for (let bits = 1; bits <= 15; bits++) {
    code = (code + blCount[bits - 1]) << 1;
    nextCode[bits] = code;
  }
  for (let s = 0; s < n; s++) {
    const len = lengths[s];
    if (len === 0) continue;
    // Reverse the code's bits: DEFLATE writes Huffman codes MSB-first into
    // the bit stream, which is LSB-first overall, so store them reversed.
    let c = nextCode[len]++;
    let rev = 0;
    for (let i = 0; i < len; i++) {
      rev = (rev << 1) | (c & 1);
      c >>>= 1;
    }
    codes[s] = rev;
  }
  return codes;
}

// ---------------------------------------------------------------- Bit writer

class BitWriter {
  constructor(capacity) {
    this.buf = new Uint8Array(capacity);
    this.pos = 0;
    this.bitBuf = 0;
    this.bitCnt = 0;
  }
  grow() {
    const nb = new Uint8Array(this.buf.length * 2);
    nb.set(this.buf.subarray(0, this.pos));
    this.buf = nb;
  }
  writeBits(value, count) {
    this.bitBuf |= value << this.bitCnt;
    this.bitCnt += count;
    while (this.bitCnt >= 8) {
      if (this.pos >= this.buf.length) this.grow();
      this.buf[this.pos++] = this.bitBuf & 0xff;
      this.bitBuf >>>= 8;
      this.bitCnt -= 8;
    }
  }
  align() {
    if (this.bitCnt > 0) {
      if (this.pos >= this.buf.length) this.grow();
      this.buf[this.pos++] = this.bitBuf & 0xff;
      this.bitBuf = 0;
      this.bitCnt = 0;
    }
  }
  bytes() {
    return this.buf.subarray(0, this.pos);
  }
}

// ---------------------------------------------------------------- LZ77 parse

// A cost model: per-symbol bit costs for literals/lengths/distances.
function costModel(litLengths, distLengths) {
  const litCost = new Float64Array(288);
  for (let s = 0; s < 256; s++) litCost[s] = litLengths[s];
  for (let c = 0; c < 29; c++) litCost[257 + c] = litLengths[257 + c];
  const lenCostByLen = new Float64Array(259);
  for (let len = 3; len <= 258; len++) {
    const c = LEN_CODE_BY_LEN[len];
    lenCostByLen[len] = litCost[257 + c] + LEN_EXTRA[c];
  }
  const distCost = new Float64Array(30);
  for (let c = 0; c < 30; c++) distCost[c] = distLengths[c];
  const distCostByDist = new Float64Array(32769);
  for (let d = 1; d <= 32768; d++) {
    const c = DIST_CODE_BY_DIST[d];
    distCostByDist[d] = distCost[c] + DIST_EXTRA[c];
  }
  return { litCost, lenCostByLen, distCostByDist };
}

const FIXED_LIT_LENGTHS = (() => {
  const l = new Uint8Array(288);
  for (let s = 0; s < 144; s++) l[s] = 8;
  for (let s = 144; s < 256; s++) l[s] = 9;
  for (let s = 256; s < 280; s++) l[s] = 7;
  for (let s = 280; s < 288; s++) l[s] = 8;
  return l;
})();
const FIXED_DIST_LENGTHS = new Uint8Array(30).fill(5);
const FIXED_MODEL = costModel(FIXED_LIT_LENGTHS, FIXED_DIST_LENGTHS);

// Parse `data` into tokens with lazy hash-chain matching under a cost model.
// Tokens: tSym[i] is a literal byte, or 257 + length-code index for a match;
// tLen[i] / tDist[i] hold the match's exact length and distance.
function parse(data, model, maxChain) {
  const n = data.length;
  const head = new Int32Array(32768).fill(-1);
  const prev = new Int32Array(n).fill(-1);
  const tSym = new Uint16Array(n + 1);
  const tLen = new Uint16Array(n + 1);
  const tDist = new Uint16Array(n + 1);
  let nt = 0;
  let nextInsert = 0;
  const { litCost, lenCostByLen, distCostByDist } = model;

  const insertPos = (p) => {
    const h = (Math.imul(((data[p] << 16) ^ (data[p + 1] << 8) ^ data[p + 2]), 2654435761) >>> 17) & 32767;
    prev[p] = head[h];
    head[h] = p;
  };
  const ensureInserted = (upto) => {
    const lim = Math.min(upto, n - 3);
    while (nextInsert <= lim) {
      insertPos(nextInsert);
      nextInsert++;
    }
  };
  // Find the longest match at position i. The hash chain holds only
  // positions before i: `start` is the newest such candidate (prev[i] once
  // i is inserted, head[h] while it is not).
  const findMatch = (i, chainLimit, start) => {
    const maxLen = Math.min(258, n - i);
    let cand = start;
    let bestLen = 0;
    let bestDist = 0;
    let chain = chainLimit;
    const minCand = i - 32768;
    while (cand >= minCand && cand >= 0 && chain-- > 0) {
      if (bestLen > 0 && data[cand + bestLen] !== data[i + bestLen]) {
        cand = prev[cand];
        continue;
      }
      let len = 0;
      while (len < maxLen && data[cand + len] === data[i + len]) len++;
      if (len > bestLen) {
        bestLen = len;
        bestDist = i - cand;
        if (len >= maxLen) break;
      }
      cand = prev[cand];
    }
    return [bestLen, bestDist];
  };
  const hashOf = (i) => (Math.imul(((data[i] << 16) ^ (data[i + 1] << 8) ^ data[i + 2]), 2654435761) >>> 17) & 32767;

  let i = 0;
  let prevLen = 0;
  while (i < n) {
    if (i + 3 > n) {
      while (i < n) {
        tSym[nt] = data[i];
        nt++;
        i++;
      }
      break;
    }
    ensureInserted(i);
    const chainLimit = prevLen >= 32 ? (maxChain >> 2) : maxChain;
    // The chain for i starts at prev[i]: the newest position before i with
    // the same hash (i itself was just inserted above).
    const [len1, dist1] = findMatch(i, chainLimit, prev[i]);
    if (len1 >= 3 && len1 < 258) {
      // Lazy: is the match one byte later better? Take the literal when the
      // next match is longer (or as long but nearer): the two paths cover
      // different spans, so a raw bit-cost comparison would favour a short
      // match at i over a long one at i+1 and burn the long match.
      const [len2, dist2] = findMatch(i + 1, chainLimit, head[hashOf(i + 1)]);
      if (len2 > len1 || (len2 === len1 && distCostByDist[dist2] < distCostByDist[dist1])) {
        tSym[nt] = data[i];
        nt++;
        i++;
        prevLen = 0;
        continue;
      }
    }
    if (len1 >= 3) {
      tSym[nt] = 257 + LEN_CODE_BY_LEN[len1];
      tLen[nt] = len1;
      tDist[nt] = dist1;
      nt++;
      ensureInserted(i + len1 - 1);
      i += len1;
      prevLen = len1;
    } else {
      tSym[nt] = data[i];
      nt++;
      i++;
      prevLen = 0;
    }
  }
  return { tSym: tSym.subarray(0, nt), tLen: tLen.subarray(0, nt), tDist: tDist.subarray(0, nt), nt };
}

// ---------------------------------------------------------------- Token encoding

const CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

// RLE-encode a code-length sequence into (symbol, extra, extraBits) triples.
function rleLengths(lengths) {
  const out = [];
  let i = 0;
  const n = lengths.length;
  while (i < n) {
    const v = lengths[i];
    let run = 1;
    while (i + run < n && lengths[i + run] === v) run++;
    if (v === 0) {
      let left = run;
      while (left >= 3) {
        if (left <= 10) {
          out.push([17, left - 3, 3]);
          left = 0;
        } else {
          const take = Math.min(left, 138);
          out.push([18, take - 11, 7]);
          left -= take;
        }
      }
      while (left-- > 0) out.push([0, 0, 0]);
    } else {
      out.push([v, 0, 0]);
      let left = run - 1;
      while (left >= 3) {
        const take = Math.min(left, 6);
        out.push([16, take - 3, 2]);
        left -= take;
      }
      while (left-- > 0) out.push([v, 0, 0]);
    }
    i += run;
  }
  return out;
}

function writeDynamicHeader(bw, litLengths, distLengths) {
  let hlitCount = 257;
  for (let s = 285; s >= 257; s--) if (litLengths[s] > 0) { hlitCount = s + 1; break; }
  let hdistCount = 1;
  for (let s = 29; s >= 0; s--) if (distLengths[s] > 0) { hdistCount = s + 1; break; }
  const seq = new Uint8Array(hlitCount + hdistCount);
  seq.set(litLengths.subarray(0, hlitCount), 0);
  seq.set(distLengths.subarray(0, hdistCount), hlitCount);
  const rle = rleLengths(seq);
  const clenFreq = new Uint32Array(19);
  for (const [s] of rle) clenFreq[s]++;
  const clenLengths = buildLengths(clenFreq, 7);
  const clenCodes = canonicalCodes(clenLengths);
  let hclen = 19;
  while (hclen > 4 && clenLengths[CLEN_ORDER[hclen - 1]] === 0) hclen--;
  bw.writeBits(2, 2); // BTYPE dynamic
  bw.writeBits(hlitCount - 257, 5);
  bw.writeBits(hdistCount - 1, 5);
  bw.writeBits(hclen - 4, 4);
  for (let k = 0; k < hclen; k++) bw.writeBits(clenLengths[CLEN_ORDER[k]], 3);
  for (const [s, extra, bits] of rle) {
    bw.writeBits(clenCodes[s], clenLengths[s]);
    if (bits > 0) bw.writeBits(extra, bits);
  }
}

function writeTokens(bw, toks, from, to, litCodes, litLengths, distCodes, distLengths) {
  const { tSym, tLen, tDist } = toks;
  for (let i = from; i < to; i++) {
    const s = tSym[i];
    if (s < 256) {
      bw.writeBits(litCodes[s], litLengths[s]);
    } else {
      const lc = s - 257;
      const len = tLen[i];
      bw.writeBits(litCodes[257 + lc], litLengths[257 + lc]);
      if (LEN_EXTRA[lc] > 0) bw.writeBits(len - LEN_BASE[lc], LEN_EXTRA[lc]);
      const dist = tDist[i];
      const dc = DIST_CODE_BY_DIST[dist];
      bw.writeBits(distCodes[dc], distLengths[dc]);
      if (DIST_EXTRA[dc] > 0) bw.writeBits(dist - DIST_BASE[dc], DIST_EXTRA[dc]);
    }
  }
  bw.writeBits(litCodes[256], litLengths[256]);
}

// Frequencies of a token range.
function tokenFreqs(toks, from, to) {
  const litFreq = new Uint32Array(288);
  const distFreq = new Uint32Array(30);
  const { tSym, tLen, tDist } = toks;
  for (let i = from; i < to; i++) {
    const s = tSym[i];
    litFreq[s]++;
    if (s >= 256) distFreq[DIST_CODE_BY_DIST[tDist[i]]]++;
  }
  litFreq[256]++; // end of block
  if (distFreq.every((f) => f === 0)) distFreq[0] = 1; // a block needs a distance code
  return { litFreq, distFreq };
}

// Encode tokens as one dynamic block, or split into blocks of at most
// `splitBytes` input bytes, and as one fixed-Huffman block; return the
// smallest raw DEFLATE stream.
function encodeTokens(toks, dataLen, splitBytes) {
  const capacity = Math.floor(dataLen * 1.15) + 8192;
  const variants = [];
  // Whole stream, one dynamic block.
  {
    const { litFreq, distFreq } = tokenFreqs(toks, 0, toks.nt);
    const litLengths = buildLengths(litFreq, 15);
    const distLengths = buildLengths(distFreq, 15);
    const bw = new BitWriter(capacity);
    bw.writeBits(1, 1); // BFINAL
    writeDynamicHeader(bw, litLengths, distLengths);
    writeTokens(bw, toks, 0, toks.nt, canonicalCodes(litLengths), litLengths, canonicalCodes(distLengths), distLengths);
    bw.align();
    variants.push(bw.bytes());
  }
  // Split into blocks.
  if (splitBytes > 0) {
    const bounds = [0];
    let covered = 0;
    for (let i = 0; i < toks.nt; i++) {
      covered += toks.tSym[i] < 256 ? 1 : toks.tLen[i];
      if (covered >= splitBytes && i + 1 < toks.nt) {
        bounds.push(i + 1);
        covered = 0;
      }
    }
    bounds.push(toks.nt);
    if (bounds.length > 2) {
      const bw = new BitWriter(capacity);
      for (let b = 0; b + 1 < bounds.length; b++) {
        const from = bounds[b];
        const to = bounds[b + 1];
        const last = b + 2 === bounds.length;
        const { litFreq, distFreq } = tokenFreqs(toks, from, to);
        const litLengths = buildLengths(litFreq, 15);
        const distLengths = buildLengths(distFreq, 15);
        bw.writeBits(last ? 1 : 0, 1); // BFINAL
        writeDynamicHeader(bw, litLengths, distLengths);
        writeTokens(bw, toks, from, to, canonicalCodes(litLengths), litLengths, canonicalCodes(distLengths), distLengths);
      }
      bw.align();
      variants.push(bw.bytes());
    }
  }
  // One fixed-Huffman block.
  {
    const bw = new BitWriter(capacity);
    bw.writeBits(1, 1); // BFINAL
    bw.writeBits(1, 2); // BTYPE fixed
    const litCodes = canonicalCodes(FIXED_LIT_LENGTHS);
    const distCodes = canonicalCodes(FIXED_DIST_LENGTHS);
    writeTokens(bw, toks, 0, toks.nt, litCodes, FIXED_LIT_LENGTHS, distCodes, FIXED_DIST_LENGTHS);
    bw.align();
    variants.push(bw.bytes());
  }
  let best = variants[0];
  for (const v of variants) if (v.length < best.length) best = v;
  return best;
}

// The own optimal-parsing DEFLATE of `data`: two parses (fixed costs, then
// costs from the first parse's frequencies), each encoded several ways;
// the smallest raw DEFLATE stream wins.
function optDeflate(data) {
  const n = data.length;
  const toks1 = parse(data, FIXED_MODEL, 4096);
  const { litFreq, distFreq } = tokenFreqs(toks1, 0, toks1.nt);
  const litLengths = buildLengths(litFreq, 15);
  const distLengths = buildLengths(distFreq, 15);
  const model2 = costModel(litLengths, distLengths);
  const toks2 = parse(data, model2, 2048);
  const best1 = encodeTokens(toks1, n, 131072);
  const best2 = encodeTokens(toks2, n, 131072);
  return best1.length <= best2.length ? best1 : best2;
}

function zlibWrap(raw, original) {
  const out = new Uint8Array(raw.length + 6);
  out[0] = 0x78;
  out[1] = 0xda;
  out.set(raw, 2);
  const v = new DataView(out.buffer);
  v.setUint32(out.length - 4, adler32(original));
  return out;
}

// ---------------------------------------------------------------- Final encoding

function pngOfForm(width, height, form, idat) {
  const chunks = [['IHDR', ihdrOf(width, height, form.colourType)]];
  if (form.plte !== null) chunks.push(['PLTE', form.plte]);
  if (form.trns !== null) chunks.push(['tRNS', form.trns]);
  chunks.push(['IDAT', idat]);
  chunks.push(['IEND', new Uint8Array(0)]);
  return pngOf(chunks);
}

// Encode one image: screen forms/filters at zlib level 6, then encode the
// winner's filtered data several ways and keep the smallest file.
function encodeImage(rgba, width, height) {
  const { best, second } = screen(rgba, width, height);
  const encodeData = (data, chunks) => {
    const cands = [];
    const add = (stream) => cands.push({ total: stream.length + chunks, stream });
    add(deflateSync(data, { level: 6, memLevel: 9 }));
    add(deflateSync(data, { level: 9, memLevel: 9 }));
    add(deflateSync(data, { level: 9, memLevel: 9, strategy: constants.Z_FILTERED }));
    add(zlibWrap(optDeflate(data), data));
    let b = cands[0];
    for (const c of cands) if (c.total < b.total) b = c;
    return b;
  };
  const chunksBest = chunkBytes(best.form);
  const b1 = encodeData(best.data, chunksBest);
  let png = pngOfForm(width, height, best.form, b1.stream);
  if (second !== null && second.form !== best.form && second.bytes <= best.bytes * 1.02) {
    const chunks2 = chunkBytes(second.form);
    const b2 = encodeData(second.data, chunks2);
    if (b2.total < b1.total) png = pngOfForm(width, height, second.form, b2.stream);
  }
  return png;
}

// ---------------------------------------------------------------- Main

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(I).sort()) {
  const image = decodePng(new Uint8Array(readFileSync(join(I, id, 'image.png'))));
  const png = encodeImage(image.rgba, image.width, image.height);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'image.png'), png);
}
