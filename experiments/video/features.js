/*
 * features.js — throwing mechanics from a 3D skeleton track.
 *
 * The same code runs in two places:
 *   - the browser, on MediaPipe Pose world landmarks from an uploaded video
 *   - Node, on the 411 motion-captured lab pitches (clean, and degraded to look like phone video)
 * so the model trained on the lab is fed exactly what the website measures.
 *
 * Input: { t: [s], J: { name: [[x,y,z], ...] } } in metres with z up. Joint names are anatomical:
 *   nose, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_an, r_an
 */
(function (root) {
  "use strict";
  const FS = 240;                 // everything is resampled to a uniform 240 Hz grid
  const NAMES = ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"];

  // ---------- small vector helpers ----------
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const unit = (a) => { const n = norm(a) || 1e-9; return [a[0] / n, a[1] / n, a[2] / n]; };
  const mid = (a, b) => mul(add(a, b), 0.5);
  const deg = (r) => r * 180 / Math.PI;
  const angle = (a, b) => deg(Math.acos(Math.max(-1, Math.min(1, dot(unit(a), unit(b))))));
  const Z = [0, 0, 1];
  const horiz = (a) => [a[0], a[1], 0];

  // ---------- signal processing ----------
  function resample(t, X, t0, t1) {
    const n = Math.max(2, Math.floor((t1 - t0) * FS) + 1), out = new Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const ti = t0 + i / FS;
      while (j < t.length - 2 && t[j + 1] < ti) j++;
      const u = Math.min(1, Math.max(0, (ti - t[j]) / ((t[j + 1] - t[j]) || 1e-9)));
      out[i] = [0, 1, 2].map(c => X[j][c] + (X[j + 1][c] - X[j][c]) * u);
    }
    return out;
  }
  function butter2(fc) { // 2nd-order Butterworth low-pass (bilinear transform)
    const K = Math.tan(Math.PI * fc / FS), n = 1 / (1 + Math.SQRT2 * K + K * K);
    const b0 = K * K * n;
    return { b: [b0, 2 * b0, b0], a: [2 * (K * K - 1) * n, (1 - Math.SQRT2 * K + K * K) * n] };
  }
  function lfilter(f, x) {
    const y = new Array(x.length); let x1 = x[0], x2 = x[0], y1 = x[0], y2 = x[0];
    for (let i = 0; i < x.length; i++) {
      const v = f.b[0] * x[i] + f.b[1] * x1 + f.b[2] * x2 - f.a[0] * y1 - f.a[1] * y2;
      x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
    return y;
  }
  function filtfilt1(f, x) { // zero-lag, odd-reflection padded
    const p = Math.min(40, x.length - 1), pad = [];
    for (let i = p; i >= 1; i--) pad.push(2 * x[0] - x[i]);
    const ext = pad.concat(x);
    for (let i = 1; i <= p; i++) ext.push(2 * x[x.length - 1] - x[x.length - 1 - i]);
    const y = lfilter(f, lfilter(f, ext).reverse()).reverse();
    return y.slice(p, p + x.length);
  }
  const filt3 = (f, X) => { const c = [0, 1, 2].map(k => filtfilt1(f, X.map(v => v[k]))); return X.map((_, i) => [c[0][i], c[1][i], c[2][i]]); };
  const speed = (X) => X.map((_, i) => { const a = X[Math.max(0, i - 1)], b = X[Math.min(X.length - 1, i + 1)], dt = (Math.min(X.length - 1, i + 1) - Math.max(0, i - 1)) / FS; return norm(sub(b, a)) / dt; });
  const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

  /** MediaPipe Pose output (per frame: worldLandmarks array of 33) → anatomical joints, z up. */
  function fromMediaPipe(frames) {
    const IDX = { nose: 0, l_sh: 11, r_sh: 12, l_el: 13, r_el: 14, l_wr: 15, r_wr: 16, l_hip: 23, r_hip: 24, l_kn: 25, r_kn: 26, l_an: 27, r_an: 28 };
    const ok = frames.filter(f => f.wl && f.wl.length >= 29);
    const J = {}; NAMES.forEach(n => { J[n] = ok.map(f => { const p = f.wl[IDX[n]]; return [p[0], p[2], -p[1]]; }); }); // (x, y down, z) → (x, z, -y): proper rotation, z up
    return { t: ok.map(f => f.t), J, vis: ok.map(f => f.wl[IDX.r_wr][3] ?? 1) };
  }

  /**
   * Extract events and mechanics. hand = "R" or "L".
   * opts.fc: low-pass cutoff (Hz). Returns { ok, events, features, series } or { ok:false, reason }.
   */
  function extract(seq, hand, opts) {
    opts = Object.assign({ fc: 14 }, opts || {});
    const t = seq.t;
    if (!t || t.length < 40) return { ok: false, reason: "Not enough frames with a detected body." };
    const t0 = t[0], t1 = t[t.length - 1];
    const f = butter2(opts.fc);
    const P = {};
    NAMES.forEach(n => { if (seq.J[n]) P[n] = filt3(f, resample(t, seq.J[n], t0, t1)); });
    const N = P.r_sh.length, T = (i) => t0 + i / FS;

    // mirror check (video only): with a proper frame, (R−L shoulder) × up points backwards, away from the nose
    let mirrored = false;
    if (P.nose && opts.mirrorCheck !== false) {
      let s = 0;
      for (let i = 0; i < Math.min(N, Math.floor(N * 0.3)); i++) {
        const U = sub(mid(P.l_sh[i], P.r_sh[i]), mid(P.l_hip[i], P.r_hip[i]));
        const back = cross(sub(P.r_sh[i], P.l_sh[i]), U);
        s += dot(back, sub(P.nose[i], mid(P.l_sh[i], P.r_sh[i])));
      }
      if (s > 0) { mirrored = true; NAMES.forEach(n => { if (P[n]) P[n] = P[n].map(v => [v[0], -v[1], v[2]]); }); }
    }

    const th = hand === "L" ? "l" : "r", gl = hand === "L" ? "r" : "l";
    const SH = P[th + "_sh"], EL = P[th + "_el"], WR = P[th + "_wr"], GSH = P[gl + "_sh"];
    const LEAD_AN = P[gl + "_an"], REAR_AN = P[th + "_an"], LEAD_HIP = P[gl + "_hip"], LEAD_KN = P[gl + "_kn"];
    const MH = P.l_hip.map((v, i) => mid(v, P.r_hip[i])), MS = P.l_sh.map((v, i) => mid(v, P.r_sh[i]));

    // release ≈ peak throwing-wrist speed
    const ws = speed(WR), edge = Math.floor(N * 0.04);
    let br = edge; for (let i = edge; i < N - edge; i++) if (ws[i] > ws[br]) br = i;
    // foot plant: after the lead ankle's fastest moment in the stride, the first frame it has nearly stopped
    const as = speed(LEAD_AN), w0 = Math.max(0, br - Math.round(0.7 * FS));
    let amax = w0; for (let i = w0; i < br; i++) if (as[i] > as[amax]) amax = i;
    let fp = amax; while (fp < br - 1 && as[fp] > 0.2 * as[amax]) fp++;
    if (br - fp < Math.round(0.03 * FS) || br - fp > Math.round(0.4 * FS)) {
      return { ok: false, reason: "Couldn't find a clear stride and release. Film the whole delivery, side-on, one throw per clip." };
    }

    // trunk frame and layback (shoulder external rotation proxy)
    const lay = new Array(N).fill(NaN), flex = new Array(N).fill(NaN);
    for (let i = 0; i < N; i++) {
      const U = unit(sub(MS[i], MH[i])), A = unit(cross(U, sub(P.r_sh[i], P.l_sh[i])));   // anterior = up × (R−L)
      const h = unit(sub(EL[i], SH[i])), fa = sub(WR[i], EL[i]);
      flex[i] = 180 - angle(sub(SH[i], EL[i]), fa);
      const a0 = unit(sub(A, mul(h, dot(A, h)))), u0 = unit(sub(sub(U, mul(h, dot(U, h))), mul(a0, dot(U, a0))));
      let a = deg(Math.atan2(dot(fa, u0), dot(fa, a0)));
      if (a < -90) a += 360;                    // layback lives in [-90, 270): forward → up → back, no wrap-around
      lay[i] = a;
    }
    // max layback: largest rotation while the elbow is clearly bent (the angle is undefined with a straight arm)
    let mer = -1;
    for (let i = fp; i <= br; i++) if (flex[i] > 50 && (mer < 0 || lay[i] > lay[mer])) mer = i;
    if (mer < 0) mer = Math.round((fp + br) / 2);

    // directions: stride direction (horizontal), and lateral axis pointing to the glove side
    const s0 = Math.max(0, fp - Math.round(0.5 * FS));
    const F = unit(horiz(sub(LEAD_AN[fp], LEAD_AN[s0])));
    let Lh = unit(cross(Z, F));
    if (dot(sub(WR[br], MH[br]), Lh) > 0) Lh = mul(Lh, -1);                              // wrist is on the arm side at release
    const Ubr = unit(sub(MS[br], MH[br])), fbr = unit(sub(WR[br], EL[br]));
    const legLen = median(LEAD_HIP.map((v, i) => norm(sub(v, LEAD_KN[i])) + norm(sub(LEAD_KN[i], LEAD_AN[i]))));

    const features = {
      layback: lay[mer] - 0,                                                            // deg
      elbow_flex_mer: flex[mer],                                                        // deg
      arm_slot_br: deg(Math.asin(Math.max(-1, Math.min(1, fbr[2])))),                   // forearm angle above horizontal at release
      trunk_fwd_br: deg(Math.atan2(dot(Ubr, F), Ubr[2])),                               // + = leaning toward the target
      trunk_lat_br: deg(Math.atan2(dot(Ubr, Lh), Ubr[2])),                              // + = leaning toward the glove side
      stride_rel: norm(horiz(sub(LEAD_AN[fp], REAR_AN[fp]))) / legLen,                  // stride / leg length
      fp_br_ms: (br - fp) * 1000 / FS,
      wrist_speed: ws[br],                                                              // m/s in the track's own scale
      leg_len: legLen,
    };
    return {
      ok: true, mirrored,
      events: { fp: T(fp), mer: T(mer), br: T(br) },
      features,
      series: { t0, fs: FS, n: N, wristSpeed: ws, layback: lay },
    };
  }

  const api = { extract, fromMediaPipe, NAMES, FS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FEAT = api;
})(typeof window !== "undefined" ? window : globalThis);
