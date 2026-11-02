/*
 * build.js — edit a real motion-captured delivery to new mechanics ("Build your pitcher").
 *
 * Starting from one real pitch, four rigid rotations re-pose the body over time:
 *   trunk tilt  upper body about the hip centre, around the line to home plate
 *   arm slot    whole throwing arm about the shoulder, around the same line
 *   layback     forearm + hand about the upper-arm axis (shoulder external rotation)
 *   elbow bend  forearm + hand about the elbow's flexion axis
 * Each change fades in during the stride and out again, so the delivery stays continuous and bone
 * lengths never change. Layback and elbow bend hit their targets at max layback and are back to the
 * real motion by release; arm slot is solved so the forearm angle at release hits its target.
 *
 * Torque = this pitch's real physics + the lab model's change for the new mechanics (see build_model.py).
 * Shared by the browser (window.BUILD) and Node (pipeline/build_check.js).
 */
(function (root) {
  "use strict";
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const unit = (a) => { const n = norm(a) || 1e-9; return [a[0] / n, a[1] / n, a[2] / n]; };
  const D2R = Math.PI / 180, R2D = 180 / Math.PI;
  const ss = (x, a, b) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };

  // rotate point p about the line through c along unit axis k by angle a (Rodrigues)
  function rot(p, c, k, a) {
    if (!a) return p;
    const v = sub(p, c), ca = Math.cos(a), sa = Math.sin(a), kv = cross(k, v), kd = dot(k, v) * (1 - ca);
    return [c[0] + v[0] * ca + kv[0] * sa + k[0] * kd, c[1] + v[1] * ca + kv[1] * sa + k[1] * kd, c[2] + v[2] * ca + kv[2] * sa + k[2] * kd];
  }

  function make(base) {
    const J = base.joints, ix = (n) => J.indexOf(n), ev = base.events, t = base.t, F = t.length;
    const I = {}; ["rear_hip", "lead_hip", "shoulder_jc", "elbow_jc", "wrist_jc", "hand_jc", "glove_shoulder_jc", "thorax_prox", "thorax_dist"].forEach(n => { I[n] = ix(n); });
    const UPPER = ["shoulder_jc", "elbow_jc", "wrist_jc", "hand_jc", "glove_shoulder_jc", "glove_elbow_jc", "glove_wrist_jc", "glove_hand_jc", "thorax_dist", "thorax_prox"].map(ix);
    const ARM = ["elbow_jc", "wrist_jc", "hand_jc"].map(ix), FORE = ["wrist_jc", "hand_jc"].map(ix);
    const near = (tt) => { let b = 0; for (let i = 1; i < F; i++) if (Math.abs(t[i] - tt) < Math.abs(t[b] - tt)) b = i; return b; };
    const iMER = near(ev.MER_time), iBR = near(0), fp = ev.fp_100_time, mer = ev.MER_time;

    // the throwing arm's side decides which way is "toward the glove side": rotate about ±x (the line to the plate)
    const f0 = base.frames[iMER], gloveY = Math.sign(f0[I.glove_shoulder_jc][1] - f0[I.shoulder_jc][1]) || 1;
    const AX = [-gloveY, 0, 0];   // positive angle tilts the trunk toward the glove side and raises the arm

    // time weights (0..1) for each edit
    const W = t.map(tt => ({
      tilt: ss(tt, fp - 0.30, fp - 0.02) * (1 - ss(tt, 0.06, 0.25)),
      slot: ss(tt, fp - 0.20, mer - 0.01) * (1 - ss(tt, 0.03, 0.22)),
      arm: tt <= mer ? ss(tt, fp - 0.12, mer) : 1 - ss(tt, mer, -0.004),
    }));

    // layback angle: forearm direction around the upper-arm axis, measured from the trunk's front toward up
    function laybackOf(f) {
      const U = unit(sub(f[I.thorax_prox], f[I.thorax_dist])), right = gloveY > 0 ? sub(f[I.shoulder_jc], f[I.glove_shoulder_jc]) : sub(f[I.glove_shoulder_jc], f[I.shoulder_jc]);
      const A = unit(cross(U, right));
      const h = unit(sub(f[I.elbow_jc], f[I.shoulder_jc])), fa = sub(f[I.wrist_jc], f[I.elbow_jc]);
      const a0 = unit(sub(A, mul(h, dot(A, h)))), u0 = unit(sub(sub(U, mul(h, dot(U, h))), mul(a0, dot(U, a0))));
      let a = Math.atan2(dot(fa, u0), dot(fa, a0)) * R2D; if (a < -90) a += 360; return a;
    }
    const flexOf = (f) => 180 - Math.acos(Math.max(-1, Math.min(1, dot(unit(sub(f[I.shoulder_jc], f[I.elbow_jc])), unit(sub(f[I.wrist_jc], f[I.elbow_jc])))))) * R2D;
    const slotOf = (f) => { const fa = unit(sub(f[I.wrist_jc], f[I.elbow_jc])); return Math.asin(Math.max(-1, Math.min(1, fa[2]))) * R2D; };
    const tiltOf = (f) => { const u = sub(f[I.thorax_prox], f[I.thorax_dist]); return Math.atan2(gloveY * u[1], u[2]) * R2D; };

    // sign of a rotation about the upper-arm axis that increases layback
    let laySign = 1;
    { const f = f0.map(p => p.slice()), el = f[I.elbow_jc], h = unit(sub(el, f[I.shoulder_jc])), a0 = laybackOf(f);
      FORE.forEach(j => { f[j] = rot(f[j], el, h, 0.02); }); if (laybackOf(f) < a0) laySign = -1; }

    function frame(i, ang) {
      const src = base.frames[i], w = W[i], f = src.map(p => p.slice());
      const ht = w.tilt * ang.tilt * D2R, hs = w.slot * ang.slot * D2R, hl = w.arm * ang.lay * D2R * laySign, hf = w.arm * ang.flex * D2R;
      if (ht) { const c = mul(add(f[I.rear_hip], f[I.lead_hip]), 0.5); UPPER.forEach(j => { f[j] = rot(f[j], c, AX, ht); }); }
      if (hs) { const c = f[I.shoulder_jc]; ARM.forEach(j => { f[j] = rot(f[j], c, AX, hs); }); }
      if (hl) { const el = f[I.elbow_jc], h = unit(sub(el, f[I.shoulder_jc])); FORE.forEach(j => { f[j] = rot(f[j], el, h, hl); }); }
      if (hf) { const el = f[I.elbow_jc], h = unit(sub(el, f[I.shoulder_jc])), k = unit(cross(h, sub(f[I.wrist_jc], el))); FORE.forEach(j => { f[j] = rot(f[j], el, k, hf); }); }
      return f;
    }

    // angles for target mechanics (deltas from the real pitch). Arm slot is solved at release, so trunk tilt
    // doesn't move it; layback is then solved at max layback, so raising the arm doesn't move it either.
    const slot0 = slotOf(base.frames[iBR]), lay0 = laybackOf(base.frames[iMER]);
    function solve(ang, key, i, meas, want) {
      for (let k = 0; k < 5; k++) {
        const v = meas(frame(i, ang)), e = v - want; if (Math.abs(e) < 0.01) break;
        const g = (meas(frame(i, Object.assign({}, ang, { [key]: ang[key] + 0.5 }))) - v) / 0.5;
        ang[key] -= e / (Math.abs(g) > 0.05 ? g : 1);
      }
    }
    function angles(d) {
      const ang = { tilt: d.tilt || 0, slot: d.slot || 0, lay: d.lay || 0, flex: d.flex || 0 };
      solve(ang, "slot", iBR, slotOf, slot0 + (d.slot || 0));
      solve(ang, "lay", iMER, laybackOf, lay0 + (d.lay || 0));
      return ang;
    }
    function all(d) { const ang = angles(d); return base.frames.map((_, i) => frame(i, ang)); }

    return { frame, angles, all, W, iMER, iBR, laybackOf, flexOf, slotOf, tiltOf, laySign };
  }

  const api = { make };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BUILD = api;
})(typeof window !== "undefined" ? window : globalThis);
