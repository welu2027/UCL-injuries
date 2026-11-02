// Step 7b — check that "Build your pitcher" edits do what the sliders say.
// For each mechanic at its lab minimum and maximum (and a few combinations), re-pose the real delivery with
// public/build.js, then measure the edited motion: the targeted angle must move by the requested amount,
// the other angles must not, bones must keep their length, and the motion must stay smooth.
// Usage: node build_check.js
const fs = require("fs"), path = require("path");
const BUILD = require(path.join(__dirname, "..", "public", "build.js"));
const B = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "public", "data", "build.json"), "utf8"));
const base = B.base, M = BUILD.make(base), J = base.joints, ix = (n) => J.indexOf(n);
const K = Object.fromEntries(B.knobs.map(k => [k.key, k]));
const BONES = [["shoulder_jc", "elbow_jc"], ["elbow_jc", "wrist_jc"], ["wrist_jc", "hand_jc"], ["shoulder_jc", "glove_shoulder_jc"], ["thorax_dist", "thorax_prox"], ["glove_shoulder_jc", "glove_elbow_jc"]].map(([a, b]) => [ix(a), ix(b)]);
const len = (f, [a, b]) => Math.hypot(f[a][0] - f[b][0], f[a][1] - f[b][1], f[a][2] - f[b][2]);
const fpI = base.t.findIndex(x => x >= base.events.fp_100_time);

function measure(frames) {
  let maxLay = -1e9, iMax = 0;
  for (let i = fpI; i <= M.iBR; i++) { const a = M.laybackOf(frames[i]); if (M.flexOf(frames[i]) > 50 && a > maxLay) { maxLay = a; iMax = i; } }
  const hand = ix("hand_jc"), dt = base.t[1] - base.t[0];
  let peakAcc = 0, minZ = 9;
  for (let i = 1; i < frames.length - 1; i++) {
    const a = [0, 1, 2].map(c => (frames[i + 1][hand][c] - 2 * frames[i][hand][c] + frames[i - 1][hand][c]) / (dt * dt));
    if (base.t[i] > base.events.fp_100_time && base.t[i] < 0.05) peakAcc = Math.max(peakAcc, Math.hypot(...a));
    minZ = Math.min(minZ, frames[i][hand][2], frames[i][ix("wrist_jc")][2]);
  }
  return { lay: M.laybackOf(frames[M.iMER]), maxLay, merShiftMs: (base.t[iMax] - base.t[M.iMER]) * 1000, flex: M.flexOf(frames[M.iMER]),
    tilt: M.tiltOf(frames[M.iMER]), slot: M.slotOf(frames[M.iBR]), peakAcc, minZ };
}
const m0 = measure(base.frames);
const cases = [];
for (const k of ["lay", "flex", "tilt", "slot"]) {
  const kb = K[k === "lay" ? "layback" : k];
  for (const v of [kb.min, kb.max]) cases.push({ name: `${kb.label} ${v.toFixed(0)}`, d: { [k]: v - kb.base } });
}
cases.push({ name: "all at lab max", d: { lay: K.layback.max - K.layback.base, flex: K.flex.max - K.flex.base, tilt: K.tilt.max - K.tilt.base, slot: K.slot.max - K.slot.base } });
cases.push({ name: "all at lab min", d: { lay: K.layback.min - K.layback.base, flex: K.flex.min - K.flex.base, tilt: K.tilt.min - K.tilt.base, slot: K.slot.min - K.slot.base } });
cases.push({ name: "tilt max, slot min", d: { tilt: K.tilt.max - K.tilt.base, slot: K.slot.min - K.slot.base } });

const pad = (s, n) => String(s).padEnd(n);
const f1 = (x) => (x >= 0 ? "+" : "") + x.toFixed(1);
console.log(pad("case", 22), "Δlayback  Δmax-layback  Δelbow  Δtilt  Δslot   (asked)        max bone Δ  hand accel  MER shift");
let worst = 0;
for (const c of cases) {
  const fr = M.all(c.d), m = measure(fr);
  let bone = 0; fr.forEach((f, i) => BONES.forEach(b => { bone = Math.max(bone, Math.abs(len(f, b) - len(base.frames[i], b))); }));
  const got = { lay: m.lay - m0.lay, flex: m.flex - m0.flex, tilt: m.tilt - m0.tilt, slot: m.slot - m0.slot };
  for (const k of ["lay", "flex", "tilt", "slot"]) worst = Math.max(worst, Math.abs(got[k] - (c.d[k] || 0)));
  console.log(pad(c.name, 22), pad(f1(got.lay), 9), pad(f1(m.maxLay - m0.maxLay), 13), pad(f1(got.flex), 7), pad(f1(got.tilt), 6), pad(f1(got.slot), 7),
    pad(`(${["lay", "flex", "tilt", "slot"].map(k => f1(c.d[k] || 0)).join(" ")})`, 26), pad((bone * 1000).toFixed(2) + " mm", 11),
    pad((m.peakAcc / m0.peakAcc).toFixed(2) + "×", 11), m.merShiftMs.toFixed(0) + " ms");
}
console.log(`largest miss between asked and measured angle: ${worst.toFixed(2)}°`);
