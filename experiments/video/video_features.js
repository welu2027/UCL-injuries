// Video experiment, step 2 — run the browser feature extractor (features.js) on every simulated sequence.
// Usage: node video_features.js   → writes data/video_features.csv
const fs = require("fs"), path = require("path");
const FEAT = require(path.join(__dirname, "features.js"));
const lines = fs.readFileSync(path.join(__dirname, "..", "..", "data", "video_sim.jsonl"), "utf8").split("\n").filter(Boolean);
const cols = ["layback", "elbow_flex_mer", "arm_slot_br", "trunk_fwd_br", "trunk_lat_br", "stride_rel", "fp_br_ms", "wrist_speed", "leg_len"];
const out = [["session_pitch", "user", "version", "ok", "fp_err_ms", "br_err_ms", "mer_err_ms", ...cols].join(",")];
let bad = 0;
for (const l of lines) {
  const r = JSON.parse(l);
  const res = FEAT.extract({ t: r.t, J: r.J }, "R", { mirrorCheck: r.version !== "clean" });
  if (!res.ok) { bad++; out.push([r.session_pitch, r.user, r.version, 0, "", "", "", ...cols.map(() => "")].join(",")); continue; }
  const e = res.events;
  out.push([r.session_pitch, r.user, r.version, 1, ((e.fp - r.events.fp) * 1000).toFixed(1), ((e.br - r.events.br) * 1000).toFixed(1),
    ((e.mer - r.events.mer) * 1000).toFixed(1), ...cols.map(c => res.features[c].toFixed(4))].join(","));
}
fs.writeFileSync(path.join(__dirname, "..", "..", "data", "video_features.csv"), out.join("\n"));
console.log(`features for ${lines.length} sequences, ${bad} failed`);
