"""
Video experiment, step 1 — turn the motion-capture pitches into what a phone + pose tracker would see.

For every lab pitch we write three versions of the skeleton, in the joint layout MediaPipe uses:
  clean  : lab joint centres at 360 Hz (no degradation)
  v240   : 240 fps slow-mo with ~10% dropped frames, tracking jitter, extra depth error along a
           random camera direction, extra blur noise on the wrist and elbow, ±10% scale error
  v60    : the same degradation at 60 fps (normal phone video)

Noise levels are assumptions about monocular pose tracking (a few cm), stated here so they can
be changed. Output: data/video_sim.jsonl, read by video_features.js (Node) and video_model.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RNG = np.random.default_rng(11)

MAP = {"r_sh": "shoulder_jc", "r_el": "elbow_jc", "r_wr": "wrist_jc", "l_sh": "glove_shoulder_jc",
       "l_el": "glove_elbow_jc", "l_wr": "glove_wrist_jc", "r_hip": "rear_hip", "l_hip": "lead_hip",
       "r_kn": "rear_knee_jc", "l_kn": "lead_knee_jc", "r_an": "rear_ankle_jc", "l_an": "lead_ankle_jc"}
NOISE = dict(smooth_sd=0.025, smooth_hz=4.0, white_sd=0.010, depth_sd=0.040, blur_sd=0.015, drop=0.10, scale_sd=0.05)


def smooth_noise(n, fs, sd, hz):
    b, a = butter(2, hz / (fs / 2))
    x = filtfilt(b, a, RNG.normal(0, 1, (n + 200, 3)), axis=0)[100:-100]
    return x / x.std(axis=0, keepdims=True) * sd


def degrade(t, J, fs):
    tt = np.arange(t[0], t[-1], 1 / fs)
    keep = RNG.random(len(tt)) > NOISE["drop"]
    keep[:3] = keep[-3:] = True
    tt = tt[keep]
    az = RNG.uniform(0, 2 * np.pi)
    cam = np.array([np.cos(az), np.sin(az), 0.0])           # depth direction of a random side-ish camera
    scale = 1 + RNG.normal(0, NOISE["scale_sd"])
    out = {}
    for k, X in J.items():
        Xi = np.stack([np.interp(tt, t, X[:, c]) for c in range(3)], 1)
        n = len(tt)
        e = smooth_noise(n, fs, NOISE["smooth_sd"], NOISE["smooth_hz"]) + RNG.normal(0, NOISE["white_sd"], (n, 3))
        e += np.outer(smooth_noise(n, fs, NOISE["depth_sd"], NOISE["smooth_hz"])[:, 0], cam)
        if k in ("r_wr", "r_el"):
            e += RNG.normal(0, NOISE["blur_sd"], (n, 3))
        out[k] = ((Xi - Xi.mean(0)) * scale + Xi.mean(0) + e)
    return tt, out


def main():
    lm = pd.read_parquet(DATA / "landmarks.parquet")
    phys = pd.read_csv(DATA / "physics_per_pitch.csv").set_index("session_pitch")
    lines = []
    for sp, a in lm.groupby("session_pitch", sort=False):
        if not phys.loc[sp, "fp_valid"]:
            continue
        a = a.sort_values("time")
        ev = a.iloc[0]
        w = (a.time >= ev.pkh_time - 0.3) & (a.time <= ev.BR_time + 0.3)
        a = a[w]
        t = a.time.to_numpy() - ev.BR_time
        J = {k: a[[f"{v}_x", f"{v}_y", f"{v}_z"]].to_numpy(float).copy() for k, v in MAP.items()}
        if phys.loc[sp, "p_throws"] == "L":                  # mirror lefties into a right-handed body
            for X in J.values():
                X[:, 1] *= -1
        tp = a[["thorax_prox_x", "thorax_prox_y", "thorax_prox_z"]].to_numpy(float).copy()
        td = a[["thorax_dist_x", "thorax_dist_y", "thorax_dist_z"]].to_numpy(float).copy()
        if phys.loc[sp, "p_throws"] == "L":
            tp[:, 1] *= -1; td[:, 1] *= -1
        up = (tp - td) / np.linalg.norm(tp - td, axis=1, keepdims=True)
        ant = np.cross(up, J["r_sh"] - J["l_sh"]); ant /= np.linalg.norm(ant, axis=1, keepdims=True)
        J["nose"] = tp + 0.24 * up + 0.10 * ant
        rec = dict(session_pitch=sp, user=int(phys.loc[sp, "user"]), events=dict(
            fp=float(ev.fp_100_time - ev.BR_time), mer=float(ev.MER_time - ev.BR_time), br=0.0))
        for name, fs in (("clean", None), ("v240", 240), ("v60", 60)):
            if fs is None:
                tt, JJ = t, J
            else:
                tt, JJ = degrade(t, J, fs)
            lines.append(json.dumps(dict(rec, version=name, t=np.round(tt, 5).tolist(),
                                         J={k: np.round(v, 4).tolist() for k, v in JJ.items()})))
    (DATA / "video_sim.jsonl").write_text("\n".join(lines))
    print(f"wrote {len(lines)} sequences ({len(lines)//3} pitches × clean/240/60)")


if __name__ == "__main__":
    main()
