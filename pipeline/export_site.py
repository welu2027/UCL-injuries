"""
Step 5 — export compact JSON for the static website (public/data/).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from physics import arm_inverse_dynamics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "public" / "data"
EXAMPLE_PAIR = (72, 581)   # ~88 mph, ~100 kg each; very different elbow torque

JOINTS = ["rear_ankle_jc", "rear_knee_jc", "rear_hip", "lead_ankle_jc", "lead_knee_jc", "lead_hip",
          "shoulder_jc", "elbow_jc", "wrist_jc", "hand_jc",
          "glove_shoulder_jc", "glove_elbow_jc", "glove_wrist_jc", "glove_hand_jc",
          "thorax_dist", "thorax_prox", "centerofmass"]


def r(x, n=3):
    return np.round(np.asarray(x, dtype=float), n).tolist()


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stress = json.loads((DATA / "stress_model.json").read_text())
    risk = json.loads((DATA / "risk_model.json").read_text())
    pt = pd.read_csv(DATA / "pitch_table.csv")
    phys = pd.read_csv(DATA / "physics_per_pitch.csv")
    pr = pd.read_csv(DATA / "pitcher_risk.csv")
    curves = np.load(DATA / "varus_curves.npz", allow_pickle=True)

    ok = phys[phys.fp_valid]
    validation = dict(
        r=float(np.corrcoef(ok.varus_peak_Nm, ok.varus_peak_lab_Nm)[0, 1]),
        bias=float((ok.varus_peak_Nm - ok.varus_peak_lab_Nm).mean()),
        sd_diff=float((ok.varus_peak_Nm - ok.varus_peak_lab_Nm).std()),
        mine_mean=float(ok.varus_peak_Nm.mean()), lab_mean=float(ok.varus_peak_lab_Nm.mean()),
        mean_13hz=float(ok.varus_peak_13Hz_Nm.mean()),
        pct_bwht=float(ok.varus_peak_pct_bwht.mean()),
        t_peak_ms=float(1000 * ok.t_varus_peak_rel_mer.median()),
        flex_at_peak=float(ok.elbow_flex_at_varus_peak.median()),
        rot_share=float(ok.varus_rotational_share.median()),
        elbow_force=float(ok.elbow_force_peak_N.median()),
        elbow_force_bw=float((ok.elbow_force_peak_N / (ok.mass_kg * 9.81)).median()),
        arm_ke=float(ok.arm_ke_br_J.median()),
        elbow_power_min=float(ok.elbow_power_min_W.median()),
        seq_proper=float(ok.sequence_proper.mean()),
        lag_torso_arm_ms=float(1000 * ok.lag_torso_arm.median()),
    )
    g = pt.groupby("user").varus_peak_Nm
    cv = (g.std() / g.mean())[g.count() >= 3]

    y = curves["y"]
    ids = list(curves["ids"])
    keep = [i for i, sp in enumerate(ids) if sp in set(ok.session_pitch)]
    yy = y[keep]
    curve = dict(t_ms=r(1000 * curves["t"], 1),
                 p10=r(np.nanpercentile(yy, 10, axis=0), 1), p25=r(np.nanpercentile(yy, 25, axis=0), 1),
                 p50=r(np.nanpercentile(yy, 50, axis=0), 1), p75=r(np.nanpercentile(yy, 75, axis=0), 1),
                 p90=r(np.nanpercentile(yy, 90, axis=0), 1))

    pitches = pt[["session_pitch", "user", "playing_level", "pitch_speed_mph", "mass_kg", "varus_peak_Nm",
                  "varus_peak_lab_Nm", "excess_torque_Nm", "pred_cv_ridge", "max_shoulder_external_rotation"]]
    pitches = dict(cols=list(pitches.columns), rows=[[v if not isinstance(v, float) else round(v, 2)
                                                      for v in row] for row in pitches.itertuples(index=False)])
    pit_cols = ["user", "n", "level", "hand", "velo", "mass", "height", "age", "torque", "torque_sd", "excess",
                "ucl_stress_pct", "pitches_to_failure_median", "season_risk", "relative_risk",
                "season_risk_if_typical_mech", "torque_per_mph", "layback", "elbow_flex_mer", "lat_tilt",
                "arm_slot", "seq"]
    pitchers = dict(cols=pit_cols, rows=[[None if (isinstance(v, float) and not np.isfinite(v))
                                          else (round(v, 4) if isinstance(v, float) else v) for v in row]
                                         for row in pr[pit_cols].itertuples(index=False)])

    site = dict(
        n_pitches=int(len(phys)), n_clean=int(len(ok)), n_pitchers=int(phys.user.nunique()),
        levels=phys.groupby("user").playing_level.first().value_counts().to_dict(),
        velo_range=[float(phys.pitch_speed_mph.min()), float(phys.pitch_speed_mph.max())],
        validation=validation,
        within_session_cv=dict(median=float(cv.median()), q25=float(cv.quantile(.25)), q75=float(cv.quantile(.75))),
        curve=curve, stress=stress, risk=risk, pitches=pitches, pitchers=pitchers,
    )
    (OUT / "site.json").write_text(json.dumps(clean(site), separators=(",", ":")))

    # --- animation data for the example pair -----------------------------------
    lm = pd.read_parquet(DATA / "landmarks.parquet")
    meta = pd.read_csv(DATA / "metadata.csv").set_index("session_pitch")
    poi = pd.read_csv(DATA / "poi_metrics.csv").set_index("session_pitch")
    phi = risk["constants"]["phi"]
    anim = []
    for user in EXAMPLE_PAIR:
        sub = pt[pt.user == user]
        sp = sub.iloc[(sub.varus_peak_Nm - sub.varus_peak_Nm.median()).abs().argsort().iloc[0]].session_pitch
        a = lm[lm.session_pitch == sp].sort_values("time")
        ev = a.iloc[0]
        t = a.time.to_numpy()
        lefty = poi.loc[sp, "p_throws"] == "L"
        P = {j: a[[f"{j}_x", f"{j}_y", f"{j}_z"]].to_numpy(float).copy() for j in JOINTS}
        if lefty:
            for v in P.values():
                v[:, 1] *= -1
        res = arm_inverse_dynamics(t, P["shoulder_jc"], P["elbow_jc"], P["wrist_jc"], P["hand_jc"],
                                   float(meta.loc[sp, "session_mass_kg"]), ev.BR_time)
        w = (t >= ev.pkh_time - 0.15) & (t <= ev.BR_time + 0.25)
        origin = P["rear_ankle_jc"][w][0].copy()
        origin[2] = 0.0
        frames = np.stack([P[j][w] - origin for j in JOINTS], axis=1)     # (F, J, 3)
        anim.append(dict(
            user=int(user), session_pitch=sp, lefty=bool(lefty), level=sub.playing_level.iloc[0],
            velo=float(meta.loc[sp, "pitch_speed_mph"]), mass=float(meta.loc[sp, "session_mass_kg"]),
            height=float(meta.loc[sp, "session_height_m"]),
            peak=float(sub.loc[sub.session_pitch == sp, "varus_peak_Nm"].iloc[0]),
            t=r(t[w] - ev.BR_time, 4), varus=r(res["varus"][w], 1), ucl=r(phi * res["varus"][w], 2),
            frames=r(frames, 3),
            events={k: float(ev[k] - ev.BR_time) for k in ("pkh_time", "fp_100_time", "MER_time", "BR_time")},
        ))
    (OUT / "anim.json").write_text(json.dumps(dict(joints=JOINTS, pitches=anim), separators=(",", ":")))
    for f in OUT.iterdir():
        print(f.name, f"{f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
