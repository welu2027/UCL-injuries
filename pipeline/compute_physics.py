"""
Step 2 — run the physics engine on every pitch and build a per-pitch feature table.

Outputs
  data/physics_per_pitch.csv   one row per pitch: elbow kinetics (mine + lab), energy, sequence timing
  data/varus_curves.npz        varus-torque time series aligned to max external rotation (for plots)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from physics import arm_inverse_dynamics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVENTS = ["pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"]
CURVE_T = np.round(np.arange(-0.150, 0.0701, 1 / 360.0), 5)  # s relative to MER


def xyz(df, name):
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=float).copy()


def peak_time(t, y, lo, hi):
    w = (t >= lo) & (t <= hi)
    if not w.any():
        return np.nan, np.nan
    i = np.argmax(np.where(w, y, -np.inf))
    return t[i], y[i]


def main():
    lm = pd.read_parquet(DATA / "landmarks.parquet")
    jv = pd.read_parquet(DATA / "joint_velos.parquet")
    meta = pd.read_csv(DATA / "metadata.csv")
    poi = pd.read_csv(DATA / "poi_metrics.csv")
    hand = poi.set_index("session_pitch")["p_throws"]
    meta = meta.set_index("session_pitch")

    rows, curves, curve_ids = [], [], []
    for sp, a in lm.groupby("session_pitch", sort=False):
        a = a.sort_values("time")
        m = meta.loc[sp]
        ev = a.iloc[0][EVENTS].astype(float)
        t = a["time"].to_numpy()
        pts = [xyz(a, k) for k in ("shoulder_jc", "elbow_jc", "wrist_jc", "hand_jc")]
        if hand[sp] == "L":                       # mirror lefties into a right-handed frame
            for p in pts:
                p[:, 1] *= -1.0
        mass, height = float(m.session_mass_kg), float(m.session_height_m)

        r = arm_inverse_dynamics(t, *pts, mass, ev.BR_time)
        r13 = arm_inverse_dynamics(t, *pts, mass, ev.BR_time, cutoff=13.4)  # classic ASMI-style cutoff

        fp, mer, br = ev.fp_100_time, ev.MER_time, ev.BR_time
        # foot-plant markers are occasionally missing or mis-detected; flag those pitches
        fp_valid = bool(np.isfinite(fp) and 0.08 <= br - fp <= 0.30 and mer < br)
        # peak varus occurs in late cocking, so search a window anchored on MER (robust to bad FP)
        win = (t >= mer - 0.08) & (t <= br)
        i = int(np.argmax(np.where(win, r["varus"], -np.inf)))
        i13 = int(np.argmax(np.where(win, r13["varus"], -np.inf)))
        ibr = int(np.argmin(np.abs(t - br)))
        wpost = (t >= mer - 0.08) & (t <= br + 0.06)

        # kinematic sequence (deg/s), searched from peak knee height to just after release
        v = jv[jv.session_pitch == sp].sort_values("time")
        tv = v["time"].to_numpy()
        seq = {}
        for seg, col in [("pelvis", "pelvis_velo_z"), ("torso", "torso_velo_z"),
                         ("elbow_ext", "elbow_velo_x"), ("shoulder_ir", "shoulder_velo_z")]:
            tp, vp = peak_time(tv, v[col].to_numpy(), ev.pkh_time, br + 0.08)
            seq[f"t_peak_{seg}"] = tp - fp if fp_valid else np.nan
            seq[f"peak_{seg}_velo"] = vp
        order = [seq[f"t_peak_{s}"] for s in ("pelvis", "torso", "elbow_ext", "shoulder_ir")]
        seq["sequence_proper"] = int(order[0] <= order[1] <= order[3]) if fp_valid else np.nan
        seq["lag_pelvis_torso"] = order[1] - order[0]
        seq["lag_torso_arm"] = order[3] - order[1]

        bwht = mass * 9.81 * height
        rows.append(dict(
            session_pitch=sp, session=int(m.session), user=int(m.user), p_throws=hand[sp],
            mass_kg=mass, height_m=height, age_yrs=m.age_yrs, playing_level=m.playing_level,
            pitch_speed_mph=m.pitch_speed_mph, fp_valid=fp_valid,
            varus_peak_Nm=r["varus"][i],
            varus_peak_13Hz_Nm=r13["varus"][i13],
            varus_peak_pct_bwht=100 * r["varus"][i] / bwht,
            t_varus_peak_rel_mer=t[i] - mer,
            elbow_flex_at_varus_peak=r["elbow_flex"][i],
            varus_rotational_share=r["varus_rot_part"][i] / r["varus"][i],
            elbow_force_peak_N=np.max(np.where(wpost, r["elbow_force_mag"], -np.inf)),
            arm_ke_peak_J=np.max(np.where(wpost, r["arm_ke"], -np.inf)),
            arm_ke_br_J=r["arm_ke"][ibr],
            elbow_power_peak_W=np.max(np.where(wpost, r["elbow_power"], -np.inf)),
            elbow_power_min_W=np.min(np.where(wpost, r["elbow_power"], np.inf)),
            hand_speed_br=r["hand_speed"][ibr],
            forearm_len_m=r["seg"]["L_f"], upper_arm_len_m=r["seg"]["L_u"],
            t_fp_to_br=(br - fp) if fp_valid else np.nan, t_mer_to_br=br - mer,
            **seq,
        ))
        curves.append(np.interp(CURVE_T, t - mer, r["varus"], left=np.nan, right=np.nan))
        curve_ids.append(sp)

    df = pd.DataFrame(rows)
    lab = poi.set_index("session_pitch")[["elbow_varus_moment", "shoulder_internal_rotation_moment"]]
    df = df.join(lab.rename(columns={"elbow_varus_moment": "varus_peak_lab_Nm",
                                     "shoulder_internal_rotation_moment": "shoulder_ir_moment_lab_Nm"}),
                 on="session_pitch")
    df["torque_per_mph"] = df.varus_peak_Nm / df.pitch_speed_mph
    df.to_csv(DATA / "physics_per_pitch.csv", index=False)
    np.savez_compressed(DATA / "varus_curves.npz", t=CURVE_T, y=np.array(curves), ids=np.array(curve_ids))

    ok = df[df.fp_valid]
    r = np.corrcoef(ok.varus_peak_Nm, ok.varus_peak_lab_Nm)[0, 1]
    bias = (ok.varus_peak_Nm - ok.varus_peak_lab_Nm).mean()
    print(f"{len(df)} pitches ({len(ok)} pass QC) | my peak varus {ok.varus_peak_Nm.mean():.1f} ± {ok.varus_peak_Nm.std():.1f} Nm "
          f"| lab {ok.varus_peak_lab_Nm.mean():.1f} | r = {r:.3f} | bias {bias:+.1f} Nm "
          f"| 13.4 Hz cutoff {ok.varus_peak_13Hz_Nm.mean():.1f} Nm")


if __name__ == "__main__":
    main()
