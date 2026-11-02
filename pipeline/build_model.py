"""
Step 7 — "Build your pitcher": how elbow torque changes with four mechanics you can pose.

Model: ridge regression of peak elbow varus torque (our inverse dynamics) on velocity, body size and
four mechanics, validated grouped by pitcher. Mass and height are held at the base pitcher's values in
the browser, so only velocity and mechanics move the number.

Base pitch: a real right-handed delivery whose mechanics sit near the lab median and whose measured
torque is close to what the model predicts. The website edits this delivery to the mechanics you pick,
and shows torque = this pitch's real physics + the model's change for your edits.

Output: public/data/build.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physics import arm_inverse_dynamics
from export_site import JOINTS, r

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "public" / "data"

SIZE = ["mass_kg", "height_m"]
KNOBS = ["pitch_speed_mph", "max_shoulder_external_rotation", "elbow_flexion_mer", "torso_lateral_tilt_mer", "arm_slot"]
KEYS = ["velo", "layback", "flex", "tilt", "slot"]
LABELS = ["Fastball", "Layback", "Elbow bend", "Trunk tilt", "Arm slot"]
UNITS = [" mph", "°", "°", "°", "°"]
HINTS = ["", "how far the forearm lays back at max layback", "elbow angle at max layback",
         "lean toward the glove side at max layback", "forearm angle above horizontal at release"]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def grouped_cv(df, cols, reps=10):
    y, g = df.varus_peak_Nm.to_numpy(), df.user.to_numpy()
    rng = np.random.default_rng(3)
    r2s, maes = [], []
    for _ in range(reps):
        ug = np.unique(g)
        perm = dict(zip(ug, rng.permutation(len(ug))))
        gg = np.array([perm[u] for u in g])
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(df, y, gg):
            p[te] = ridge().fit(df[cols].iloc[tr], y[tr]).predict(df[cols].iloc[te])
        r2s.append(r2_score(y, p)); maes.append(mean_absolute_error(y, p))
    return float(np.mean(r2s)), float(np.mean(maes))


def raw_coef(df, cols):
    m = ridge().fit(df[cols], df.varus_peak_Nm)
    sc, rr = m.named_steps["standardscaler"], m.named_steps["ridgecv"]
    return rr.coef_ / sc.scale_, m


def main():
    pt = pd.read_csv(DATA / "pitch_table.csv")          # 397 clean pitches (valid foot plant)
    cols = KNOBS[:1] + SIZE + KNOBS[1:]
    r2, mae = grouped_cv(pt, cols)
    r2_base, _ = grouped_cv(pt, KNOBS[:1] + SIZE)
    coef, model = raw_coef(pt, cols)
    c = dict(zip(cols, coef))

    # 90% intervals by resampling pitchers
    rng = np.random.default_rng(0)
    users = pt.user.unique()
    boot = []
    for _ in range(300):
        pick = rng.choice(users, len(users))
        boot.append(raw_coef(pd.concat([pt[pt.user == u] for u in pick]), cols)[0])
    boot = np.array(boot)
    ci = {k: [float(np.percentile(boot[:, i], 5)), float(np.percentile(boot[:, i], 95))] for i, k in enumerate(cols)}

    # base pitch: right-handed, typical speed and size, mechanics near the median, small model residual
    pt["pred"] = model.predict(pt[cols])
    z = sum(((pt[k] - pt[k].median()) / pt[k].std()) ** 2 for k in KNOBS[1:])
    cand = pt[(pt.p_throws == "R") & pt.pitch_speed_mph.between(84, 89) & pt.mass_kg.between(84, 98)].copy()
    cand["score"] = z[cand.index] + ((cand.varus_peak_Nm - cand.pred) / 5) ** 2 + ((cand.pitch_speed_mph - 86.5) / 2) ** 2
    base = cand.sort_values("score").iloc[0]
    sp = base.session_pitch

    lm = pd.read_parquet(DATA / "landmarks.parquet")
    meta = pd.read_csv(DATA / "metadata.csv").set_index("session_pitch")
    a = lm[lm.session_pitch == sp].sort_values("time")
    ev = a.iloc[0]
    t = a.time.to_numpy()
    P = {j: a[[f"{j}_x", f"{j}_y", f"{j}_z"]].to_numpy(float).copy() for j in JOINTS}
    res = arm_inverse_dynamics(t, P["shoulder_jc"], P["elbow_jc"], P["wrist_jc"], P["hand_jc"],
                               float(meta.loc[sp, "session_mass_kg"]), ev.BR_time)
    w = (t >= ev.pkh_time - 0.15) & (t <= ev.BR_time + 0.25)
    origin = P["rear_ankle_jc"][w][0].copy()
    origin[2] = 0.0
    frames = np.stack([P[j][w] - origin for j in JOINTS], axis=1)

    knobs = []
    for k, key, lab, unit, hint in zip(KNOBS, KEYS, LABELS, UNITS, HINTS):
        lo, hi = (70.0, 100.0) if key == "velo" else (float(pt[k].quantile(0.02)), float(pt[k].quantile(0.98)))
        knobs.append(dict(key=key, feature=k, label=lab, unit=unit, hint=hint, coef=float(c[k]), ci=ci[k],
                          base=float(base[k]), min=round(lo, 1), max=round(hi, 1), median=float(pt[k].median())))
    out = dict(
        model=dict(r2=r2, r2_velo_size=r2_base, mae=mae, n_pitches=int(len(pt)), n_pitchers=int(pt.user.nunique()),
                   size_coef={k: float(c[k]) for k in SIZE}),
        knobs=knobs,
        base=dict(session_pitch=sp, user=int(base.user), level=base.playing_level, velo=float(base.pitch_speed_mph),
                  mass=float(base.mass_kg), height=float(base.height_m), peak=float(base.varus_peak_Nm),
                  pred=float(base.pred), joints=JOINTS, t=r(t[w] - ev.BR_time, 4), varus=r(res["varus"][w], 1),
                  frames=r(frames, 3),
                  events={k: float(ev[k] - ev.BR_time) for k in ("pkh_time", "fp_100_time", "MER_time", "BR_time")}),
    )
    (OUT / "build.json").write_text(json.dumps(out, separators=(",", ":")))

    print(f"mechanics model: R² {r2:.2f} (velocity + size alone {r2_base:.2f}), typical error {mae:.1f} Nm")
    for kb in knobs:
        print(f"  {kb['label']:11s} {kb['coef']:+6.2f} Nm per unit  90% CI [{kb['ci'][0]:+.2f}, {kb['ci'][1]:+.2f}]"
              f"  base {kb['base']:.1f}  range {kb['min']}–{kb['max']}")
    print(f"base pitch {sp} (user {int(base.user)}, {base.playing_level}): {base.pitch_speed_mph} mph, "
          f"{base.mass_kg:.0f} kg, torque {base.varus_peak_Nm:.1f} Nm (model {base.pred:.1f})")
    print(f"build.json {(OUT / 'build.json').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
