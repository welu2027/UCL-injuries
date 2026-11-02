"""
Video experiment, step 3 — how much does a phone video add? Train the video model the website uses.

Target: peak elbow varus torque from our inverse dynamics (lab truth).
Inputs: velocity, mass, height (+ mechanics measured by public/features.js on simulated video).
Trained on the degraded 240 fps version (what an upload looks like), validated grouped by pitcher.

Outputs: experiments/video/video_model.json (weights for the browser) and a printed accuracy table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MECH = ["layback", "elbow_flex_mer", "arm_slot_br", "trunk_fwd_br", "trunk_lat_br", "stride_rel", "fp_br_ms"]
LABELS = {"layback": "Layback (max shoulder rotation)", "elbow_flex_mer": "Elbow bend at max layback",
          "arm_slot_br": "Arm slot at release", "trunk_fwd_br": "Trunk lean toward the target",
          "trunk_lat_br": "Trunk tilt to the glove side", "stride_rel": "Stride length (leg lengths)",
          "fp_br_ms": "Foot plant to release time"}
UNITS = {"layback": "°", "elbow_flex_mer": "°", "arm_slot_br": "°", "trunk_fwd_br": "°", "trunk_lat_br": "°",
         "stride_rel": "", "fp_br_ms": " ms"}
BASE = ["pitch_speed_mph", "mass_kg", "height_m"]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def cv(df, cols, target="varus_peak_Nm", model=ridge, reps=10):
    y, X, g = df[target].to_numpy(), df[cols], df.user.to_numpy()
    rng = np.random.default_rng(3)
    r2s, maes, P = [], [], np.zeros((reps, len(y)))
    for r in range(reps):
        ug = np.unique(g); perm = dict(zip(ug, rng.permutation(len(ug))))
        gg = np.array([perm[u] for u in g])
        for tr, te in GroupKFold(5).split(X, y, gg):
            m = model().fit(X.iloc[tr], y[tr]); P[r, te] = m.predict(X.iloc[te])
        r2s.append(r2_score(y, P[r])); maes.append(mean_absolute_error(y, P[r]))
    return float(np.mean(r2s)), float(np.mean(maes)), P.mean(0)


def main():
    phys = pd.read_csv(DATA / "physics_per_pitch.csv")
    vf = pd.read_csv(DATA / "video_features.csv")
    vf = vf[vf.ok == 1]
    keep = ["session_pitch", "varus_peak_Nm", "pitch_speed_mph", "mass_kg", "height_m"]
    D = {v: vf[vf.version == v].merge(phys[keep], on="session_pitch") for v in ("clean", "v240", "v60")}
    common = set.intersection(*[set(d.session_pitch) for d in D.values()])
    D = {k: d[d.session_pitch.isin(common)].reset_index(drop=True) for k, d in D.items()}

    # velocity from the video: wrist speed in leg-lengths per second, times a typical leg length for that height
    ratio = float((D["clean"].leg_len / D["clean"].height_m).median())
    for d in D.values():
        d["wrist_mps_est"] = d.wrist_speed / d.leg_len * ratio * d.height_m
    velo_fit = LinearRegression().fit(D["v240"][["wrist_mps_est"]], D["v240"].pitch_speed_mph)
    for d in D.values():
        d["velo_from_video"] = velo_fit.predict(d[["wrist_mps_est"]])
    v_r2 = r2_score(D["v240"].pitch_speed_mph, D["v240"].velo_from_video)
    v_mae = mean_absolute_error(D["v240"].pitch_speed_mph, D["v240"].velo_from_video)

    rows = []
    for name, d, cols in [
        ("Velocity + size (no video)", D["v240"], BASE),
        ("+ mechanics, perfect lab tracking", D["clean"], BASE + MECH),
        ("+ mechanics, 240 fps phone video", D["v240"], BASE + MECH),
        ("+ mechanics, 60 fps phone video", D["v60"], BASE + MECH),
    ]:
        r2, mae, _ = cv(d, cols)
        rows.append(dict(name=name, r2=r2, mae=mae))
    d = D["v240"].copy(); d["pitch_speed_mph"] = d.velo_from_video
    r2, mae, _ = cv(d, BASE + MECH)
    rows.append(dict(name="240 fps video, no radar (velocity from video)", r2=r2, mae=mae))
    for r in rows:
        print(f"{r['name']:48s} R² {r['r2']:.2f}   typical error {r['mae']:.1f} Nm")
    print(f"velocity from video: R² {v_r2:.2f}, typical error {v_mae:.1f} mph")

    # final model for the site: trained on all 240 fps simulated videos
    m = ridge().fit(D["v240"][BASE + MECH], D["v240"].varus_peak_Nm)
    sc, rr = m.named_steps["standardscaler"], m.named_steps["ridgecv"]
    pop = D["clean"][MECH]
    out = dict(
        features=BASE + MECH, mech=MECH, labels=LABELS, units=UNITS,
        mean=sc.mean_.round(5).tolist(), scale=sc.scale_.round(5).tolist(), coef=rr.coef_.round(5).tolist(),
        intercept=float(rr.intercept_), resid_mae=rows[2]["mae"],
        accuracy=rows, leg_ratio=ratio,
        velo=dict(intercept=float(velo_fit.intercept_), coef=float(velo_fit.coef_[0]), r2=v_r2, mae=v_mae),
        pop={f: dict(p10=float(pop[f].quantile(.1)), p50=float(pop[f].median()), p90=float(pop[f].quantile(.9)),
                     sorted=np.round(np.sort(pop[f].to_numpy()), 3).tolist()) for f in MECH},
    )
    (Path(__file__).parent / "video_model.json").write_text(json.dumps(out, separators=(",", ":")))
    print("coefficients (Nm per SD):", dict(zip(BASE + MECH, np.round(rr.coef_, 2))))


if __name__ == "__main__":
    main()
