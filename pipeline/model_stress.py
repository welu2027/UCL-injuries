"""
Step 3 — which mechanics drive elbow stress?

Target: peak elbow varus torque (Nm) from our own inverse dynamics.
Inputs: kinematics, ground-reaction forces, body size, velocity. Anything computed from
arm joint moments (lab torques, energy-flow terms) is excluded so the model can't cheat.

Validation is grouped by pitcher (GroupKFold over athletes, repeated with shuffles), so a
pitcher's own pitches never appear in both train and test.

Outputs  data/stress_model.json   (metrics, importances, partial dependence, calculator weights)
         data/pitch_table.csv     (per-pitch predictions and "excess torque" residuals)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 7

LABELS = {
    "pitch_speed_mph": "Pitch velocity (mph)",
    "mass_kg": "Body mass (kg)",
    "height_m": "Height (m)",
    "max_shoulder_external_rotation": "Max layback / shoulder ER (deg)",
    "elbow_flexion_mer": "Elbow flexion at max layback (deg)",
    "max_elbow_flexion": "Peak elbow flexion (deg)",
    "elbow_flexion_fp": "Elbow flexion at foot plant (deg)",
    "shoulder_abduction_fp": "Shoulder abduction at foot plant (deg)",
    "shoulder_horizontal_abduction_fp": "Horizontal abduction at foot plant (deg)",
    "shoulder_external_rotation_fp": "Shoulder ER at foot plant (deg)",
    "max_shoulder_horizontal_abduction": "Peak horizontal abduction / scap load (deg)",
    "torso_anterior_tilt_fp": "Trunk forward tilt at foot plant (deg)",
    "torso_lateral_tilt_fp": "Trunk lateral tilt at foot plant (deg)",
    "torso_rotation_fp": "Trunk rotation at foot plant (deg)",
    "torso_anterior_tilt_mer": "Trunk forward tilt at max layback (deg)",
    "torso_lateral_tilt_mer": "Trunk lateral tilt at max layback (deg)",
    "torso_rotation_mer": "Trunk rotation at max layback (deg)",
    "torso_anterior_tilt_br": "Trunk forward tilt at release (deg)",
    "torso_lateral_tilt_br": "Trunk lateral tilt at release (deg)",
    "pelvis_rotation_fp": "Pelvis rotation at foot plant (deg)",
    "pelvis_anterior_tilt_fp": "Pelvis anterior tilt at foot plant (deg)",
    "rotation_hip_shoulder_separation_fp": "Hip–shoulder separation at foot plant (deg)",
    "max_rotation_hip_shoulder_separation": "Peak hip–shoulder separation (deg)",
    "torso_rotation_min": "Trunk counter-rotation (deg)",
    "lead_knee_extension_from_fp_to_br": "Lead knee extension FP→release (deg)",
    "lead_knee_extension_angular_velo_br": "Lead knee ext. velocity at release (deg/s)",
    "max_cog_velo_x": "Peak center-of-mass speed to plate (m/s)",
    "stride_length": "Stride length (% height)",
    "stride_angle": "Stride angle, + = closed (deg)",
    "arm_slot": "Arm slot (deg)",
    "glove_shoulder_abduction_mer": "Glove-arm abduction at max layback (deg)",
    "lead_grf_z_max": "Lead leg vertical GRF (N)",
    "lead_grf_x_max": "Lead leg braking GRF (N)",
    "rear_grf_x_max": "Drive leg propulsive GRF (N)",
    "peak_pelvis_velo": "Peak pelvis rotation speed (deg/s)",
    "peak_torso_velo": "Peak trunk rotation speed (deg/s)",
    "peak_shoulder_ir_velo": "Peak shoulder IR speed (deg/s)",
    "peak_elbow_ext_velo": "Peak elbow extension speed (deg/s)",
    "lag_pelvis_torso": "Pelvis→trunk peak timing gap (s)",
    "lag_torso_arm": "Trunk→arm peak timing gap (s)",
    "t_fp_to_br": "Foot plant → release time (s)",
    "forearm_len_m": "Forearm length (m)",
}
FEATURES = list(LABELS)

# Small, interpretable set used by the in-browser calculator (linear, so it ships as weights).
CALC_FEATURES = ["pitch_speed_mph", "mass_kg", "height_m", "max_shoulder_external_rotation",
                 "elbow_flexion_mer", "torso_lateral_tilt_mer", "arm_slot",
                 "max_shoulder_horizontal_abduction", "lag_torso_arm"]


def grouped_cv(model_fn, X, y, groups, n_repeats=10):
    preds = np.zeros((n_repeats, len(y)))
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    for r in range(n_repeats):
        perm = dict(zip(uniq, rng.permutation(len(uniq))))
        g = np.array([perm[x] for x in groups])
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = model_fn()
            m.fit(X.iloc[tr], y[tr])
            preds[r, te] = m.predict(X.iloc[te])
    r2 = np.array([r2_score(y, p) for p in preds])
    mae = np.array([mean_absolute_error(y, p) for p in preds])
    return preds.mean(0), dict(r2=float(r2.mean()), r2_sd=float(r2.std()), mae=float(mae.mean()))


def main():
    phys = pd.read_csv(DATA / "physics_per_pitch.csv")
    poi = pd.read_csv(DATA / "poi_metrics.csv")
    poi_cols = [c for c in FEATURES if c in poi.columns and c not in phys.columns]
    df = phys.merge(poi[["session_pitch"] + poi_cols], on="session_pitch")
    df = df[df.fp_valid].reset_index(drop=True)
    y = df.varus_peak_Nm.to_numpy()
    groups = df.user.to_numpy()
    X = df[FEATURES]

    ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-2, 3, 30)))
    gbm = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15,
                                                min_samples_leaf=12, l2_regularization=1.0, random_state=SEED)
    base = lambda: make_pipeline(StandardScaler(), LinearRegression())

    results = {}
    _, results["baseline_size_velo"] = grouped_cv(base, df[["pitch_speed_mph", "mass_kg", "height_m"]], y, groups)
    p_ridge, results["ridge_all"] = grouped_cv(ridge, X, y, groups)
    p_gbm, results["gbm_all"] = grouped_cv(gbm, X, y, groups)
    p_calc, results["calculator_linear"] = grouped_cv(ridge, df[CALC_FEATURES], y, groups)
    for k, v in results.items():
        print(f"{k:22s} R2 {v['r2']:.3f} ± {v['r2_sd']:.3f}   MAE {v['mae']:.1f} Nm")

    # --- permutation importance on held-out pitchers (GBM) -----------------------
    imp = np.zeros(len(FEATURES))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = gbm().fit(X.iloc[tr], y[tr])
        pi = permutation_importance(m, X.iloc[te], y[te], n_repeats=15, random_state=SEED, scoring="r2")
        imp += pi.importances_mean / 5
    order = np.argsort(-imp)
    importances = [dict(feature=FEATURES[i], label=LABELS[FEATURES[i]], drop_r2=float(imp[i])) for i in order]

    # --- "excess torque": torque left over after accounting for velocity and body size ----
    size_velo = LinearRegression().fit(df[["pitch_speed_mph", "mass_kg", "height_m"]], y)
    df["expected_from_size_velo"] = size_velo.predict(df[["pitch_speed_mph", "mass_kg", "height_m"]])
    df["excess_torque_Nm"] = y - df.expected_from_size_velo

    # which mechanics explain excess torque? pitcher-level means + bootstrap over pitchers
    pit = df.groupby("user")[FEATURES + ["excess_torque_Nm"]].mean()
    rng = np.random.default_rng(SEED)
    users = pit.index.to_numpy()
    drivers = []
    for f in FEATURES:
        if f in ("pitch_speed_mph", "mass_kg", "height_m"):
            continue
        sub = pit[[f, "excess_torque_Nm"]].dropna()
        r = sub.corr().iloc[0, 1]
        boots = []
        for _ in range(1000):
            s = sub.loc[rng.choice(sub.index, len(sub), replace=True)]
            boots.append(s.corr().iloc[0, 1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        slope = np.polyfit(sub[f], sub.excess_torque_Nm, 1)[0]
        drivers.append(dict(feature=f, label=LABELS[f], r=float(r), lo=float(lo), hi=float(hi),
                            slope_Nm_per_unit=float(slope), sd=float(sub[f].std())))
    drivers.sort(key=lambda d: -abs(d["r"]))

    # --- partial dependence (GBM fit on all data) for top features ----------------
    full = gbm().fit(X, y)
    pdp = []
    for d in importances[:6]:
        f = d["feature"]
        grid = np.linspace(*np.nanpercentile(X[f], [5, 95]), 25)
        vals = []
        for g in grid:
            Xg = X.copy()
            Xg[f] = g
            vals.append(float(full.predict(Xg).mean()))
        pdp.append(dict(feature=f, label=LABELS[f], x=grid.round(4).tolist(), y=np.round(vals, 2).tolist()))

    # --- calculator weights (ridge on small set, refit on all data) ---------------
    calc = ridge().fit(df[CALC_FEATURES], y)
    imputer, scaler, rr = calc.named_steps.values()
    calc_json = dict(features=CALC_FEATURES, labels=[LABELS[f] for f in CALC_FEATURES],
                     mean=scaler.mean_.round(5).tolist(), scale=scaler.scale_.round(5).tolist(),
                     coef=rr.coef_.round(5).tolist(), intercept=float(rr.intercept_),
                     defaults=df[CALC_FEATURES].median().round(3).tolist(),
                     p5=df[CALC_FEATURES].quantile(0.05).round(3).tolist(),
                     p95=df[CALC_FEATURES].quantile(0.95).round(3).tolist(),
                     resid_sd=float(np.std(y - p_calc)), alpha=float(rr.alpha_))

    df["pred_cv_gbm"] = p_gbm
    df["pred_cv_ridge"] = p_ridge
    keep = ["session_pitch", "session", "user", "p_throws", "playing_level", "age_yrs", "mass_kg", "height_m",
            "pitch_speed_mph", "varus_peak_Nm", "varus_peak_lab_Nm", "varus_peak_13Hz_Nm", "varus_peak_pct_bwht",
            "expected_from_size_velo", "excess_torque_Nm", "pred_cv_gbm", "pred_cv_ridge", "torque_per_mph",
            "t_varus_peak_rel_mer", "elbow_flex_at_varus_peak", "varus_rotational_share", "elbow_force_peak_N",
            "arm_ke_br_J", "elbow_power_min_W", "sequence_proper", "lag_pelvis_torso", "lag_torso_arm",
            "max_shoulder_external_rotation", "elbow_flexion_mer", "torso_lateral_tilt_mer", "arm_slot",
            "max_shoulder_horizontal_abduction", "peak_pelvis_velo", "peak_torso_velo", "peak_shoulder_ir_velo"]
    df[keep].to_csv(DATA / "pitch_table.csv", index=False)

    out = dict(n_pitches=int(len(df)), n_pitchers=int(df.user.nunique()), metrics=results,
               importances=importances, drivers=drivers, pdp=pdp, calculator=calc_json,
               size_velo_coef=dict(intercept=float(size_velo.intercept_),
                                   coef=dict(zip(["pitch_speed_mph", "mass_kg", "height_m"],
                                                 size_velo.coef_.round(4).tolist()))))
    (DATA / "stress_model.json").write_text(json.dumps(out, indent=1))
    print("top importances:", [(d["feature"], round(d["drop_r2"], 3)) for d in importances[:8]])
    print("top excess-torque drivers:", [(d["feature"], round(d["r"], 2), round(d["lo"], 2), round(d["hi"], 2)) for d in drivers[:8]])
    print("size+velo coef:", out["size_velo_coef"])


if __name__ == "__main__":
    main()
