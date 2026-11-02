"""
Step 4 — UCL injury risk as a tissue fatigue-failure process.

Idea: the UCL doesn't snap from one pitch; it accumulates micro-damage. Collagenous tissue
follows a fatigue (S-N) law — the higher the load as a fraction of its strength, the fewer
cycles it survives — with no endurance limit. So:

  1. Load per pitch   tau_UCL = phi * tau_varus           (phi = share the UCL carries after
                                                           the flexor-pronator muscles and bone)
  2. Stress ratio     S = 100 * tau_UCL / tau_fail         (% of the ligament's strength)
  3. Cycles to fail   log10 N_f = (101.3 - S) / 14.8       (human tendon S-N, Schechtman & Bader 1997)
  4. Damage           D = sum over pitches of 1 / N_f      (Palmgren–Miner rule)
  5. Injury           when D >= 1 within a season; ligament strength varies between people
                      (log-normal), so P(injury) = P(strength below the level where D hits 1).
  5b. Body size      Bigger athletes have bigger ligaments: strength scales with (mass / ref mass)^(2/3),
                      the usual allometric rule for tissue cross-sectional area. Equivalent to using a
                      size-adjusted torque  tau / (mass / ref mass)^(2/3).
  6. Calibration      phi is the one unknown. It is solved so a reference MLB pitcher at a
                      reference workload has the observed league injury rate per season.

Pitch-to-pitch torque variability is included: with a log-linear S-N curve, variable loading
does more damage than steady loading at the same average (convexity), which is the
"bad luck" mechanism discussed by van Trigt / Veeger et al. (2020).

Everything here is deliberately simple and transparent; the website re-implements the same
equations in JavaScript.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SN_A, SN_B = 101.3, 14.8           # S = A - B log10(N), S in % of ultimate (Schechtman & Bader 1997)
TAU_FAIL_MED = 34.0                 # Nm, intact UCL ultimate valgus torque (cadaver, ~30–34 Nm reported)
STRENGTH_SIGMA = 0.20               # log-sd of ligament strength between people (assumption)
TORQUE_CV = 0.08                    # pitch-to-pitch torque variability in games (assumption; lab session CV ≈ 2–3%)
P_REF = 0.05                        # per pitcher-season UCL injury rate used for calibration (order of magnitude)
REF_VELO = 94.8                     # mph, 2026 MLB average four-seam velocity
REF_WORKLOAD = 2500                 # max-effort pitches per season (≈ 30 starts × 83 pitches)
SIZE_EXP = 2.0 / 3.0                # ligament strength scales with body mass^(2/3) (cross-section allometry)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)   # Gauss–Hermite (probabilists') nodes
GH_W = GH_W / GH_W.sum()


def damage_per_pitch(S):
    """Fraction of fatigue life consumed by one cycle at stress ratio S (% of ultimate)."""
    return 10.0 ** ((np.asarray(S) - SN_A) / SN_B)


def log_season_damage(k, n_pitches, cv=TORQUE_CV, n_low=0, low_intensity=0.7):
    """
    Natural log of expected Miner damage for one season, where k = 100 * phi * tau / tau_fail is the
    stress ratio (% of ultimate) of an average max-effort pitch. Torque varies pitch to pitch as a
    log-normal with the given CV (mean preserved). Computed in log space to avoid overflow.
    """
    s_ln = np.sqrt(np.log(1 + cv ** 2))
    mult = np.exp(s_ln * GH_X - 0.5 * s_ln ** 2)
    ln10 = np.log(10.0)
    lw = np.log(GH_W)
    l_hi = np.log(n_pitches) + logsumexp(lw + ln10 * (k * mult - SN_A) / SN_B)
    if n_low <= 0:
        return l_hi
    l_lo = np.log(n_low) + logsumexp(lw + ln10 * (k * low_intensity * mult - SN_A) / SN_B)
    return np.logaddexp(l_hi, l_lo)


def critical_ratio(n_pitches, **kw):
    """Stress ratio k* (% of a ligament's strength) at which the season's damage reaches exactly 1.
    Depends only on workload and variability — not on torque, phi or strength."""
    return brentq(lambda k: log_season_damage(k, n_pitches, **kw), 1e-6, 400.0)


def season_risk(tau_mean, n_pitches, phi, tau_fail_med=TAU_FAIL_MED, sigma=STRENGTH_SIGMA, **kw):
    """P(UCL fatigue failure within the season) across a log-normal population of ligament strengths.
    A ligament fails if its strength is below tau* = 100*phi*tau/k*."""
    tstar = 100.0 * phi * tau_mean / critical_ratio(n_pitches, **kw)
    return float(norm.cdf((np.log(tstar) - np.log(tau_fail_med)) / sigma))


def calibrate_phi(tau_ref, n_ref=REF_WORKLOAD, p_ref=P_REF, **kw):
    return brentq(lambda ph: season_risk(tau_ref, n_ref, ph, **kw) - p_ref, 1e-4, 2.0)


def main():
    stress = json.loads((DATA / "stress_model.json").read_text())
    pt = pd.read_csv(DATA / "pitch_table.csv")
    c = stress["size_velo_coef"]

    pro = pt[pt.playing_level == "milb"]
    ref_mass, ref_height = float(pro.mass_kg.median()), float(pro.height_m.median())
    tau_ref = c["intercept"] + c["coef"]["pitch_speed_mph"] * REF_VELO + c["coef"]["mass_kg"] * ref_mass \
        + c["coef"]["height_m"] * ref_height
    phi = calibrate_phi(tau_ref)
    S_ref = 100 * phi * tau_ref / TAU_FAIL_MED
    print(f"reference MLB arm: {REF_VELO} mph, {ref_mass:.0f} kg, {ref_height:.2f} m -> {tau_ref:.1f} Nm varus")
    print(f"calibrated UCL load share phi = {phi:.3f}  (UCL carries {phi*tau_ref:.1f} Nm, "
          f"S = {S_ref:.1f}% of median strength, N_f = {10**((SN_A-S_ref)/SN_B):,.0f} pitches)")

    # --- per-pitcher risk at a standardised workload ------------------------------
    pit = pt.groupby("user").agg(
        n=("varus_peak_Nm", "size"), level=("playing_level", "first"), hand=("p_throws", "first"),
        velo=("pitch_speed_mph", "mean"), mass=("mass_kg", "mean"), height=("height_m", "mean"),
        age=("age_yrs", "mean"), torque=("varus_peak_Nm", "mean"), torque_sd=("varus_peak_Nm", "std"),
        excess=("excess_torque_Nm", "mean"), layback=("max_shoulder_external_rotation", "mean"),
        elbow_flex_mer=("elbow_flexion_mer", "mean"), lat_tilt=("torso_lateral_tilt_mer", "mean"),
        arm_slot=("arm_slot", "mean"), seq=("sequence_proper", "mean")).reset_index()
    pit["size_scale"] = (pit.mass / ref_mass) ** SIZE_EXP
    pit["torque_size_adj"] = pit.torque / pit.size_scale
    pit["ucl_stress_pct"] = 100 * phi * pit.torque_size_adj / TAU_FAIL_MED
    pit["pitches_to_failure_median"] = 10 ** ((SN_A - pit.ucl_stress_pct) / SN_B)
    pit["season_risk"] = [season_risk(t, REF_WORKLOAD, phi) for t in pit.torque_size_adj]
    med_risk = season_risk(pit.torque_size_adj.median(), REF_WORKLOAD, phi)
    pit["relative_risk"] = pit.season_risk / med_risk
    # risk if the same pitcher threw with zero excess torque (mechanics matched to size+velo norm)
    pit["season_risk_if_typical_mech"] = [season_risk(max(t - e, 1) / sc, REF_WORKLOAD, phi)
                                          for t, e, sc in zip(pit.torque, pit.excess, pit.size_scale)]
    pit["torque_per_mph"] = pit.torque / pit.velo
    pit = pit.sort_values("season_risk", ascending=False)
    pit.round(5).to_csv(DATA / "pitcher_risk.csv", index=False)

    # --- sensitivity / trade-off surfaces -----------------------------------------
    torques = np.linspace(70, 170, 41)
    workloads = np.linspace(250, 4000, 31)
    grid = [[season_risk(t, w, phi) for w in workloads] for t in torques]
    curve_torque = [season_risk(t, REF_WORKLOAD, phi) for t in torques]
    curve_cv = {f"{cv:.2f}": [season_risk(t, REF_WORKLOAD, phi, cv=cv) for t in torques]
                for cv in (0.0, 0.05, 0.10, 0.15)}

    # equivalence: how much extra workload equals +10% torque?
    base = season_risk(tau_ref, REF_WORKLOAD, phi)
    up10 = season_risk(1.1 * tau_ref, REF_WORKLOAD, phi)
    w_equiv = brentq(lambda w: season_risk(tau_ref, w, phi) - up10, REF_WORKLOAD, 5e6)
    cv_effect = season_risk(tau_ref, REF_WORKLOAD, phi, cv=0.15) / season_risk(tau_ref, REF_WORKLOAD, phi, cv=0.0)

    # robustness: does the "+10% torque" effect depend on the unverifiable assumptions?
    robust = []
    for sig in (0.15, 0.20, 0.30):
        for pref in (0.02, 0.05, 0.10):
            ph = calibrate_phi(tau_ref, p_ref=pref, sigma=sig)
            b = season_risk(tau_ref, REF_WORKLOAD, ph, sigma=sig)
            u = season_risk(1.1 * tau_ref, REF_WORKLOAD, ph, sigma=sig)
            try:
                w2 = brentq(lambda w: season_risk(tau_ref, w, ph, sigma=sig) - u, REF_WORKLOAD, 5e6)
            except ValueError:            # risk level not reachable by volume alone
                w2 = np.nan
            robust.append(dict(sigma=sig, p_ref=pref, phi=round(ph, 4), rr_10pct=round(u / b, 2),
                               workload_equiv_x=round(w2 / REF_WORKLOAD, 2)))

    out = dict(
        constants=dict(SN_A=SN_A, SN_B=SN_B, TAU_FAIL_MED=TAU_FAIL_MED, STRENGTH_SIGMA=STRENGTH_SIGMA,
                       TORQUE_CV=TORQUE_CV, P_REF=P_REF, REF_VELO=REF_VELO, REF_WORKLOAD=REF_WORKLOAD,
                       REF_MASS=ref_mass, REF_HEIGHT=ref_height, SIZE_EXP=SIZE_EXP, phi=phi, tau_ref=tau_ref, S_ref=S_ref,
                       median_pitcher_torque=float(pit.torque_size_adj.median()), median_pitcher_risk=med_risk),
        headline=dict(rr_plus10_torque=up10 / base, workload_equiv_multiplier=w_equiv / REF_WORKLOAD,
                      cv15_vs_steady=cv_effect, base_risk=base),
        robustness=robust,
        surface=dict(torques=torques.round(2).tolist(), workloads=workloads.round(0).tolist(),
                     risk=np.round(grid, 5).tolist()),
        curve_torque=np.round(curve_torque, 5).tolist(),
        curve_cv={k: np.round(v, 5).tolist() for k, v in curve_cv.items()},
    )
    (DATA / "risk_model.json").write_text(json.dumps(out, indent=1))
    print(f"+10% torque -> x{up10/base:.2f} season risk; equivalent to x{w_equiv/REF_WORKLOAD:.2f} workload; "
          f"15% CV vs steady -> x{cv_effect:.2f}")
    print("robustness:", [(r['sigma'], r['p_ref'], r['rr_10pct'], r['workload_equiv_x']) for r in robust])
    print(pit[["user", "level", "velo", "torque", "excess", "ucl_stress_pct", "season_risk", "relative_risk"]]
          .head(8).round(3).to_string(index=False))
    print("risk range:", pit.season_risk.min().round(4), pit.season_risk.median().round(4), pit.season_risk.max().round(4))


if __name__ == "__main__":
    main()
