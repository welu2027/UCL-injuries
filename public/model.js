/*
 * model.js — the UCL fatigue-failure model and the elbow-torque estimator, in plain JavaScript.
 * Mirrors pipeline/risk_model.py and the ridge model in pipeline/model_stress.py so the
 * calculator on the page runs the exact same equations as the Python pipeline.
 */
(function (root) {
  "use strict";
  const LN10 = Math.log(10);

  // standard-normal quadrature grid for torque variability (z in [-7, 7])
  const Z = [], ZW = [];
  (function () {
    const n = 281, lo = -7, hi = 7, h = (hi - lo) / (n - 1);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const z = lo + i * h, w = Math.exp(-0.5 * z * z);
      Z.push(z); ZW.push(w); s += w;
    }
    for (let i = 0; i < n; i++) ZW[i] /= s;
  })();

  function logSumExp(arr) {
    let m = -Infinity;
    for (const v of arr) if (v > m) m = v;
    let s = 0;
    for (const v of arr) s += Math.exp(v - m);
    return m + Math.log(s);
  }

  // ln of expected Miner damage over a season; k = stress ratio (% of strength) of an average max-effort pitch
  function logSeasonDamage(k, nPitches, c, opts) {
    const cv = opts.cv, sLn = Math.sqrt(Math.log(1 + cv * cv));
    const terms = (scale) => Z.map((z, i) =>
      Math.log(ZW[i]) + LN10 * (k * scale * Math.exp(sLn * z - 0.5 * sLn * sLn) - c.SN_A) / c.SN_B);
    const hi = Math.log(Math.max(nPitches, 1e-9)) + logSumExp(terms(1));
    if (!opts.nLow || opts.nLow <= 0) return hi;
    const lo = Math.log(opts.nLow) + logSumExp(terms(opts.lowIntensity));
    const m = Math.max(hi, lo);
    return m + Math.log(Math.exp(hi - m) + Math.exp(lo - m));
  }

  // stress ratio k* at which a season's damage reaches 1 (bisection; damage rises with k)
  function criticalRatio(nPitches, c, opts) {
    let a = 1e-6, b = 400;
    for (let i = 0; i < 90; i++) {
      const m = 0.5 * (a + b);
      if (logSeasonDamage(m, nPitches, c, opts) > 0) b = m; else a = m;
    }
    return 0.5 * (a + b);
  }

  function erf(x) { // Abramowitz & Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

  function normInv(p) { // Acklam's rational approximation
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const cc = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  /** P(UCL fatigue failure within one season). */
  function seasonRisk(tau, nPitches, c, opts) {
    const o = Object.assign({ cv: c.TORQUE_CV, nLow: 0, lowIntensity: 0.7, phi: c.phi, sigma: c.STRENGTH_SIGMA }, opts || {});
    const tstar = 100 * o.phi * tau / criticalRatio(nPitches, c, o);
    return normCdf((Math.log(tstar) - Math.log(c.TAU_FAIL_MED)) / o.sigma);
  }

  /** Closed-form calibration: the UCL load share phi that gives the reference arm risk pRef. */
  function calibratePhi(c, pRef, sigma) {
    const kRef = criticalRatio(c.REF_WORKLOAD, c, { cv: c.TORQUE_CV });
    return c.TAU_FAIL_MED * Math.exp(sigma * normInv(pRef)) * kRef / (100 * c.tau_ref);
  }

  /** Ridge estimate of peak elbow varus torque (Nm) from mechanics. */
  function predictTorque(calc, values) {
    let y = calc.intercept;
    calc.features.forEach((f, i) => { y += calc.coef[i] * (values[f] - calc.mean[i]) / calc.scale[i]; });
    return y;
  }

  /** Elbow varus torque (Nm) from fastball speed and body size (the size + velocity model). */
  function torqueFromSizeVelo(sv, mph, massKg, heightM) {
    return sv.intercept + sv.coef.pitch_speed_mph * mph + sv.coef.mass_kg * massKg + sv.coef.height_m * heightM;
  }

  /** UCL stress per pitch, % of a size-scaled median ligament strength. */
  function stressFromTorque(c, tau, massKg) {
    return 100 * c.phi * tau / (c.TAU_FAIL_MED * Math.pow(massKg / c.REF_MASS, c.SIZE_EXP));
  }

  /** Season risk from stress S (%) and pitch count. Equivalent to seasonRisk() with size-scaled strength. */
  const kCache = new Map();
  function riskFromStress(c, S, nPitches, cv) {
    const v = cv == null ? c.TORQUE_CV : cv, key = Math.round(nPitches) + "|" + v;
    let k = kCache.get(key);
    if (k == null) { k = criticalRatio(Math.max(nPitches, 1), c, { cv: v }); kCache.set(key, k); }
    return normCdf(Math.log(S / k) / c.STRENGTH_SIGMA);
  }

  const api = { seasonRisk, calibratePhi, criticalRatio, predictTorque, normCdf, normInv,
                torqueFromSizeVelo, stressFromTorque, riskFromStress };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.UCL = api;
})(typeof window !== "undefined" ? window : globalThis);
