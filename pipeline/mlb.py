"""
Step 4b — put every 2026 MLB pitcher (1,000+ pitches) on the elbow-stress map.

For each pitcher:
  fastball velocity = faster of his average four-seam and sinker (Statcast)
  elbow torque      = size + velocity model from the motion-capture data
                      (no mocap exists for MLB pitchers, so mechanics are assumed average)
  elbow stress S    = UCL load / size-scaled ligament strength, % of strength
  season risk       = fatigue model with his real 2026 pitch count

Output: public/data/mlb.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import SN_A, SN_B, TAU_FAIL_MED, season_risk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main(season=2026):
    risk = json.loads((DATA / "risk_model.json").read_text())
    stress = json.loads((DATA / "stress_model.json").read_text())
    c, sv = risk["constants"], stress["size_velo_coef"]
    phi = c["phi"]

    m = pd.read_csv(DATA / f"mlb_{season}.csv")
    m["velo"] = m[["ff_mph", "si_mph"]].max(axis=1)
    m = m[m.velo.notna()].copy()
    m["mass_kg"] = m.weight_lb * 0.45359237
    m["height_m"] = m.height_in * 0.0254
    m["torque"] = (sv["intercept"] + sv["coef"]["pitch_speed_mph"] * m.velo
                   + sv["coef"]["mass_kg"] * m.mass_kg + sv["coef"]["height_m"] * m.height_m)
    m["scale"] = (m.mass_kg / c["REF_MASS"]) ** c["SIZE_EXP"]
    m["stress"] = 100 * phi * m.torque / (TAU_FAIL_MED * m.scale)
    m["risk"] = [season_risk(t / s, n, phi) for t, s, n in zip(m.torque, m.scale, m.pitches)]
    m["role"] = np.where(m.gs >= 0.5 * m.g, "SP", "RP")
    m = m.sort_values("risk", ascending=False)

    cols = ["name", "role", "throws", "pitches", "velo", "weight_lb", "height_in", "torque", "stress", "risk"]
    out = dict(season=season, n=int(len(m)), min_pitches=int(m.pitches.min()), cols=cols,
               rows=[[r.name, r.role, r.throws, int(r.pitches), round(r.velo, 1), int(r.weight_lb),
                      int(r.height_in), round(r.torque, 1), round(r.stress, 2), round(r.risk, 5)]
                     for r in m.itertuples()])
    (ROOT / "public" / "data" / "mlb.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(m)} MLB pitchers | median risk {m.risk.median():.3f} | stress {m.stress.min():.1f}–{m.stress.max():.1f}%")
    print(m[["name", "role", "pitches", "velo", "weight_lb", "torque", "stress", "risk"]].head(10).round(3).to_string(index=False))
    print(m.groupby("role").risk.median().round(4).to_dict())


if __name__ == "__main__":
    main()
