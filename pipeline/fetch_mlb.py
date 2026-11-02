"""
Optional — re-download the MLB season table used on the site (data/mlb_2026.csv).

Sources (public):
  MLB Stats API      season pitch counts, games, starts; height and weight
  Baseball Savant    average four-seam and sinker velocity (pitch-arsenals leaderboard)

Keeps pitchers with at least MIN_PITCHES regular-season pitches.
Usage:  python3 fetch_mlb.py [season]
"""
import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
MIN_PITCHES = 1000
API = "https://statsapi.mlb.com/api/v1"
SAVANT = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?year={y}&min=1&type=avg_speed&hand=&csv=true"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ucl-load-lab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8-sig")


def main():
    stats = json.loads(get(f"{API}/stats?stats=season&group=pitching&season={SEASON}&sportId=1&playerPool=ALL"
                           f"&limit=2000&gameType=R&fields=stats,splits,stat,numberOfPitches,gamesStarted,"
                           f"gamesPitched,player,id,fullName"))
    rows = [s for s in stats["stats"][0]["splits"] if s["stat"]["numberOfPitches"] >= MIN_PITCHES]

    people = {}
    ids = [r["player"]["id"] for r in rows]
    for i in range(0, len(ids), 100):
        js = json.loads(get(f"{API}/people?personIds={','.join(map(str, ids[i:i+100]))}"
                            f"&fields=people,id,height,weight,pitchHand,code"))
        for p in js["people"]:
            ft, inch = map(int, re.match(r"(\d+)' (\d+)", p["height"]).groups())
            people[p["id"]] = (12 * ft + inch, p["weight"], p["pitchHand"]["code"])

    velo = {}
    for rec in csv.DictReader(io.StringIO(get(SAVANT.format(y=SEASON)))):
        velo[int(rec["pitcher"])] = (rec["ff_avg_speed"], rec["si_avg_speed"])

    out = ROOT / "data" / f"mlb_{SEASON}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "name", "pitches", "gs", "g", "height_in", "weight_lb", "throws", "ff_mph", "si_mph"])
        for r in sorted(rows, key=lambda r: -r["stat"]["numberOfPitches"]):
            pid = r["player"]["id"]
            h, wt, hand = people[pid]
            ff, si = velo.get(pid, ("", ""))
            w.writerow([pid, r["player"]["fullName"].replace(",", ""), r["stat"]["numberOfPitches"],
                        r["stat"]["gamesStarted"], r["stat"]["gamesPitched"], h, wt, hand, ff, si])
    print(f"wrote {out} ({len(rows)} pitchers)")


if __name__ == "__main__":
    main()
