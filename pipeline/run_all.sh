#!/usr/bin/env bash
# Rebuild everything from raw motion capture to the website's data files.
#   bash pipeline/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "1/7  fetch + verify data";        python3 fetch_data.py
echo "2/7  inverse dynamics";           python3 compute_physics.py
echo "3/7  elbow-stress models";        python3 model_stress.py
echo "4/7  UCL fatigue-failure model";  python3 risk_model.py
echo "5/7  MLB 2026 pitch counts";      python3 mlb.py        # refresh the season table first with: python3 fetch_mlb.py
echo "6/7  export website data";        python3 export_site.py
echo "7/7  build-your-pitcher model";    python3 build_model.py && node build_check.js
echo "done — open public/ with any static server, e.g.  cd public && python3 -m http.server"
