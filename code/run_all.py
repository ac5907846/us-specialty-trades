"""Run the pipeline.

    python run_all.py                  everything, including downloads
    python run_all.py --skip-download  data already in data/
    python run_all.py --only 06        run a single stage
    python run_all.py --from 05        run from a stage onwards

Stage 14 (the location-scale model) is excluded by default because its
cluster bootstrap takes hours; run it separately:
    python 14_location_scale_model.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("01_download_census.py", True),
    ("02_download_supplementary.py", True),
    ("03_download_oews.py", True),
    ("04_build_panels.py", False),
    ("05_descriptive_figures.py", False),
    ("06_tail_estimation.py", False),
    ("07_county_transitions.py", False),
    ("08_spatial_analysis.py", False),
    ("09_county_clustering.py", False),
    ("10_growth_model.py", False),
    ("11_pay_structure.py", False),
    ("12_exit_by_size.py", False),
    ("13_summary_tables.py", False),
    ("15_oews_dispersion.py", False),
]


def selected():
    argv = sys.argv[1:]
    stages = list(STAGES)
    if "--skip-download" in argv:
        stages = [s for s in stages if not s[1]]
    if "--only" in argv:
        want = argv[argv.index("--only") + 1]
        stages = [s for s in stages if s[0].startswith(want)]
    if "--from" in argv:
        want = argv[argv.index("--from") + 1]
        idx = next((i for i, s in enumerate(stages)
                    if s[0].startswith(want)), 0)
        stages = stages[idx:]
    return stages


def main():
    summary, t_start = [], time.time()
    for script, _ in selected():
        print(f"\n=== {script} ===")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)])
        status = ("ok" if r.returncode == 0 else "FAILED")
        summary.append((script, f"{status} ({(time.time() - t0) / 60:.1f} min)"))

    print("\n=== summary ===")
    for script, status in summary:
        print(f"  {script:32s} {status}")
    print(f"  total {(time.time() - t_start) / 60:.1f} min")
    print("\nFigures: output/figures    Tables: output/tables")


if __name__ == "__main__":
    main()
