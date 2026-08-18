# us-specialty-trades

Reproduction pipeline for the research manuscript **"Consolidation, business
dynamism, and pay in US specialty trade contracting"** (under review at
*Construction Management and Economics*). The pipeline builds a
population-level picture of the US electrical (NAICS 238210) and
plumbing/HVAC (238220) contracting industries from federal statistical
data: firm size distributions and their Pareto tails, business dynamism and
firm dissolution, geographic concentration and county market dynamics, and
the relation between local consolidation and pay.

An interactive companion application built from the same outputs is at
https://specialty.electriai.com.

## Data sources

All inputs are public products of the US Census Bureau (Statistics of US
Businesses, County Business Patterns, Business Dynamics Statistics,
Economic Census, Population Estimates, Building Permits Survey, TIGER/Line
boundaries) and the US Bureau of Labor Statistics (Occupational Employment
and Wage Statistics). The download scripts fetch everything needed; no
license or API key is required.

## Pipeline

Scripts run in numeric order from `code/`; `run_all.py` runs the full
sequence. Each stage writes its numerical results to CSV before any figure
is drawn.

| Script | What it does |
| --- | --- |
| `01_download_census.py` | SUSB, CBP, BDS, and Economic Census downloads |
| `02_download_supplementary.py` | population, building permits, boundaries |
| `03_download_oews.py` | OEWS state and national files |
| `04_build_panels.py` | NAICS-harmonised panels across the 1998-2023 window |
| `05_descriptive_figures.py` | industry overview and geography figures |
| `06_tail_estimation.py` | discrete-MLE Pareto tails, bootstrap CIs, breaks |
| `07_county_transitions.py` | county Markov transitions, beta convergence |
| `08_spatial_analysis.py` | Moran's I and LISA clusters of county growth |
| `09_county_clustering.py` | k-means trajectory clusters |
| `10_growth_model.py` | gradient-boosted growth model, leave-state-out CV |
| `11_pay_structure.py` | pay levels, wage premium, fixed-effects estimates |
| `12_exit_by_size.py` | firm death rates by size class (BDS) |
| `13_summary_tables.py` | headline and validation tables |
| `14_location_scale_model.py` | hierarchical location-scale model of the premium |
| `15_oews_dispersion.py` | worker-level wage dispersion (OEWS) |

## Setup

```
python -m venv .venv && source .venv/bin/activate   # or your preferred env
pip install -r requirements.txt
cd code
python run_all.py
```

The location-scale model uses PyTorch and runs on GPU when available; the
cluster bootstrap treats each drawn state as a distinct cluster and refits
with the full optimiser budget, so the full run takes several hours on CPU.

## Notes on comparability

County Business Patterns changed disclosure rules inside the study window;
the pipeline applies the post-2017 publication floor (fewer than three
establishments unpublished) to every year wherever a statistic has a set of
counties in its denominator, so series are comparable across the redesign.
NAICS 1997 codes (235310, 235110) are concatenated with their 2002+
successors (238210, 238220) after verifying the seam.

## Citation

A citation file will be added when the article is published. Until then,
please cite the manuscript "Consolidation, business dynamism, and pay in US
specialty trade contracting" (under review).
