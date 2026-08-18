"""Shared paths, industry definitions, and data loaders."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Workbook contains no default style.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DIR_BDS = RAW / "bds"
DIR_CBP = RAW / "cbp"
DIR_SUSB = RAW / "susb"
DIR_EC = RAW / "economic_census"
SUPP = RAW / "supplementary"
DIR_POP = SUPP / "population"
DIR_BPS = SUPP / "building_permits"
DIR_GEO = SUPP / "geography"
DIR_OEWS = SUPP / "oews"
INTERIM = DATA / "interim" / "cbp_sector23"
PROCESSED = DATA / "processed"
FIGS = ROOT / "output" / "figures"
TABS = ROOT / "output" / "tables"

for _d in (PROCESSED, FIGS, TABS):
    _d.mkdir(parents=True, exist_ok=True)


def ensure_dirs():
    for d in (DIR_BDS, DIR_CBP, DIR_SUSB, DIR_EC, DIR_POP, DIR_BPS, DIR_GEO,
              DIR_OEWS, INTERIM, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)


LBL = {"electrical": "Electrical (238210)",
       "plumbing_hvac": "Plumbing/HVAC (238220)"}
SHORT = {"electrical": "Electrical", "plumbing_hvac": "Plumbing/HVAC"}

# NAICS 1997 codes concatenated with their NAICS 2002+ successors so the
# six-digit series run continuously from 1998
TRADE_CODES = {"235310": "electrical", "238210": "electrical",
               "235110": "plumbing_hvac", "238220": "plumbing_hvac"}
CONSTR_4DIGIT = ["2361", "2362", "2371", "2372", "2373", "2379",
                 "2381", "2382", "2383", "2389"]

# CBP establishment size classes and their midpoints (top class open ended)
SIZES = ["n1_4", "n5_9", "n10_19", "n20_49", "n50_99",
         "n100_249", "n250_499", "n500_999", "n1000"]
MID = np.array([2.5, 7, 14.5, 34.5, 74.5, 174.5, 374.5, 749.5, 1500.0])
LARGE_FROM = 5

# SUSB detailed firm size classes used for tail estimation
DETAILED_BINS = [("20-24", 20), ("25-29", 25), ("30-34", 30), ("35-39", 35),
                 ("40-44", 40), ("45-49", 45), ("50-74", 50), ("75-99", 75),
                 ("100-149", 100), ("150-199", 150), ("200-299", 200),
                 ("300-399", 300), ("400-499", 400), ("500-749", 500),
                 ("750-999", 750), ("1000-1499", 1000), ("1500-1999", 1500),
                 ("2000-2499", 2000), ("2500-4999", 2500), ("5000+", 5000)]
BIN_UPPER = {l: u for (l, _), u in zip(
    DETAILED_BINS, [25, 30, 35, 40, 45, 50, 75, 100, 150, 200, 300, 400,
                    500, 750, 1000, 1500, 2000, 2500, 5000, np.inf])}

# From reference year 2017 CBP withholds county-industry cells with fewer
# than three establishments; applying the same floor to every year gives a
# series comparable across the redesign
CBP_PUBLICATION_BREAK = 2016.5
MIN_EST_HARMONISED = 3


def save_table(df, name):
    df.to_csv(TABS / f"{name}.csv", index=True)
    print(f"  table -> {name}.csv")


def save_fig(fig, name):
    fig.savefig(FIGS / f"{name}.png")
    print(f"  figure -> {name}.png")


def load_cbp(force=False):
    """County x year x NAICS panel for construction, 1998-2023."""
    out = PROCESSED / "cbp_county.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    files = sorted(INTERIM.glob("cbp*co_sector23.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"no filtered CBP files in {INTERIM}; run 01_download_census.py")
    frames = []
    for f in files:
        year = int(f.name[3:7])
        df = pd.read_csv(f, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"n<5": "n1_4"})
        keep = ["fipstate", "fipscty", "naics", "emp", "qp1", "ap", "est"] \
            + SIZES + (["empflag"] if "empflag" in df.columns else [])
        df = df[[c for c in keep if c in df.columns]].copy()
        for c in ["emp", "qp1", "ap", "est"] + SIZES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["year"] = year
        df["naics"] = df["naics"].str.replace("-", "").str.replace("/", "")
        df["fips"] = df["fipstate"].str.zfill(2) + df["fipscty"].str.zfill(3)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(out)
    return panel


def load_susb(force=False):
    """Firm counts by employment size class, US and states, 1998-2022."""
    out = PROCESSED / "susb_sizes.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    frames = []

    def keep(df):
        n = df["naics"].astype(str).str.strip()
        return df[(n == "--") | n.str.startswith("23")].copy()

    def add(df, year):
        df = df.copy()
        df["year"] = year
        for c in ["firms", "estabs", "emp", "payr", "rcpt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                                      errors="coerce")
            else:
                df[c] = np.nan
        if "state" not in df.columns:
            df["state"] = "00"
        df["state"] = df["state"].astype(str).str.zfill(2)
        df["naics"] = df["naics"].astype(str).str.strip()
        df["size_label"] = (df["size_label"].astype(str)
                            .str.replace(r"^\d+:\s*", "", regex=True)
                            .str.replace(" employees", "", regex=False)
                            .str.strip())
        frames.append(df[["year", "state", "naics", "size_label",
                          "firms", "estabs", "emp", "payr", "rcpt"]])

    def load_txt(path, year):
        df = pd.read_csv(path, dtype=str, encoding="latin-1")
        df.columns = [c.strip().upper() for c in df.columns]
        ren = {"STATE": "state", "NAICS": "naics", "FIRM": "firms",
               "ESTB": "estabs", "EMPL": "emp", "PAYR": "payr",
               "RCPT": "rcpt", "ENTRSIZEDSCR": "size_label"}
        df = df.rename(columns={k: v for k, v in ren.items()
                                if k in df.columns})
        add(keep(df), year)

    def load_old_xls(path, year):
        raw = pd.read_excel(path, header=None)
        hdr = raw.index[raw[0].astype(str).str.strip().eq("CODE")][0]
        cols = ["naics", "descr", "data_type"] + [str(c).strip()
                                                  for c in raw.iloc[hdr, 3:]]
        df = raw.iloc[hdr + 1:].copy()
        df.columns = cols
        df = keep(df)
        long = df.melt(id_vars=["naics", "descr", "data_type"],
                       var_name="size_label", value_name="value")
        piv = long.pivot_table(index=["naics", "size_label"],
                               columns="data_type", values="value",
                               aggfunc="first").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        piv = piv.rename(columns={"Firms": "firms",
                                  "Establishments": "estabs",
                                  "Employment": "emp",
                                  "Annual Payroll ($1,000)": "payr"})
        add(piv, year)

    def load_xlsx(path, year):
        raw = pd.read_excel(path, header=None)
        hdr = None
        for i in range(min(10, len(raw))):
            vals = [str(v).upper() for v in raw.iloc[i]]
            if any("NAICS" in v for v in vals) and any("SIZE" in v
                                                       for v in vals):
                hdr = i
                break
        if hdr is None:
            return
        df = raw.iloc[hdr + 1:].copy()
        cols = []
        for v in raw.iloc[hdr]:
            v = str(v).upper().replace("\n", " ")
            if v.startswith("STATE NAME"):
                cols.append("state_name")
            elif v.startswith("STATE"):
                cols.append("state")
            elif "DESCRIPTION" in v:
                cols.append("descr")
            elif "NAICS" in v or v.strip() == "CODE":
                cols.append("naics")
            elif "SIZE" in v:
                cols.append("size_label")
            elif "FIRM" in v:
                cols.append("firms")
            elif "ESTABLISHMENT" in v:
                cols.append("estabs")
            elif "EMPLOYMENT" in v and "FLAG" not in v:
                cols.append("emp")
            elif "PAYROLL" in v and "FLAG" not in v:
                cols.append("payr")
            elif "RECEIPTS" in v and "FLAG" not in v:
                cols.append("rcpt")
            else:
                cols.append(f"x{len(cols)}")
        df.columns = cols
        df = df.dropna(subset=["naics"])
        add(keep(df), year)

    for f in sorted(DIR_SUSB.glob("susb_*")):
        m = re.match(r"susb_(\d{4})_", f.name)
        if not m:
            continue
        year, name = int(m.group(1)), f.name.lower()
        if not ("detailedsizes" in name or "6digitnaics" in name):
            continue
        if any(k in name for k in ("rcptsize", "emplchange", "empl2500",
                                   "large_emplsize")):
            continue
        try:
            if name.endswith((".txt", ".csv")):
                load_txt(f, year)
            elif name.endswith((".xlsx", ".xls")):
                probe = pd.read_excel(f, header=None, nrows=12)
                is_old = probe.apply(
                    lambda r: r.astype(str).str.strip()
                    .eq("DATA TYPE").any(), axis=1).any()
                if is_old:
                    load_old_xls(f, year)
                else:
                    load_xlsx(f, year)
        except Exception as e:
            print("  SUSB load failed:", f.name, "->", e)
    if not frames:
        raise FileNotFoundError(f"no SUSB files found in {DIR_SUSB}")
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(out)
    return panel


def susb_us_detailed(trade="electrical"):
    """US level year x size-bin firm counts for one trade."""
    p = load_susb()
    us = p[p.state == "00"].copy()
    us["trade"] = us.naics.map(TRADE_CODES)
    us = us[us.trade == trade]
    us["lab"] = (us.size_label.str.upper().str.replace(" ", "")
                 .str.replace(",", ""))
    return us.drop_duplicates(subset=["year", "lab"])


def load_bds(kind="vcn4"):
    """BDS series. kind in {'', 'sec', 'vcn4', 'vcn4_fa', 'vcn4_fz', 'st_sec'}."""
    name = f"bds2023_{kind}.csv" if kind else "bds2023.csv"
    path = DIR_BDS / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {DIR_BDS}")
    df = pd.read_csv(path, dtype=str)
    for c in df.columns:
        if c not in ("vcnaics4", "sector", "fage", "fsize", "st"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def trade_frame(cbp, trade):
    codes = [k for k, v in TRADE_CODES.items() if v == trade]
    return cbp[cbp.naics.isin(codes)]
