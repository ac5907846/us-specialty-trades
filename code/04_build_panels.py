"""Build the analysis panels from the raw downloads. All later scripts read
the parquet and csv files written here."""
import zipfile

import numpy as np
import pandas as pd

from config import (load_cbp, load_susb, load_bds, save_table, PROCESSED,
                    DIR_EC, TRADE_CODES, CONSTR_4DIGIT, MIN_EST_HARMONISED)


def build_bds():
    v4 = load_bds("vcn4")
    v4["year"] = v4["year"].astype(int)
    spec = v4[v4.vcnaics4.isin(CONSTR_4DIGIT)].copy()
    spec["is_238"] = spec.vcnaics4.str.startswith("238")
    spec.to_parquet(PROCESSED / "bds_specialty_4digit.parquet")

    for kind, tag in (("vcn4_fa", "fa"), ("vcn4_fz", "fz")):
        d = load_bds(kind)
        d = d[d.vcnaics4.isin(CONSTR_4DIGIT)].copy()
        d.to_parquet(PROCESSED / f"bds_specialty_{tag}.parquet")

    econ = load_bds("")
    econ["level"] = "economy"
    sec = load_bds("sec")
    sec = sec[sec.sector.astype(str).str.zfill(2) == "23"].copy()
    sec["level"] = "construction_sector"
    pd.concat([econ, sec], ignore_index=True).to_parquet(
        PROCESSED / "bds_benchmarks.parquet")
    print(f"  BDS: {len(spec)} four-digit rows, "
          f"{spec.year.min()}-{spec.year.max()}")
    return spec


def build_trade_size_trends():
    p = load_susb()
    us = p[p.state == "00"].copy()
    us["trade"] = us.naics.map(TRADE_CODES)
    t = us[us.trade.notna()].copy()
    t["lab"] = (t.size_label.str.upper().str.replace(" ", "")
                .str.replace(",", ""))
    t = t[t.naics.str.len() == 6].drop_duplicates(["year", "trade", "lab"])

    rows = []
    for (yr, tr), g in t.groupby(["year", "trade"]):
        d = g.set_index("lab")
        if "TOTAL" not in d.index:
            continue
        tot_e, tot_f = d.loc["TOTAL", "emp"], d.loc["TOTAL", "firms"]
        lt20 = d.loc["<20", "emp"] if "<20" in d.index else np.nan
        lt500 = d.loc["<500", "emp"] if "<500" in d.index else np.nan
        f_lt20 = d.loc["<20", "firms"] if "<20" in d.index else np.nan
        rows.append(dict(year=yr, trade=tr, firms=tot_f, emp=tot_e,
                         payr=d.loc["TOTAL", "payr"],
                         sh_lt20=lt20 / tot_e,
                         sh_500p=(tot_e - lt500) / tot_e,
                         firmshare_lt20=f_lt20 / tot_f,
                         meansize=tot_e / tot_f))
    r = pd.DataFrame(rows).sort_values(["trade", "year"])
    r.to_csv(PROCESSED / "trade_size_trends.csv", index=False)
    save_table(r.set_index(["trade", "year"]), "tab_trade_size_trends")
    return r


def build_concentration():
    """County concentration on the published and the harmonised basis."""
    cbp = load_cbp()
    t = cbp[cbp.naics.isin(TRADE_CODES)].copy()
    t["trade"] = t.naics.map(TRADE_CODES)
    rows = []
    for (yr, tr), g in t.groupby(["year", "trade"]):
        for basis, sub in (("published", g),
                           ("harmonised", g[g.est >= MIN_EST_HARMONISED])):
            b = sub.groupby("fips").est.sum().sort_values(ascending=False)
            if b.empty:
                continue
            rows.append(dict(year=yr, trade=tr, basis=basis, counties=len(b),
                             estabs=b.sum(),
                             top20=b.head(20).sum() / b.sum(),
                             top100=b.head(100).sum() / b.sum(),
                             hhi_county=((b / b.sum()) ** 2).sum() * 10000))
    r = pd.DataFrame(rows)
    r.to_csv(PROCESSED / "cbp_concentration.csv", index=False)
    save_table(r[r.basis == "harmonised"].set_index(["trade", "year"])
               .drop(columns="basis"), "tab_concentration_harmonised")
    return r


def build_ec_receipts():
    rows = []
    for zname, ncol, year in (("ec2022_EC2223BASIC.zip", "NAICS2022", 2022),
                              ("ec2017_EC1723BASIC.zip", "NAICS2017", 2017)):
        path = DIR_EC / zname
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as zf:
            member = [n for n in zf.namelist() if n.endswith(".dat")][0]
            df = pd.read_csv(zf.open(member), dtype=str, sep="|")
        df.columns = [c.replace("#", "") for c in df.columns]
        us = df[df["GEO_ID"].str.contains("0100000US", na=False)]
        for code, trade in (("238210", "electrical"),
                            ("238220", "plumbing_hvac")):
            sub = us[us[ncol] == code]
            if sub.empty:
                continue
            rec = pd.to_numeric(sub["RCPTOT"], errors="coerce").iloc[0]
            firm = pd.to_numeric(sub["FIRM"], errors="coerce").iloc[0]
            emp = pd.to_numeric(sub["EMP"], errors="coerce").iloc[0]
            pay = pd.to_numeric(sub["PAYANN"], errors="coerce").iloc[0]
            rows.append(dict(year=year, trade=trade, firms=firm, emp=emp,
                             receipts_musd=rec / 1000, payroll_musd=pay / 1000,
                             receipts_per_firm_musd=rec / 1000 / firm,
                             receipts_per_worker_usd=rec * 1000 / emp))
    if rows:
        r = pd.DataFrame(rows).sort_values(["trade", "year"])
        r.to_csv(PROCESSED / "economic_census_receipts.csv", index=False)
        save_table(r.set_index(["trade", "year"]), "tab_economic_census")
        return r
    print("  Economic Census files not found, skipping")
    return None


def validate(trends):
    """Cross-source firm-count check, SUSB against the Economic Census."""
    checks = []
    ec = PROCESSED / "economic_census_receipts.csv"
    if ec.exists():
        e = pd.read_csv(ec)
        for tr in ("electrical", "plumbing_hvac"):
            for yr in (2017, 2022):
                a = trends[(trends.trade == tr) & (trends.year == yr)]
                b = e[(e.trade == tr) & (e.year == yr)]
                if len(a) and len(b):
                    s, c = a.firms.iloc[0], b.firms.iloc[0]
                    checks.append(dict(trade=tr, year=yr, susb_firms=s,
                                       ec_firms=c, pct_diff=(s - c) / c * 100))
    if checks:
        save_table(pd.DataFrame(checks).set_index(["trade", "year"]),
                   "tab_validation_susb_vs_ec")


def main():
    print("=== building panels ===")
    cbp = load_cbp(force=True)
    print(f"  CBP: {len(cbp):,} rows, {cbp.year.min()}-{cbp.year.max()}")
    load_susb(force=True)
    build_bds()
    trends = build_trade_size_trends()
    build_concentration()
    build_ec_receipts()
    validate(trends)
    print("done.")


if __name__ == "__main__":
    main()
