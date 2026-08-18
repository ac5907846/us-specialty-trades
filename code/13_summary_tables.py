"""Data-source inventory and stylized-facts tables. Reads only files the
earlier stages have written, so it runs last."""
import numpy as np
import pandas as pd

from config import PROCESSED, TABS, save_table


def data_sources():
    rows = [
        ("Statistics of US Businesses (SUSB)", "Census Bureau", "1998-2022",
         "firms by enterprise size class, US and state",
         "firm size distribution, tail exponents"),
        ("County Business Patterns (CBP)", "Census Bureau", "1998-2023",
         "establishments, employment, payroll by county and NAICS",
         "geography, county dynamics, pay"),
        ("Business Dynamics Statistics (BDS)", "Census Bureau", "1978-2023",
         "entry, exit, firm age and firm size, 4-digit NAICS",
         "dynamism, aging, exit margins"),
        ("Economic Census", "Census Bureau", "2012, 2017, 2022",
         "receipts, payroll, employment by 6-digit NAICS",
         "value dimension, validation"),
        ("Occupational Employment and Wage Statistics (OEWS)",
         "Bureau of Labor Statistics", "2003-2024",
         "hourly wage percentiles by occupation, national and state",
         "worker-level wage dispersion"),
        ("Population Estimates Program", "Census Bureau", "2000-2024",
         "county population", "growth model features"),
        ("Building Permits Survey", "Census Bureau", "2013-2016",
         "county residential permits", "growth model features"),
        ("TIGER/Line cartographic boundaries", "Census Bureau", "2020, 2023",
         "county geometries", "maps, spatial weights"),
    ]
    df = pd.DataFrame(rows, columns=["source", "agency", "coverage",
                                     "content", "role_in_paper"])
    save_table(df.set_index("source"), "tab_data_sources")


def stylized_facts():
    tr = pd.read_csv(PROCESSED / "trade_size_trends.csv")
    alpha = pd.read_csv(TABS / "tab_alpha_by_year.csv")
    conc = pd.read_csv(PROCESSED / "cbp_concentration.csv")
    pay = pd.read_csv(TABS / "tab_pay_trends.csv")
    spec = pd.read_parquet(PROCESSED / "bds_specialty_4digit.parquet")
    fa = pd.read_parquet(PROCESSED / "bds_specialty_fa.parquet")

    rows = []
    for trade in ("electrical", "plumbing_hvac"):
        t = tr[tr.trade == trade].sort_values("year")
        a, b = t.iloc[0], t.iloc[-1]
        al = alpha[alpha.trade == trade].sort_values("year")
        c = conc[(conc.trade == trade)
                 & (conc.basis == "harmonised")].sort_values("year")
        p = pay[pay.trade == trade].sort_values("year")

        def add(ind, v0, v1, y0, y1, unit):
            rows.append(dict(trade=trade, indicator=ind, unit=unit,
                             first_year=int(y0), first=v0,
                             latest_year=int(y1), latest=v1))

        add("Employer firms", a.firms, b.firms, a.year, b.year, "count")
        add("Employment", a.emp, b.emp, a.year, b.year, "count")
        add("Mean firm size", a.emp / a.firms, b.emp / b.firms,
            a.year, b.year, "employees")
        add("Employment share, firms 500+", a.sh_500p * 100, b.sh_500p * 100,
            a.year, b.year, "%")
        add("Employment share, firms <20", a.sh_lt20 * 100, b.sh_lt20 * 100,
            a.year, b.year, "%")
        add("Tail exponent (MLE)", al.alpha_mle.iloc[0],
            al.alpha_mle.iloc[-1], al.year.iloc[0], al.year.iloc[-1],
            "alpha")
        add("Top-20 county share (harmonised)", c.top20.iloc[0] * 100,
            c.top20.iloc[-1] * 100, c.year.iloc[0], c.year.iloc[-1], "%")
        add("Average pay per employee", p.pay_national.iloc[0],
            p.pay_national.iloc[-1], p.year.iloc[0], p.year.iloc[-1],
            "$ nominal")
        add("County wage premium (weighted)", p.premium_wtd.iloc[0] * 100,
            p.premium_wtd.iloc[-1] * 100, p.year.iloc[0], p.year.iloc[-1],
            "log points")

    b2 = spec[spec.vcnaics4.astype(str) == "2382"].sort_values("year")
    dec70 = b2[b2.year.between(1970, 1979)].estabs_entry_rate.mean()
    dec20 = b2[b2.year >= 2020].estabs_entry_rate.mean()
    g = fa[fa.vcnaics4.astype(str) == "2382"].copy()
    g["firms"] = pd.to_numeric(g.firms, errors="coerce")
    mature = ["i) 16 to 20", "j) 21 to 25", "k) 26+", "l) Left Censored"]
    tot = g.groupby("year").firms.sum()
    ms = g[g.fage.isin(mature)].groupby("year").firms.sum() / tot
    rows.append(dict(trade="2382 (both trades)",
                     indicator="Entry rate, decade mean",
                     unit="%", first_year=1970, first=dec70,
                     latest_year=2020, latest=dec20))
    rows.append(dict(trade="2382 (both trades)",
                     indicator="Share of firms aged 16+",
                     unit="%", first_year=1992,
                     first=ms.loc[1992] * 100,
                     latest_year=int(ms.index.max()),
                     latest=ms.iloc[-1] * 100))

    df = pd.DataFrame(rows)
    df["change_pct"] = np.where(
        df.unit == "count", (df.latest / df.first - 1) * 100, np.nan)
    save_table(df.set_index(["trade", "indicator"]).round(3),
               "tab_stylized_facts")
    print(f"  {len(df)} stylized-fact rows")


def main():
    data_sources()
    stylized_facts()


if __name__ == "__main__":
    main()
