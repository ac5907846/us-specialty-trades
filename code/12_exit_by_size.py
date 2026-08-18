"""Firm death rates and death shares by firm size, BDS industry 2382,
1978-2023. firmdeath counts firms whose every establishment closed, so it
measures dissolution, not acquisition."""
import pandas as pd

from config import load_bds, save_table

GROUP = {"a) 1 to 4": "small", "b) 5 to 9": "small", "c) 10 to 19": "small",
         "d) 20 to 99": "mid", "e) 100 to 499": "mid",
         "f) 500 to 999": "large", "g) 1000 to 2499": "large",
         "h) 2500 to 4999": "large", "i) 5000 to 9999": "large",
         "j) 10000+": "large"}


def main():
    d = load_bds("vcn4_fz")
    d = d[d.vcnaics4.astype(str) == "2382"].copy()
    for c in ("firms", "firmdeath_firms", "emp"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["group"] = d.fsize.map(GROUP)
    a = (d.groupby(["year", "group"])[["firms", "firmdeath_firms"]]
         .sum().reset_index())
    a["death_rate_pct"] = 100 * a.firmdeath_firms / a.firms
    tot = a.groupby("year").firmdeath_firms.transform("sum")
    a["share_of_deaths_pct"] = 100 * a.firmdeath_firms / tot

    wide = a.pivot(index="year", columns="group",
                   values=["firms", "firmdeath_firms", "death_rate_pct",
                           "share_of_deaths_pct"]).round(2)
    save_table(wide, "tab_exit_by_size")

    rec = a[a.year.between(2017, 2023)].groupby("group").agg(
        death_rate_pct=("death_rate_pct", "mean"),
        share_of_deaths_pct=("share_of_deaths_pct", "mean")).round(2)
    print("  2382, 2017-2023 means:")
    print(rec.to_string())


if __name__ == "__main__":
    main()
