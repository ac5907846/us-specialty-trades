"""Pay from CBP payroll: average pay is ap*1000/emp and the trade wage
premium is log pay relative to the same county's all-industry average.
Includes the two-way fixed effects regression of the premium on the local
large-establishment employment share."""
import numpy as np
import pandas as pd

from config import (load_cbp, save_table, save_fig, PROCESSED,
                    TRADE_CODES, SIZES, MID, LBL,
                    CBP_PUBLICATION_BREAK)
import theme

plt = theme.apply()
C_EL, C_PL, GRAY = theme.PETROL, theme.OLIVE, theme.INK_3

MIN_EMP = 50
BREAK = CBP_PUBLICATION_BREAK

FIPS_TO_USPS = {
 "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
 "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
 "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
 "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
 "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
 "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
 "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
 "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
 "54": "WV", "55": "WI", "56": "WY"}


def build_panel():
    cbp = load_cbp()
    cbp = cbp[cbp.emp.notna() & (cbp.emp > 0) & cbp.ap.notna()]

    tot = (cbp[cbp.naics == ""]
           .groupby(["fips", "year"])[["emp", "ap"]].sum()
           .rename(columns={"emp": "emp_all", "ap": "ap_all"}))

    frames = []
    for trade in ("electrical", "plumbing_hvac"):
        codes = [k for k, v in TRADE_CODES.items() if v == trade]
        d = cbp[cbp.naics.isin(codes)].copy()
        g = (d.groupby(["fips", "year"])
             .agg(emp=("emp", "sum"), ap=("ap", "sum"), est=("est", "sum"),
                  **{c: (c, "sum") for c in SIZES}))
        g["trade"] = trade
        frames.append(g)
    panel = pd.concat(frames).join(tot, how="inner").reset_index()

    panel["pay"] = panel.ap * 1000.0 / panel.emp
    panel["pay_all"] = panel.ap_all * 1000.0 / panel.emp_all
    panel["premium"] = np.log(panel.pay) - np.log(panel.pay_all)
    emp_mid = panel[SIZES].fillna(0).values * MID
    with np.errstate(invalid="ignore", divide="ignore"):
        panel["sh_lg"] = emp_mid[:, 5:].sum(1) / np.where(
            emp_mid.sum(1) > 0, emp_mid.sum(1), np.nan)
    panel["state"] = panel.fips.str[:2]
    panel = panel[(panel.emp >= MIN_EMP) & panel.pay.between(10_000, 250_000)]
    return panel


def main():
    panel = build_panel()
    panel.to_parquet(PROCESSED / "cbp_pay_panel.parquet")
    print(f"  panel: {len(panel):,} county-year-trade cells, "
          f"{panel.fips.nunique():,} counties")

    rows = []
    for (yr, tr), g in panel.groupby(["year", "trade"]):
        w = g.emp
        rows.append(dict(
            year=yr, trade=tr, counties=len(g),
            pay_national=(g.ap.sum() * 1000.0 / g.emp.sum()),
            pay_p10=g.pay.quantile(.10), pay_p50=g.pay.median(),
            pay_p90=g.pay.quantile(.90),
            p90_p10=g.pay.quantile(.90) / g.pay.quantile(.10),
            premium_median=g.premium.median(),
            premium_wtd=np.average(g.premium, weights=w),
            sd_log_pay=np.std(np.log(g.pay))))
    tr_df = pd.DataFrame(rows).sort_values(["trade", "year"])
    save_table(tr_df.set_index(["trade", "year"]), "tab_pay_trends")

    reg_rows = []
    import statsmodels.formula.api as smf
    for trade in ("electrical", "plumbing_hvac"):
        d = panel[(panel.trade == trade) & panel.sh_lg.notna()].copy()
        for era, sub in (("1998-2016", d[d.year <= 2016]),
                         ("2017-2023", d[d.year >= 2017])):
            sub = sub[sub.groupby("fips").fips.transform("size") >= 3]
            if len(sub) < 500:
                continue
            m = smf.ols("premium ~ sh_lg + C(fips) + C(year)", data=sub)\
                .fit(cov_type="cluster",
                     cov_kwds={"groups": sub.state})
            reg_rows.append(dict(
                trade=trade, era=era, spec="county + year FE",
                beta_sh_lg=m.params["sh_lg"], se=m.bse["sh_lg"],
                t=m.tvalues["sh_lg"], p=m.pvalues["sh_lg"],
                n=int(m.nobs), counties=sub.fips.nunique()))

    # highest and lowest paying markets with at least 1,000 trade employees
    last = panel[(panel.year == 2023) & (panel.trade == "electrical")
                 & (panel.emp >= 1000)].copy()
    last["state"] = last.fips.str[:2].map(FIPS_TO_USPS)
    try:
        import geopandas as gpd
        from config import DIR_GEO
        shp = next(DIR_GEO.glob("cb_*_us_county_20m.zip"))
        names = gpd.read_file(f"zip://{shp}")[["GEOID", "NAME"]]
        last = last.merge(names.rename(columns={"GEOID": "fips",
                                                "NAME": "county"}),
                          on="fips", how="left")
    except Exception:
        last["county"] = ""
    cols = ["county", "state", "emp", "est", "pay", "premium"]
    top = last.nlargest(12, "pay")[["fips"] + cols]
    bot = last.nsmallest(8, "pay")[["fips"] + cols]
    save_table(pd.concat([top.assign(group="highest"),
                          bot.assign(group="lowest")]).set_index("fips"),
               "tab_pay_extremes")

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2))

    ax = axes[0, 0]
    for tr, c in (("electrical", C_EL), ("plumbing_hvac", C_PL)):
        d = tr_df[tr_df.trade == tr]
        for seg in (d[d.year <= 2016], d[d.year >= 2017]):
            ax.plot(seg.year, seg.pay_national / 1000, color=c)
        ax.annotate(LBL[tr].split(" (")[0],
                    (d.year.iloc[-1] + 0.4, d.pay_national.iloc[-1] / 1000),
                    color=c, fontsize=8, fontweight="bold", va="center")
    ax.axvline(BREAK, color=theme.RULE, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Average pay per employee, $000 nominal")
    theme.panel_tag(ax, "A")
    ax.set_xlim(1998, 2028)

    ax = axes[0, 1]
    for tr, c in (("electrical", C_EL), ("plumbing_hvac", C_PL)):
        d = tr_df[tr_df.trade == tr]
        for seg in (d[d.year <= 2016], d[d.year >= 2017]):
            ax.plot(seg.year, seg.p90_p10, color=c)
    ax.axvline(BREAK, color=theme.RULE, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cross-county p90/p10 of pay")
    theme.panel_tag(ax, "B")
    ax.set_xlim(1998, 2024)

    ax = axes[1, 0]
    for tr, c in (("electrical", C_EL), ("plumbing_hvac", C_PL)):
        d = tr_df[tr_df.trade == tr]
        for seg in (d[d.year <= 2016], d[d.year >= 2017]):
            ax.plot(seg.year, seg.premium_wtd * 100, color=c)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(BREAK, color=theme.RULE, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Wage premium, log points vs all industries")
    theme.panel_tag(ax, "C")
    ax.set_xlim(1998, 2024)

    # binscatter of the premium against consolidation after removing county
    # and year effects (Frisch-Waugh), pre-2017 regime only. The sample
    # filter matches the fixed-effects regression above (>=3 observations
    # per county), so by Frisch-Waugh-Lovell the slope equals the Table
    # estimate exactly.
    ax = axes[1, 1]
    d = panel[(panel.trade == "electrical") & panel.sh_lg.notna()
              & (panel.year <= 2016)].copy()
    d = d[d.groupby("fips").fips.transform("size") >= 3]

    def demean2(s):
        v = s.values.astype(float)
        for _ in range(12):
            v = v - pd.Series(v, index=d.index).groupby(d.fips)\
                .transform("mean").values
            v = v - pd.Series(v, index=d.index).groupby(d.year)\
                .transform("mean").values
        return v

    d["x"], d["y"] = demean2(d.sh_lg), demean2(d.premium)

    fw = smf.ols("y ~ x", data=d).fit(cov_type="cluster",
                                      cov_kwds={"groups": d.state})
    beta, se, pval = fw.params["x"], fw.bse["x"], fw.pvalues["x"]
    reg_rows.append(dict(trade="electrical", era="1998-2016",
                         spec="within (FWL, binscatter sample)",
                         beta_sh_lg=beta, se=se, t=fw.tvalues["x"], p=pval,
                         n=int(fw.nobs), counties=d.fips.nunique()))
    save_table(pd.DataFrame(reg_rows).set_index(["trade", "era"]),
               "tab_pay_premium_reg")

    w = d[d.x.between(*np.percentile(d.x, [1, 99]))].copy()
    w["bin"] = pd.qcut(w.x, 20, duplicates="drop")
    b = w.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
    ax.scatter(b.x * 100, b.y * 100, s=24, color=C_EL, zorder=3)
    xs = np.linspace(d.x.min(), d.x.max(), 50)
    ax.plot(xs * 100, (fw.params["Intercept"] + beta * xs) * 100,
            color=GRAY, linewidth=1.3)
    ax.axhline(0, color="#dcdcdc", linewidth=0.8)
    ax.axvline(0, color="#dcdcdc", linewidth=0.8)
    ax.set_xlabel("Large-establishment employment share (pp, within county)")
    ax.set_ylabel("Wage premium (log points, within county)")
    sig = "p = {:.2f}".format(pval) if pval >= 0.001 else "p < 0.001"
    theme.panel_tag(ax, "D")
    theme.note(ax, f"electrical, 1998-2016\nslope {beta:+.3f} ({sig})",
               (0.03, 0.97))

    fig.tight_layout()
    save_fig(fig, "fig09_pay_structure")
    plt.close(fig)


if __name__ == "__main__":
    main()
