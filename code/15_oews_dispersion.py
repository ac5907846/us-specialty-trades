"""Worker-level wage dispersion from OEWS hourly percentiles: national
trends for the trade occupations and a state panel regression of log
p90/p10 on the state large-establishment employment share."""
import re

import numpy as np
import pandas as pd

from config import (DIR_OEWS, PROCESSED, save_table, save_fig, load_cbp,
                    TRADE_CODES, SIZES, MID)
import theme

plt = theme.apply()

OCCS = {"47-2111": "Electricians",
        "47-2152": "Plumbers and pipefitters",
        "49-9021": "HVAC mechanics",
        "00-0000": "All occupations"}
PCTS = {"H_PCT10": "p10", "H_PCT25": "p25", "H_MEDIAN": "p50",
        "H_PCT75": "p75", "H_PCT90": "p90"}

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


def read_oews_file(path):
    """Return rows for the tracked occupations from one OEWS spreadsheet."""
    try:
        raw = pd.read_excel(path, header=None, nrows=8)
    except Exception:
        return None
    hdr = None
    for i in range(len(raw)):
        vals = [str(v).upper().strip() for v in raw.iloc[i]]
        if any(v in ("OCC_CODE", "OCC CODE") for v in vals):
            hdr = i
            break
    if hdr is None:
        return None
    df = pd.read_excel(path, header=hdr)
    df.columns = [str(c).upper().strip().replace(" ", "_") for c in df.columns]
    if "OCC_CODE" not in df.columns:
        return None
    keep = df[df.OCC_CODE.isin(OCCS)].copy()
    if keep.empty:
        return None
    for src in list(PCTS) + ["H_MEAN", "TOT_EMP", "A_MEDIAN"]:
        if src in keep.columns:
            keep[src] = pd.to_numeric(
                keep[src].astype(str).str.replace(r"[#*,$]", "", regex=True),
                errors="coerce")
    if "ST" in keep.columns:
        keep["geo"] = keep["ST"].astype(str)
    elif "PRIM_STATE" in keep.columns:
        keep["geo"] = keep["PRIM_STATE"].astype(str)
    elif "STATE" in keep.columns:
        keep["geo"] = keep["STATE"].astype(str)
    else:
        keep["geo"] = "US"
    return keep


def year_from_name(name):
    m = re.search(r"(?:oesm?|_m?)((?:19|20)?\d{2})", name.lower())
    if not m:
        return None
    y = int(m.group(1))
    return y if y > 1900 else (1900 + y if y > 90 else 2000 + y)


def build_panel():
    rows = []
    for sub in sorted(DIR_OEWS.glob("*")):
        if not sub.is_dir():
            continue
        year = year_from_name(sub.name)
        scope = "state" if "st" in sub.name.lower()[-4:] else "national"
        for f in sorted(sub.glob("*.xls*")):
            t = read_oews_file(f)
            if t is None:
                continue
            for _, r in t.iterrows():
                rec = {"year": year, "scope": scope, "occ": r.OCC_CODE,
                       "geo": r.get("geo", "US"),
                       "emp": r.get("TOT_EMP", np.nan)}
                for src, name in PCTS.items():
                    rec[name] = r.get(src, np.nan)
                rows.append(rec)
    d = pd.DataFrame(rows).dropna(subset=["year"])
    d = d[d.p10.notna() & d.p90.notna() & (d.p10 > 0)]
    d["p90_p10"] = d.p90 / d.p10
    d["p90_p50"] = d.p90 / d.p50
    d["p50_p10"] = d.p50 / d.p10
    d = d.drop_duplicates(subset=["year", "scope", "occ", "geo"])
    d.to_parquet(PROCESSED / "oews_percentiles.parquet")
    return d


def state_consolidation():
    cbp = load_cbp()
    t = cbp[cbp.naics.isin(TRADE_CODES)].copy()
    t["state_fips"] = t.fips.str[:2]
    g = t.groupby(["state_fips", "year"])[SIZES].sum()
    empmid = g.values * MID
    with np.errstate(invalid="ignore", divide="ignore"):
        share = empmid[:, 5:].sum(1) / empmid.sum(1)
    out = g.reset_index()[["state_fips", "year"]]
    out["sh_lg_state"] = share
    return out


def state_panel_regression(d):
    import statsmodels.formula.api as smf
    st = d[(d.scope == "state") & (d.occ != "00-0000")].copy()
    st["usps"] = st.geo.str.upper().str[:2]
    cons = state_consolidation()
    cons["usps"] = cons.state_fips.map(FIPS_TO_USPS)
    m = st.merge(cons, on=["usps", "year"], how="inner")
    m["log9010"] = np.log(m.p90_p10)
    m = m.dropna(subset=["log9010", "sh_lg_state"])
    rows = []
    for occ in ("47-2111", "47-2152", "49-9021"):
        sub = m[m.occ == occ]
        if len(sub) < 300:
            continue
        fit = smf.ols("log9010 ~ sh_lg_state + C(usps) + C(year)", data=sub)\
            .fit(cov_type="cluster", cov_kwds={"groups": sub.usps})
        rows.append(dict(occ=OCCS[occ], n=int(fit.nobs),
                         states=sub.usps.nunique(),
                         years=f"{sub.year.min():.0f}-{sub.year.max():.0f}",
                         beta=fit.params["sh_lg_state"],
                         se=fit.bse["sh_lg_state"],
                         t=fit.tvalues["sh_lg_state"],
                         p=fit.pvalues["sh_lg_state"]))
    res = pd.DataFrame(rows)
    save_table(res.set_index("occ"), "tab_oews_state_panel")
    return res, m


def figure(nat, res, m):
    fig = plt.figure(figsize=(6.5, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3)
    colors = {"47-2111": theme.PETROL, "47-2152": theme.OLIVE,
              "49-9021": theme.CLAY, "00-0000": theme.INK_3}

    ax = fig.add_subplot(gs[0, 0])
    dy = {"47-2111": -0.9, "47-2152": +0.9, "49-9021": 0.0, "00-0000": 0.0}
    for occ, g in nat.groupby("occ"):
        g = g.sort_values("year")
        ls = ":" if occ == "00-0000" else "-"
        ax.plot(g.year, g.p50, ls, color=colors[occ], lw=1.8)
        theme.label_end(ax, g.year.iloc[-1], g.p50.iloc[-1] + dy[occ],
                        OCCS[occ].split()[0], colors[occ], dx=0.4, size=7.2)
    ax.set_ylabel("Median hourly wage, $, national")
    theme.panel_tag(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    for occ, g in nat.groupby("occ"):
        g = g.sort_values("year")
        ls = ":" if occ == "00-0000" else "-"
        ax.plot(g.year, g.p90_p10, ls, color=colors[occ], lw=1.8)
    ax.set_ylabel("Hourly p90/p10 within occupation")
    theme.panel_tag(ax, "B")
    theme.note(ax, "dotted: all occupations", (0.02, 0.98))

    ax = fig.add_subplot(gs[1, 0])
    sub = m[m.occ == "47-2111"].copy()
    if len(sub):
        def dm(v, by):
            return v - v.groupby(by).transform("mean")
        sub["xd"] = dm(sub.sh_lg_state, sub.usps) \
            - dm(sub.sh_lg_state, sub.usps).groupby(sub.year).transform("mean")
        sub["yd"] = dm(sub.log9010, sub.usps) \
            - dm(sub.log9010, sub.usps).groupby(sub.year).transform("mean")
        q = pd.qcut(sub.xd, 16, duplicates="drop")
        b = sub.groupby(q, observed=True)[["xd", "yd"]].mean()
        ax.scatter(b.xd * 100, b.yd * 100, s=22, color=theme.PETROL, zorder=3)
        z = np.polyfit(sub.xd, sub.yd, 1)
        xs = np.linspace(sub.xd.min(), sub.xd.max(), 40)
        ax.plot(xs * 100, np.polyval(z, xs) * 100, color=theme.INK_3, lw=1.3)
        theme.rule_line(ax, 0)
    ax.set_xlabel("Large-establishment share, pp within state")
    ax.set_ylabel("log p90/p10, within state")
    theme.panel_tag(ax, "C")
    theme.note(ax, "electricians", (0.03, 0.97))

    ax = fig.add_subplot(gs[1, 1])
    if len(res):
        for i, (_, r) in enumerate(res.iterrows()):
            lo, hi = r.beta - 1.96 * r.se, r.beta + 1.96 * r.se
            ax.plot([lo, hi], [i, i], color=theme.PETROL, lw=2.2,
                    solid_capstyle="butt")
            ax.plot(r.beta, i, "o", color=theme.PETROL, ms=5)
        short = {"Electricians": "Electricians",
                 "Plumbers and pipefitters": "Plumbers",
                 "HVAC mechanics": "HVAC"}
        ax.set_yticks(range(len(res)), [short.get(o, o) for o in res.occ])
        ax.axvline(0, color=theme.RULE, lw=0.9)
        ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect on log p90/p10, state and year FE")
    theme.panel_tag(ax, "D")

    save_fig(fig, "fig11_worker_dispersion")
    plt.close(fig)


def main():
    if any(DIR_OEWS.glob("*/")):
        d = build_panel()
    elif (PROCESSED / "oews_percentiles.parquet").exists():
        d = pd.read_parquet(PROCESSED / "oews_percentiles.parquet")
        print("  using saved oews_percentiles.parquet")
    else:
        raise SystemExit("no OEWS files found; run 03_download_oews.py first")
    print(f"  parsed {len(d):,} occupation-geography-year rows, "
          f"years {d.year.min():.0f}-{d.year.max():.0f}")
    nat = d[d.scope == "national"].sort_values("year")
    trend = nat.pivot_table(index="year", columns="occ",
                            values=["p50", "p90_p10"])
    save_table(trend.round(3), "tab_oews_dispersion_trends")
    res, m = state_panel_regression(d)
    if len(res):
        print(res.round(4).to_string(index=False))
    figure(nat, res, m)


if __name__ == "__main__":
    main()
