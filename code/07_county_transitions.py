"""County market-structure dynamics: Markov transition matrices, mobility
indices, ergodic distributions, and beta convergence."""
import numpy as np
import pandas as pd

from config import load_cbp, save_table, save_fig, TRADE_CODES
import theme

plt = theme.apply()

CATS = ["0", "1-4", "5-19", "20-99", "100+"]
EDGES = [0, 1, 5, 20, 100, np.inf]


def classify(est):
    return pd.cut(est.fillna(0), EDGES, labels=CATS, right=False)


def build_state(cbp, trade, year):
    d = cbp[(cbp.year == year) & (cbp.naics.isin(
        [k for k, v in TRADE_CODES.items() if v == trade]))]
    s = d.groupby("fips").est.sum()
    return classify(s)


def transition(cbp, trade, y0, y1):
    a, b = build_state(cbp, trade, y0), build_state(cbp, trade, y1)
    fips = a.index.union(b.index)
    a = a.reindex(fips).fillna("0")
    b = b.reindex(fips).fillna("0")
    M = pd.crosstab(a, b, normalize="index").reindex(
        index=CATS, columns=CATS).fillna(0)
    return M, len(fips)


def ergodic(M):
    vals, vecs = np.linalg.eig(M.values.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    v = np.abs(v)
    return pd.Series(v / v.sum(), index=M.index)


def shorrocks(M):
    k = len(M)
    return (k - np.trace(M.values)) / (k - 1)


def convergence(cbp, trade, y0, y1):
    """Beta convergence of log establishment counts across counties."""
    import statsmodels.formula.api as smf
    codes = [k for k, v in TRADE_CODES.items() if v == trade]
    d0 = cbp[(cbp.year == y0) & cbp.naics.isin(codes)].groupby("fips").est.sum()
    d1 = cbp[(cbp.year == y1) & cbp.naics.isin(codes)].groupby("fips").est.sum()
    df = pd.DataFrame({"e0": d0, "e1": d1}).dropna()
    df = df[(df.e0 >= 3) & (df.e1 >= 1)]
    df["g"] = (np.log(df.e1) - np.log(df.e0)) / (y1 - y0)
    df["l0"] = np.log(df.e0)
    df["st"] = [f[:2] for f in df.index]
    m = smf.ols("g ~ l0 + C(st)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df.st})
    return dict(trade=trade, era=f"{y0}-{y1}", beta=m.params["l0"],
                se=m.bse["l0"], t=m.tvalues["l0"], n=int(m.nobs),
                half_life=np.log(2) / abs(m.params["l0"])
                if m.params["l0"] < 0 else np.nan)


def main():
    cbp = load_cbp()
    eras = [(1998, 2007), (2007, 2016), (2017, 2023)]

    mats, mob_rows, conv_rows = {}, [], []
    for trade in ("electrical", "plumbing_hvac"):
        for (y0, y1) in eras:
            M, n = transition(cbp, trade, y0, y1)
            mats[(trade, f"{y0}-{y1}")] = M
            mob_rows.append(dict(trade=trade, era=f"{y0}-{y1}",
                                 counties=n, shorrocks=shorrocks(M),
                                 stay_100p=M.loc["100+", "100+"],
                                 stay_5_19=M.loc["5-19", "5-19"],
                                 stay_1_4=M.loc["1-4", "1-4"]))
            try:
                conv_rows.append(convergence(cbp, trade, y0, y1))
            except Exception as e:
                print("  convergence failed:", trade, y0, y1, e)
    mob = pd.DataFrame(mob_rows).set_index(["trade", "era"])
    save_table(mob, "tab_mobility")

    long = []
    for (trade, era), M in mats.items():
        t = M.copy()
        t.insert(0, "from", t.index)
        t.insert(0, "era", era)
        t.insert(0, "trade", trade)
        long.append(t)
    save_table(pd.concat(long).set_index(["trade", "era", "from"]),
               "tab_transition_matrices")
    if conv_rows:
        save_table(pd.DataFrame(conv_rows).set_index(["trade", "era"]),
                   "tab_convergence")

    erg = pd.DataFrame({t: ergodic(mats[(t, "2017-2023")])
                        for t in ("electrical", "plumbing_hvac")})
    save_table(erg, "tab_ergodic_2017_2023")

    fig = plt.figure(figsize=(6.5, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9], hspace=0.5,
                          wspace=0.36)
    letters = "ABC"
    for j, (y0, y1) in enumerate(eras):
        ax = fig.add_subplot(gs[0, j])
        M = mats[("electrical", f"{y0}-{y1}")]
        ax.imshow(M.values, cmap=theme.SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(5), CATS, fontsize=6)
        ax.set_yticks(range(5), CATS, fontsize=6)
        for r in range(5):
            for c in range(5):
                v = M.values[r, c]
                if v < 0.005:
                    continue
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if v > 0.55 else theme.INK)
        if j == 0:
            ax.set_ylabel("class in start year", fontsize=7)
        ax.set_xlabel("class in end year", fontsize=7)
        ax.grid(False)
        theme.despine_all(ax)
        theme.panel_tag(ax, letters[j])
        ax.text(1.0, 1.06, f"electrical, {y0}–{y1}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.6, color=theme.INK_2)

    ax = fig.add_subplot(gs[1, :])
    era_colors = [theme.PETROL_TINTS[0], theme.PETROL_TINTS[1], theme.PETROL]
    xpos = np.arange(len(CATS))
    off = [-0.22, 0.0, 0.22]
    for j, (y0, y1) in enumerate(eras):
        Me = mats[("electrical", f"{y0}-{y1}")]
        Mp = mats[("plumbing_hvac", f"{y0}-{y1}")]
        ax.scatter(xpos + off[j], np.diag(Me.values), s=34,
                   color=era_colors[j], zorder=3, label=f"{y0}-{y1}")
        ax.scatter(xpos + off[j], np.diag(Mp.values), s=34,
                   facecolors="none", edgecolors=era_colors[j],
                   linewidths=1.3, zorder=3)
    ax.set_xticks(xpos, CATS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(stay in class)")
    ax.set_xlabel("county size class, establishments")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncols=3)
    theme.note(ax, "filled: electrical   open: plumbing/HVAC", (0.01, 0.05),
               va="bottom")
    theme.panel_tag(ax, "D")

    save_fig(fig, "fig05_county_transitions")
    plt.close(fig)


if __name__ == "__main__":
    main()
