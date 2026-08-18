"""Figures 1, 3, and 4: industry overview, business dynamism, geographic
concentration. Figure 2 is produced by 06_tail_estimation.py."""
import numpy as np
import pandas as pd

from config import save_fig, PROCESSED, SHORT
import theme

plt = theme.apply()

TRADE_COLOR = {"electrical": theme.PETROL, "plumbing_hvac": theme.OLIVE}

RECESSIONS = [(1980.0, 1980.6), (1981.5, 1982.9), (1990.6, 1991.2),
              (2001.2, 2001.9), (2007.9, 2009.5), (2020.1, 2020.4)]


def shade_recessions(ax):
    for x0, x1 in RECESSIONS:
        ax.axvspan(x0, x1, color=theme.WASH, zorder=0)


def fig01_firms_and_shares(tr):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axes[0]
    for t, c in TRADE_COLOR.items():
        d = tr[tr.trade == t]
        ax.plot(d.year, d.firms / 1000, color=c, lw=2.0)
        theme.label_end(ax, d.year.iloc[-1], d.firms.iloc[-1] / 1000,
                        SHORT[t], c, dx=0.4, size=7.6)
    ax.set_ylabel("Employer firms, thousands")
    theme.panel_tag(ax, "A")
    ax.set_xlim(1998, 2033)

    ax = axes[1]
    for t, c in TRADE_COLOR.items():
        d = tr[tr.trade == t]
        ax.plot(d.year, d.sh_500p * 100, color=c, lw=2.0)
        ax.plot(d.year, d.sh_lt20 * 100, color=c, ls="--", lw=1.3)
    theme.note(ax, "dashed: firms with under 20 employees\n"
                   "solid: firms with 500 or more", (0.03, 0.60))
    ax.set_ylim(5, 47)
    ax.set_ylabel("Share of trade employment, %")
    theme.panel_tag(ax, "B")
    ax.set_xlim(1998, 2023)
    save_fig(fig, "fig01_firms_and_size_shares")
    plt.close(fig)


def fig03_dynamism():
    spec = pd.read_parquet(PROCESSED / "bds_specialty_4digit.parquet")
    bench = pd.read_parquet(PROCESSED / "bds_benchmarks.parquet")
    fa = pd.read_parquet(PROCESSED / "bds_specialty_fa.parquet")
    fz = pd.read_parquet(PROCESSED / "bds_specialty_fz.parquet")

    fig = plt.figure(figsize=(6.5, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.0, 0.7],
                          hspace=0.5)

    ax = fig.add_subplot(gs[0])
    shade_recessions(ax)
    b = spec[spec.vcnaics4.astype(str) == "2382"].sort_values("year")
    econ = bench[bench.level == "economy"].sort_values("year")
    ax.plot(b.year, b.estabs_entry_rate, color=theme.PETROL, lw=2.0)
    ax.plot(b.year, b.estabs_exit_rate, color=theme.OLIVE, lw=2.0)
    ax.plot(econ.year, econ.estabs_entry_rate, color=theme.INK_3, lw=1.2,
            ls=":")
    theme.label_end(ax, 2023, b.estabs_entry_rate.iloc[-1] + 0.55,
                    "Entry, 2382", theme.PETROL, dx=0.7, size=7.4)
    theme.label_end(ax, 2023, b.estabs_exit_rate.iloc[-1] - 0.35,
                    "Exit, 2382", theme.OLIVE, dx=0.7, size=7.4)
    theme.label_end(ax, 2023, econ.estabs_entry_rate.iloc[-1] + 1.45,
                    "Entry, all sectors", theme.INK_3, dx=0.7, size=7.4)
    theme.note(ax, "shaded bands: NBER recessions", (0.012, 0.06),
               va="bottom")
    ax.set_ylabel("% of establishments")
    ax.set_xlim(1978, 2032)
    theme.panel_tag(ax, "A")

    ax = fig.add_subplot(gs[1])
    g = fa[fa.vcnaics4.astype(str) == "2382"].copy()
    g["firms"] = pd.to_numeric(g.firms, errors="coerce")
    g["fage"] = g.fage.replace("l) Left Censored", "k) 26+")
    order = ["a) 0", "b) 1", "c) 2", "d) 3", "e) 4", "f) 5",
             "g) 6 to 10", "h) 11 to 15", "i) 16 to 20", "j) 21 to 25",
             "k) 26+"]
    nice = ["0", "1", "2", "3", "4", "5", "6-10", "11-15", "16-20",
            "21-25", "26+"]
    piv = g.pivot_table(index="fage", columns="year", values="firms",
                        aggfunc="sum").reindex(order)
    sh = (piv / piv.sum(axis=0) * 100).loc[:, piv.columns >= 1982]
    im = ax.pcolormesh(sh.columns, np.arange(len(sh)), sh.values,
                       cmap=theme.SEQ, vmin=0,
                       vmax=np.nanmax(sh.values), shading="nearest")
    ax.set_yticks(np.arange(len(sh)), nice, fontsize=7)
    ax.set_ylabel("Firm age, years")
    ax.invert_yaxis()
    ax.grid(False)
    theme.despine_all(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.015, aspect=24)
    cb.set_label("Share of firms, %", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax.axvline(2008.5, color=theme.CLAY, lw=1.2, ls=(0, (4, 2)))
    ax.text(2007.6, 0.35, "2008: firms 16+ overtake firms 0-5",
            fontsize=7.0, color=theme.SIENNA, va="center", ha="right")
    theme.panel_tag(ax, "B")

    ax = fig.add_subplot(gs[2])
    z = fz[fz.vcnaics4.astype(str) == "2382"].copy()
    for c in ("firms", "firmdeath_firms"):
        z[c] = pd.to_numeric(z[c], errors="coerce")
    grp = {"a) 1 to 4": 0, "b) 5 to 9": 0, "c) 10 to 19": 0,
           "d) 20 to 99": 1, "e) 100 to 499": 1}
    z["g"] = z.fsize.map(grp).fillna(2)
    eras = [(1980, 1989, "1980s"), (1990, 1999, "1990s"),
            (2000, 2009, "2000s"), (2010, 2019, "2010s"),
            (2017, 2023, "2017-23")]
    names = [e[2] for e in eras]
    rate = np.zeros((3, len(eras)))
    deaths = np.zeros((3, len(eras)))
    for j, (y0, y1, _) in enumerate(eras):
        w = z[z.year.between(y0, y1)]
        a = w.groupby("g")[["firms", "firmdeath_firms"]].sum()
        for gg in (0, 1, 2):
            if gg in a.index and a.loc[gg, "firms"] > 0:
                rate[gg, j] = 100 * a.loc[gg, "firmdeath_firms"] \
                    / a.loc[gg, "firms"]
                deaths[gg, j] = a.loc[gg, "firmdeath_firms"] / (y1 - y0 + 1)
    ax.pcolormesh(np.arange(len(eras) + 1) - 0.5, np.arange(4) - 0.5,
                  rate, cmap=theme.SEQ, vmin=0, vmax=10,
                  shading="flat", edgecolors="white", linewidth=2)
    for gg in range(3):
        for j in range(len(eras)):
            dark = rate[gg, j] > 5.5
            ax.text(j, gg - 0.14, f"{rate[gg, j]:.1f}%", ha="center",
                    va="center", fontsize=7.6, fontweight="bold",
                    color="white" if dark else theme.INK)
            if deaths[gg, j] >= 1000:
                dtxt = f"{deaths[gg, j] / 1000:.1f}k/yr"
            elif deaths[gg, j] >= 1:
                dtxt = f"{deaths[gg, j]:,.0f}/yr"
            else:
                dtxt = "0/yr"
            ax.text(j, gg + 0.24, dtxt, ha="center", va="center",
                    fontsize=6.0, color="white" if dark else theme.INK_3)
    ax.set_xticks(range(len(names)), names, fontsize=7)
    ax.set_yticks([0, 1, 2], ["under 20", "20-499", "500+"], fontsize=7)
    ax.set_ylabel("Firm size, employees")
    ax.invert_yaxis()
    ax.grid(False)
    theme.despine_all(ax)
    theme.note(ax, "cell: annual death rate; deaths per year below",
               (0.0, -0.24), va="top", size=6.6)
    theme.panel_tag(ax, "C")

    save_fig(fig, "fig03_dynamism")
    plt.close(fig)


def fig04_concentration():
    cm = pd.read_csv(PROCESSED / "cbp_concentration.csv")
    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, :])
    for t, c in TRADE_COLOR.items():
        h = cm[(cm.trade == t) & (cm.basis == "harmonised")].sort_values("year")
        p = cm[(cm.trade == t) & (cm.basis == "published")].sort_values("year")
        ax.plot(p[p.year <= 2016].year, p[p.year <= 2016].top20 * 100,
                color=c, alpha=0.30, lw=1.1)
        ax.plot(h.year, h.top20 * 100, color=c, lw=2.0)
        theme.label_end(ax, 2023, h.top20.iloc[-1] * 100, SHORT[t], c,
                        dx=0.6, size=7.4)
    theme.note(ax, "faint: as published,\nbefore harmonisation", (0.03, 0.28))
    ax.set_ylabel("Top-20 county share, %")
    ax.set_xlim(1998, 2034)
    theme.panel_tag(ax, "A")

    ax = fig.add_subplot(gs[1, 0])
    for t, c in TRADE_COLOR.items():
        h = cm[(cm.trade == t) & (cm.basis == "harmonised")].sort_values("year")
        ax.plot(h.year, h.top100 * 100, color=c, lw=2.0)
        theme.label_end(ax, 2023, h.top100.iloc[-1] * 100, SHORT[t], c,
                        dx=0.6, size=7.4)
    ax.set_ylabel("Top-100 county share, %")
    ax.set_xlim(1998, 2034)
    theme.panel_tag(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    if "hhi_county" in cm.columns:
        for t, c in TRADE_COLOR.items():
            h = cm[(cm.trade == t)
                   & (cm.basis == "harmonised")].sort_values("year")
            ax.plot(h.year, h.hhi_county, color=c, lw=2.0)
            theme.label_end(ax, 2023, h.hhi_county.iloc[-1], SHORT[t], c,
                            dx=0.6, size=7.4)
        ax.set_xlim(1998, 2034)
    ax.set_ylabel("County Herfindahl index")
    theme.panel_tag(ax, "C")

    save_fig(fig, "fig04_geographic_concentration")
    plt.close(fig)


def main():
    tr = pd.read_csv(PROCESSED / "trade_size_trends.csv")
    fig01_firms_and_shares(tr)
    fig03_dynamism()
    fig04_concentration()


if __name__ == "__main__":
    main()
