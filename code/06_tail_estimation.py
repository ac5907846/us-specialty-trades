"""Upper-tail exponent of the firm size distribution: discrete MLE on
binned SUSB counts, multinomial bootstrap intervals, OLS comparison, and
structural break detection."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from config import (susb_us_detailed, save_table, save_fig,
                    DETAILED_BINS, BIN_UPPER)
import theme

plt = theme.apply()
C_EL, C_PL, GRAY = theme.PETROL, theme.OLIVE, theme.INK_3
RNG = np.random.default_rng(42)
SMIN = 20


def bin_counts(us, year):
    g = us[us.year == year].set_index("lab")
    lo, hi, n = [], [], []
    for lab, l in DETAILED_BINS:
        if lab in g.index and pd.notna(g.loc[lab, "firms"]):
            lo.append(l)
            hi.append(BIN_UPPER[lab])
            n.append(float(g.loc[lab, "firms"]))
    return np.array(lo, float), np.array(hi, float), np.array(n, float)


def nll(alpha, lo, hi, n, smin=SMIN):
    p_lo = (lo / smin) ** (-alpha)
    p_hi = np.where(np.isinf(hi), 0.0, (hi / smin) ** (-alpha))
    p = np.clip(p_lo - p_hi, 1e-300, None)
    return -(n * np.log(p)).sum()


def alpha_mle(lo, hi, n):
    r = minimize_scalar(nll, bounds=(0.2, 4.0), args=(lo, hi, n),
                        method="bounded")
    return r.x


def alpha_ols(lo, n):
    cc = n[::-1].cumsum()[::-1]
    return -np.polyfit(np.log(lo), np.log(cc), 1)[0]


def bootstrap_ci(lo, hi, n, B=2000):
    N = int(n.sum())
    p = n / N
    draws = RNG.multinomial(N, p, size=B)
    a = np.array([alpha_mle(lo, hi, d.astype(float)) for d in draws])
    return np.percentile(a, [2.5, 97.5])


def binseg(sig, min_seg=4, max_breaks=2):
    """Mean-shift binary segmentation with BIC selection."""
    def sse(x):
        return ((x - x.mean()) ** 2).sum() if len(x) else 0.0

    def best_split(lo, hi):
        cands = [(sse(sig[lo:k]) + sse(sig[k:hi]), k)
                 for k in range(lo + min_seg, hi - min_seg + 1)]
        return min(cands) if cands else (np.inf, None)

    n = len(sig)
    segs, bks = [(0, n)], []
    for _ in range(max_breaks):
        options = [(best_split(lo, hi), (lo, hi)) for lo, hi in segs]
        (gain_sse, k), (lo, hi) = min(options)
        if k is None:
            break
        cur = sum(sse(sig[a:b]) for a, b in segs)
        new = cur - sse(sig[lo:hi]) + gain_sse
        if n * np.log(max(new, 1e-12) / n) + \
           (len(bks) + 2) * np.log(n) < \
           n * np.log(max(cur, 1e-12) / n) + (len(bks) + 1) * np.log(n):
            bks.append(k)
            segs.remove((lo, hi))
            segs += [(lo, k), (k, hi)]
        else:
            break
    return sorted(bks)


def main():
    results = []
    ccdf_store = {}
    for trade, color in [("electrical", C_EL), ("plumbing_hvac", C_PL)]:
        us = susb_us_detailed(trade)
        for year in sorted(us.year.unique()):
            lo, hi, n = bin_counts(us, year)
            if len(lo) < 10:
                continue
            a_mle = alpha_mle(lo, hi, n)
            lo95, hi95 = bootstrap_ci(lo, hi, n)
            results.append(dict(trade=trade, year=year, alpha_mle=a_mle,
                                ci_lo=lo95, ci_hi=hi95,
                                alpha_ols=alpha_ols(lo, n),
                                n_firms_tail=int(n.sum())))
            ccdf_store[(trade, year)] = (lo, n[::-1].cumsum()[::-1])
    res = pd.DataFrame(results)
    save_table(res.set_index(["trade", "year"]), "tab_alpha_by_year")

    breaks = []
    for trade in ("electrical", "plumbing_hvac"):
        d = res[res.trade == trade].sort_values("year")
        sig = d.alpha_mle.values
        try:
            import ruptures as rpt
            bk = rpt.Pelt(model="rbf").fit(sig.reshape(-1, 1)).predict(pen=1.0)
            yrs = [int(d.year.values[i - 1]) for i in bk[:-1]]
            method = "ruptures-PELT"
        except Exception:
            yrs = [int(d.year.values[k]) for k in binseg(sig)]
            method = "binseg-BIC"
        breaks.append(dict(trade=trade, method=method, break_years=str(yrs)))
    save_table(pd.DataFrame(breaks).set_index("trade"), "tab_alpha_breaks")

    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2))

    ax = axes[0, 0]
    for trade, color in [("electrical", C_EL), ("plumbing_hvac", C_PL)]:
        d = res[res.trade == trade].sort_values("year")
        ax.plot(d.year, d.alpha_mle, color=color, marker="o", markersize=3.5)
        ax.fill_between(d.year, d.ci_lo, d.ci_hi, color=color, alpha=0.18,
                        linewidth=0)
    ax.axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
    ax.annotate("Zipf benchmark (α = 1)", (res.year.min() + 0.3, 1.02),
                color=GRAY, fontsize=7.5)
    theme.panel_tag(ax, "A")
    ax.set_ylabel("Tail exponent α")

    ax = axes[0, 1]
    d = res[res.trade == "electrical"].sort_values("year")
    ax.plot(d.year, d.alpha_mle, color=C_EL, label="MLE")
    ax.plot(d.year, d.alpha_ols, color=C_EL, linestyle=":",
            linewidth=1.4, label="OLS on CCDF")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Tail exponent α, electrical")
    theme.panel_tag(ax, "B")

    ax = axes[1, 0]
    yrs = [y for (t, y) in ccdf_store if t == "electrical"]
    pick = [min(yrs), 2011 if 2011 in yrs else yrs[len(yrs) // 2], max(yrs)]
    shades = [theme.PETROL_TINTS[0], theme.PETROL_TINTS[1], theme.PETROL]
    for y, c in zip(pick, shades):
        lo, cc = ccdf_store[("electrical", y)]
        ax.loglog(lo, cc, "o-", color=c, markersize=3.5, label=str(y))
    ax.legend(frameon=False, fontsize=8, title="electrical",
              title_fontsize=7.5)
    theme.panel_tag(ax, "C")
    ax.set_xlabel("Firm size s")
    ax.set_ylabel("Firms ≥ s")

    ax = axes[1, 1]
    d = res[res.trade == "electrical"].sort_values("year")
    n500 = []
    for y in d.year:
        us = susb_us_detailed("electrical")
        g = us[us.year == y].set_index("lab")
        big = [l for l, thr in DETAILED_BINS if thr >= 500 and l in g.index]
        n500.append(g.loc[big, "firms"].sum())
    ax.plot(d.year, n500, color=C_EL, marker="o", markersize=3.5)
    ax.set_ylabel("Firms with 500+ employees, electrical")
    theme.panel_tag(ax, "D")

    for a in (axes[0, 0], axes[0, 1], axes[1, 1]):
        a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.tight_layout()
    save_fig(fig, "fig02_upper_tail")
    plt.close(fig)


if __name__ == "__main__":
    main()
