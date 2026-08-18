"""Hierarchical location-scale model of the county wage premium:

    y[i]     ~ Normal( mu[i], sigma[i] )
    mu[i]    = a_county[c] + a_year[t] + b_mu  * x[i] + g_mu'  z[i]
    log s[i] = d_county[c] + d_year[t] + b_sig * x[i] + g_sig' z[i]

where y is the log wage premium and x the local share of trade employment
in establishments with 100 or more staff. Estimation is penalised maximum
likelihood by Adam (GPU if available); inference is a cluster bootstrap
over states with the same optimiser budget as the point estimate.

    python 14_location_scale_model.py             full run, 400 bootstrap fits
    python 14_location_scale_model.py --quick     40 fits, smoke test
    python 14_location_scale_model.py --fig-only  rebuild figure from tables
"""
import argparse
import time

import numpy as np
import pandas as pd

from config import PROCESSED, TABS, save_table, save_fig, MIN_EST_HARMONISED
import theme

plt = theme.apply()


def load(trade="electrical", era=(1998, 2016)):
    p = pd.read_parquet(PROCESSED / "cbp_pay_panel.parquet")
    d = p[(p.trade == trade) & p.sh_lg.notna()
          & p.year.between(*era) & (p.est >= MIN_EST_HARMONISED)].copy()
    d = d[d.groupby("fips").fips.transform("size") >= 5]
    d["y"] = d.premium.astype(float)
    d["x"] = d.sh_lg.astype(float)
    d["logemp"] = np.log(d.emp.astype(float))
    d["logest"] = np.log(d.est.astype(float).clip(lower=1))
    d["state"] = d.fips.str[:2]
    keep = ["y", "x", "logemp", "logest", "fips", "year", "state", "emp"]
    return d[keep].dropna().reset_index(drop=True)


def design(d):
    cidx, cuniq = pd.factorize(d.fips)
    tidx, tuniq = pd.factorize(d.year)
    Z = np.c_[d.logemp.values, d.logest.values]
    Z = (Z - Z.mean(0)) / Z.std(0)
    return (d.y.values.astype("float32"), d.x.values.astype("float32"),
            Z.astype("float32"), cidx.astype("int64"), tidx.astype("int64"),
            len(cuniq), len(tuniq))


def fit(y, x, Z, cidx, tidx, nC, nT, device, steps=1500, seed=0, verbose=False):
    import torch

    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    y_, x_, Z_ = t(y), t(x), t(Z)
    c_, t_ = t(cidx, torch.long), t(tidx, torch.long)
    k = Z.shape[1]

    P = {
        "a_c": torch.zeros(nC, device=device, requires_grad=True),
        "a_t": torch.zeros(nT, device=device, requires_grad=True),
        "d_c": torch.zeros(nC, device=device, requires_grad=True),
        "d_t": torch.zeros(nT, device=device, requires_grad=True),
        "b_mu": torch.zeros(1, device=device, requires_grad=True),
        "b_sig": torch.zeros(1, device=device, requires_grad=True),
        "g_mu": torch.zeros(k, device=device, requires_grad=True),
        "g_sig": torch.zeros(k, device=device, requires_grad=True),
        "c0": torch.zeros(1, device=device, requires_grad=True),
        "s0": torch.full((1,), -1.5, device=device, requires_grad=True),
        "lt_a": torch.zeros(1, device=device, requires_grad=True),
        "lt_d": torch.zeros(1, device=device, requires_grad=True),
    }
    opt = torch.optim.Adam(P.values(), lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for it in range(steps):
        opt.zero_grad()
        mu = (P["c0"] + P["a_c"][c_] + P["a_t"][t_]
              + P["b_mu"] * x_ + Z_ @ P["g_mu"])
        logs = (P["s0"] + P["d_c"][c_] + P["d_t"][t_]
                + P["b_sig"] * x_ + Z_ @ P["g_sig"]).clamp(-6, 3)
        nll = (logs + 0.5 * ((y_ - mu) / logs.exp()) ** 2).sum()
        ta, td = P["lt_a"].exp().clamp(1e-3, 10), P["lt_d"].exp().clamp(1e-3, 10)
        pen = (0.5 * (P["a_c"] ** 2).sum() / ta ** 2 + nC * P["lt_a"]
               + 0.5 * (P["d_c"] ** 2).sum() / td ** 2 + nC * P["lt_d"])
        loss = nll + pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 10.0)
        opt.step()
        sched.step()
        if verbose and it % 300 == 0:
            print(f"    it {it:5d}  loss {loss.item():,.1f}  "
                  f"b_mu {P['b_mu'].item():+.4f}  b_sig {P['b_sig'].item():+.4f}")
    return {k: v.detach().cpu().numpy().copy() for k, v in P.items()}


def cluster_bootstrap(d, B, device, steps, seed=0):
    rng = np.random.default_rng(seed)
    states = d.state.unique()
    idx_by_state = {s: np.flatnonzero(d.state.values == s) for s in states}
    out = []
    t0 = time.time()
    for b in range(B):
        pick = rng.choice(states, size=len(states), replace=True)
        rows = np.concatenate([idx_by_state[s] for s in pick])
        db = d.iloc[rows].reset_index(drop=True)
        try:
            P = fit(*design(db), device=device, steps=steps, seed=seed + b + 1)
            out.append((float(P["b_mu"][0]), float(P["b_sig"][0])))
        except Exception as e:
            print("    bootstrap draw failed:", e)
        if (b + 1) % max(1, B // 10) == 0:
            el = time.time() - t0
            print(f"    bootstrap {b + 1}/{B}  ({el / (b + 1):.1f}s per fit, "
                  f"{el / 60:.1f} min elapsed)")
    return np.array(out)


def implied_quantiles(b_mu, b_sig, sigma_bar, deltas=(0.10, 0.25)):
    from scipy.stats import norm
    rows = []
    for dlt in deltas:
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            z = norm.ppf(q)
            rows.append(dict(delta_share_pp=dlt * 100, q=q,
                             effect_log_points=(b_mu + z * sigma_bar * b_sig)
                             * dlt * 100))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--fig-only", action="store_true")
    a = ap.parse_args()

    if a.fig_only:
        res = pd.read_csv(TABS / "tab_dist_location_scale.csv")
        qq = pd.read_csv(TABS / "tab_dist_quantiles.csv")
        figure(res, qq)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    steps = 600 if a.quick else 1800
    B = a.boot if a.boot else (40 if a.quick else 400)

    results, qframes = [], []
    for trade in ("electrical", "plumbing_hvac"):
        for era_name, era in (("1998-2016", (1998, 2016)),
                              ("2017-2023", (2017, 2023))):
            d = load(trade, era)
            if len(d) < 2000:
                print(f"  {trade} {era_name}: only {len(d)} rows, skipped")
                continue
            print(f"\n  {trade} {era_name}: {len(d):,} county-years, "
                  f"{d.fips.nunique():,} counties, {d.state.nunique()} states")
            P = fit(*design(d), device=device, steps=steps, verbose=True)
            b_mu, b_sig = float(P["b_mu"][0]), float(P["b_sig"][0])
            sigma_bar = float(np.exp(P["s0"][0]))
            print(f"    point estimates  b_mu {b_mu:+.4f}   b_sig {b_sig:+.4f}"
                  f"   mean sigma {sigma_bar:.3f}")

            boot = cluster_bootstrap(d, B, device, steps)
            lo_mu, hi_mu = np.percentile(boot[:, 0], [2.5, 97.5])
            lo_sg, hi_sg = np.percentile(boot[:, 1], [2.5, 97.5])
            p_mu = 2 * min((boot[:, 0] > 0).mean(), (boot[:, 0] < 0).mean())
            p_sg = 2 * min((boot[:, 1] > 0).mean(), (boot[:, 1] < 0).mean())
            results.append(dict(
                trade=trade, era=era_name, n=len(d),
                counties=d.fips.nunique(), boot_draws=len(boot),
                beta_mean=b_mu, mean_ci_lo=lo_mu, mean_ci_hi=hi_mu, p_mean=p_mu,
                beta_logsd=b_sig, sd_ci_lo=lo_sg, sd_ci_hi=hi_sg, p_sd=p_sg,
                sigma_baseline=sigma_bar))
            q = implied_quantiles(b_mu, b_sig, sigma_bar)
            q["trade"], q["era"] = trade, era_name
            qframes.append(q)

    res = pd.DataFrame(results)
    save_table(res.set_index(["trade", "era"]), "tab_dist_location_scale")
    qq = pd.concat(qframes)
    save_table(qq.set_index(["trade", "era", "delta_share_pp", "q"]),
               "tab_dist_quantiles")
    figure(res, qq)


def figure(res, qq):
    fig = plt.figure(figsize=(6.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55,
                          wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    labels, ypos = [], []
    for i, (_, r) in enumerate(res.iterrows()):
        labels.append(f"{trade_label(r.trade)}  {r.era}")
        ypos.append(i)
    h = 0.18
    for i, (_, r) in enumerate(res.iterrows()):
        ax.plot([r.mean_ci_lo, r.mean_ci_hi], [i + h, i + h],
                color=theme.PETROL, lw=2.2, solid_capstyle="butt")
        ax.plot(r.beta_mean, i + h, "o", color=theme.PETROL, ms=5)
        ax.plot([r.sd_ci_lo, r.sd_ci_hi], [i - h, i - h],
                color=theme.OLIVE, lw=2.2, solid_capstyle="butt")
        ax.plot(r.beta_logsd, i - h, "o", color=theme.OLIVE, ms=5)
    ax.axvline(0, color=theme.RULE, lw=0.9)
    ax.set_yticks(ypos, labels)
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.GRID)
    theme.panel_tag(ax, "A")
    theme.note(ax, "blue: effect on the mean premium\ngreen: effect on log dispersion",
               (0.01, 0.04), ha="left", va="bottom")
    ax.set_xlabel("Coefficient on large-establishment employment share")

    ax = fig.add_subplot(gs[1, 0])
    sub = qq[(qq.delta_share_pp == 10)]
    for (tr, era), g in sub.groupby(["trade", "era"]):
        col = theme.PETROL if tr == "electrical" else theme.OLIVE
        ls = "-" if era.startswith("1998") else "--"
        ax.plot(g["q"] * 100, g.effect_log_points, ls, color=col, lw=1.8,
                marker="o", ms=3.5)
    theme.rule_line(ax, 0)
    ax.set_xlabel("Percentile of the county wage premium")
    ax.set_ylabel("Effect of a 10 pt rise, log points")
    theme.panel_tag(ax, "B")
    theme.note(ax, "solid 1998-2016, dashed 2017-2023", (0.02, 0.98))

    ax = fig.add_subplot(gs[1, 1])
    from scipy.stats import norm
    xs = np.linspace(-0.6, 0.9, 400)
    base = res.iloc[0]
    s0 = base.sigma_baseline
    ax.plot(xs, norm.pdf(xs, 0.16, s0), color=theme.INK_3, lw=1.6)
    ax.fill_between(xs, norm.pdf(xs, 0.16, s0), color=theme.WASH, zorder=0)
    s1 = s0 * np.exp(base.beta_logsd * 0.10)
    m1 = 0.16 + base.beta_mean * 0.10
    ax.plot(xs, norm.pdf(xs, m1, s1), color=theme.PETROL, lw=1.9)
    ax.set_yticks([])
    ax.set_xlabel("County wage premium, log points")
    theme.panel_tag(ax, "C")
    theme.note(ax, "grey: unconsolidated county\nblue: same county, share 10 points higher",
               (0.02, 0.98))
    theme.despine_all(ax)
    ax.grid(False)

    save_fig(fig, "fig10_dispersion")
    plt.close(fig)


def trade_label(t):
    return "Electrical" if t == "electrical" else "Plumbing/HVAC"


if __name__ == "__main__":
    main()
