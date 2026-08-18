"""Gradient-boosted model of county-level growth in electrical contractor
establishments, 2017-2023, from pre-2017 features. Validation is
leave-state-out; interpretation is SHAP with a permutation fallback."""
import numpy as np
import pandas as pd

from config import (load_cbp, save_table, save_fig, DIR_POP,
                    DIR_BPS, TRADE_CODES, SIZES, MID)
import theme

plt = theme.apply()
C_EL, GRAY = theme.PETROL, theme.INK_3

FEAT_LABEL = {
    "log_est16": "Market size 2016 (log estabs.)",
    "gr_9816": "Establishment growth 1998-2016",
    "log_est_pl16": "Plumbing/HVAC market size 2016",
    "log_ap16": "Trade payroll 2016 (log)",
    "sh_lg16": "Large-establishment share 2016",
    "log_pop16": "Population 2016 (log)",
    "pop_gr_10_16": "Population growth 2010-2016",
    "permits_1316": "Building permits 2013-2016",
    "pay16": "Average trade pay 2016",
}


def pop_features():
    frames = []
    specs = [("pop_2000s.csv", [f"POPESTIMATE{y}" for y in range(2000, 2010)]),
             ("pop_2010s.csv", [f"POPESTIMATE{y}" for y in range(2010, 2020)]),
             ("pop_2020s.csv", [f"POPESTIMATE{y}" for y in range(2020, 2025)])]
    for fname, cols in specs:
        f = DIR_POP / fname
        if not f.exists():
            continue
        df = pd.read_csv(f, encoding="latin-1", dtype={"STATE": str,
                                                       "COUNTY": str})
        if "STATE" not in df.columns:
            continue
        df["fips"] = df.STATE.str.zfill(2) + df.COUNTY.str.zfill(3)
        df = df[df.COUNTY != "000"]
        have = [c for c in cols if c in df.columns]
        frames.append(df.set_index("fips")[have])
    if not frames:
        return None
    pop = pd.concat(frames, axis=1)
    out = pd.DataFrame(index=pop.index)
    if "POPESTIMATE2016" in pop.columns:
        out["log_pop16"] = np.log(pop["POPESTIMATE2016"].clip(lower=1))
    if {"POPESTIMATE2010", "POPESTIMATE2016"} <= set(pop.columns):
        out["pop_gr_10_16"] = (np.log(pop.POPESTIMATE2016.clip(lower=1))
                               - np.log(pop.POPESTIMATE2010.clip(lower=1)))
    return out


def permit_features():
    rows = []
    for y in range(2013, 2017):
        f = DIR_BPS / f"bps_co{y}.txt"
        if not f.exists():
            continue
        df = pd.read_csv(f, skiprows=2, header=None, dtype=str,
                         on_bad_lines="skip", encoding="latin-1")
        try:
            df = df.rename(columns={1: "st", 2: "cty"})
            df["fips"] = df.st.str.zfill(2) + df.cty.str.zfill(3)
            units = pd.to_numeric(df[7], errors="coerce")
            rows.append(pd.DataFrame({"fips": df.fips, "units": units,
                                      "year": y}))
        except Exception:
            continue
    if not rows:
        return None
    d = pd.concat(rows)
    return d.groupby("fips").units.mean().rename("permits_1316").to_frame()


def qcew_features():
    frames = []
    for f in (DIR_POP.parent / "qcew").glob("qcew_238210_2016.csv"):
        q = pd.read_csv(f, dtype={"area_fips": str})
        q = q[(q.own_code == 5) & (q.agglvl_code == 78)]
        frames.append(q.set_index("area_fips")[["avg_annual_pay"]]
                      .rename(columns={"avg_annual_pay": "pay16"}))
    return frames[0] if frames else None


def build_dataset():
    cbp = load_cbp()
    el = [k for k, v in TRADE_CODES.items() if v == "electrical"]
    pl = [k for k, v in TRADE_CODES.items() if v == "plumbing_hvac"]

    def yr(codes, year, col="est"):
        d = cbp[(cbp.year == year) & cbp.naics.isin(codes)]
        return d.groupby("fips")[col].sum()

    X = pd.DataFrame({
        "log_est16": np.log(yr(el, 2016).clip(lower=1)),
        "gr_9816": np.log(yr(el, 2016).clip(lower=1))
        - np.log(yr(el, 1998).clip(lower=1)),
        "log_est_pl16": np.log(yr(pl, 2016).clip(lower=1)),
        "log_ap16": np.log(yr(el, 2016, "ap").clip(lower=1)),
    })
    d16 = cbp[(cbp.year == 2016) & cbp.naics.isin(el)].groupby("fips")[SIZES].sum()
    empmid = d16.values * MID
    with np.errstate(invalid="ignore", divide="ignore"):
        X["sh_lg16"] = pd.Series(
            empmid[:, 5:].sum(1) / empmid.sum(1), index=d16.index)
    for f, feat in [(pop_features(), "pop"), (permit_features(), "bps"),
                    (qcew_features(), "qcew")]:
        if f is not None:
            X = X.join(f, how="left")
        else:
            print(f"  note: {feat} features unavailable")
    y = (np.log(yr(el, 2023).clip(lower=1))
         - np.log(yr(el, 2016).clip(lower=1))).rename("target")
    df = X.join(y, how="inner").replace([np.inf, -np.inf], np.nan)
    df = df[df.log_est16 >= np.log(3)].dropna(subset=["target"])
    df["state"] = [f[:2] for f in df.index]
    return df


def main():
    df = build_dataset()
    feats = [c for c in df.columns if c not in ("target", "state")]
    print(f"  dataset: {len(df)} counties, {len(feats)} features")

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error

    import shutil
    import subprocess
    gpu, gpu_name = False, "none"
    if shutil.which("nvidia-smi"):
        try:
            gpu_name = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            gpu = bool(gpu_name)
        except Exception:
            pass

    try:
        import xgboost as xgb
        backend = "xgboost"
    except ImportError:
        backend = "sklearn"
        print("  xgboost unavailable, using HistGradientBoostingRegressor")
    dev = {"device": "cuda"} if (gpu and backend == "xgboost") else {}

    def make_model(p):
        if backend == "xgboost":
            return xgb.XGBRegressor(objective="reg:squarederror",
                                    random_state=42, **dev, **p)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            random_state=42, max_iter=p.get("n_estimators", 400),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", None))

    GRID = [dict(n_estimators=n, learning_rate=lr, max_depth=md,
                 subsample=0.8, colsample_bytree=0.8)
            for n in (400, 900, 1500)
            for lr in (0.01, 0.03)
            for md in (3, 4, 6)]
    if backend != "xgboost":
        for g in GRID:
            g.pop("subsample", None)
            g.pop("colsample_bytree", None)
    gkf5 = GroupKFold(n_splits=5)
    best, best_r2 = None, -np.inf
    for p in GRID:
        oof = pd.Series(index=df.index, dtype=float)
        for tr, te in gkf5.split(df[feats], df.target, groups=df.state):
            m = make_model(p)
            m.fit(df[feats].iloc[tr], df.target.iloc[tr])
            oof.iloc[te] = m.predict(df[feats].iloc[te])
        r2 = r2_score(df.target, oof)
        if r2 > best_r2:
            best, best_r2 = p, r2
    print(f"  best params: {best} (search R2 = {best_r2:.3f})")

    gkf = GroupKFold(n_splits=10)
    oof = pd.Series(index=df.index, dtype=float)
    for tr, te in gkf.split(df[feats], df.target, groups=df.state):
        m = make_model(best)
        m.fit(df[feats].iloc[tr], df.target.iloc[tr])
        oof.iloc[te] = m.predict(df[feats].iloc[te])
    metrics = pd.DataFrame([{
        "model": backend, "cv": "leave-state-out (10 groups)",
        "r2_oof": r2_score(df.target, oof),
        "mae_oof": mean_absolute_error(df.target, oof),
        "n": len(df),
        "best_params": str(best)}]).set_index("model")
    save_table(metrics, "tab_ml_metrics")

    final = make_model(best)
    final.fit(df[feats], df.target)

    shap_vals = None
    try:
        import shap
        expl = shap.TreeExplainer(final)
        shap_vals = expl.shap_values(df[feats])
        imp = pd.Series(np.abs(shap_vals).mean(0), index=feats)
    except Exception as e:
        print("  SHAP unavailable, permutation importance instead:", e)
        from sklearn.inspection import permutation_importance
        r = permutation_importance(final, df[feats], df.target,
                                   n_repeats=20, random_state=42)
        imp = pd.Series(r.importances_mean, index=feats)
    imp = imp.sort_values()
    imp_tab = imp.rename("importance").to_frame()
    imp_tab.insert(0, "feature_label",
                   [FEAT_LABEL.get(f, f) for f in imp_tab.index])
    save_table(imp_tab, "tab_ml_importance")

    fig = plt.figure(figsize=(6.5, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(df.target, oof, s=6, alpha=0.35, color=C_EL,
               edgecolors="none")
    lims = [-0.8, 0.9]
    ax.plot(lims, lims, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Actual log growth 2016-2023")
    ax.set_ylabel("Out-of-state-fold prediction")
    theme.panel_tag(ax, "A")
    theme.note(ax, f"leave-state-out R² = {metrics.r2_oof.iloc[0]:.2f}",
               (0.03, 0.97))

    ax = fig.add_subplot(gs[0, 1])
    ax.barh(range(len(imp)), imp.values, color=C_EL, height=0.62)
    import textwrap
    ax.set_yticks(range(len(imp)),
                  ["\n".join(textwrap.wrap(FEAT_LABEL.get(f, f), 17))
                   for f in imp.index], fontsize=6.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP|" if shap_vals is not None
                  else "permutation importance")
    theme.panel_tag(ax, "B")

    ax = fig.add_subplot(gs[1, :])
    top = imp.index[-1]
    if shap_vals is not None:
        j = feats.index(top)
        ax.scatter(df[top], shap_vals[:, j], s=6, alpha=0.35,
                   color=C_EL, edgecolors="none")
        ax.axhline(0, color=GRAY, linewidth=0.7)
        ax.set_xlabel(FEAT_LABEL.get(top, top))
        ax.set_ylabel("SHAP value")
    else:
        ax.scatter(df[top], df.target, s=6, alpha=0.35, color=C_EL,
                   edgecolors="none")
        ax.set_xlabel(FEAT_LABEL.get(top, top))
        ax.set_ylabel("target")
    theme.panel_tag(ax, "C")
    fig.tight_layout()
    save_fig(fig, "fig08_growth_model")
    plt.close(fig)


if __name__ == "__main__":
    main()
