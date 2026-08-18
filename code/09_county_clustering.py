"""K-means clustering of county establishment trajectories, 1998-2016, with
cluster profiles and a cluster map."""
import numpy as np
import pandas as pd

from config import (load_cbp, save_table, save_fig, PROCESSED,
                    DIR_GEO, TRADE_CODES)
import theme

plt = theme.apply()
PAL = [theme.SIENNA, theme.INK_3, theme.OLIVE, theme.PETROL]
GRAY = theme.RULE
YEARS = list(range(1998, 2017))


def county_matrix(cbp, trade="electrical"):
    codes = [k for k, v in TRADE_CODES.items() if v == trade]
    d = cbp[cbp.naics.isin(codes) & cbp.year.isin(YEARS)]
    m = d.pivot_table(index="fips", columns="year", values="est",
                      aggfunc="sum")
    m = m[(m.min(axis=1) >= 5)].dropna()
    return m


def main():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    cbp = load_cbp()
    m = county_matrix(cbp)
    X = np.log(m.values)
    X = StandardScaler().fit_transform(X.T).T
    scores = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    k = max(scores, key=scores.get)
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
    lab = pd.Series(km.labels_, index=m.index, name="cluster")
    print(f"  chosen k={k}, silhouette={scores[k]:.3f}")

    growth = (m[2016] / m[1998]).groupby(lab).median().sort_values()
    order = {old: new for new, old in enumerate(growth.index)}
    lab = lab.map(order)

    CLUSTER_NAMES = {0: "Declining", 1: "Boom-bust", 2: "Sustained growth"}
    prof = pd.DataFrame({
        "name": pd.Series({i: CLUSTER_NAMES.get(i, f"Cluster {i+1}")
                           for i in range(k)}),
        "counties": lab.value_counts().sort_index(),
        "median_est_1998": m[1998].groupby(lab).median(),
        "median_est_2016": m[2016].groupby(lab).median(),
        "median_growth": (m[2016] / m[1998]).groupby(lab).median(),
        "share_estabs_2016": m[2016].groupby(lab).sum() / m[2016].sum(),
    })
    save_table(prof, "tab_cluster_profiles")
    lab.to_frame().to_csv(PROCESSED / "county_clusters.csv")

    try:
        import geopandas as gpd
        shp = DIR_GEO / "cb_2020_us_county_20m.zip"
        gdf = gpd.read_file(f"zip://{shp}") if shp.exists() else None
    except Exception as e:
        print("  map skipped:", repr(e))
        gdf = None

    per_row = min(k, 3)
    ntraj_rows = int(np.ceil(k / per_row))
    nrows = ntraj_rows + (1 if gdf is not None else 0)
    heights = [1.0] * ntraj_rows + ([1.55] if gdf is not None else [])
    fig = plt.figure(figsize=(6.5, 2.3 * ntraj_rows
                              + (3.6 if gdf is not None else 0)))
    gs = fig.add_gridspec(nrows, per_row, height_ratios=heights,
                          wspace=0.34, hspace=0.45)

    for c in range(k):
        ax = fig.add_subplot(gs[c // per_row, c % per_row])
        idx = lab[lab == c].index
        sub = np.log(m.loc[idx])
        base = sub.sub(sub[1998], axis=0)
        med = base.median()
        p25, p75 = base.quantile(0.25), base.quantile(0.75)
        color = PAL[c % len(PAL)]
        ax.fill_between(YEARS, p25, p75, color=color, alpha=0.2, linewidth=0)
        ax.plot(YEARS, med, color=color)
        ax.axhline(0, color=GRAY, linewidth=0.7)
        nm = prof.name.iloc[c] if c < len(prof) else f"Cluster {c + 1}"
        theme.panel_tag(ax, "abcd"[c])
        ax.text(0.03, 0.94, f"{nm}\nn={len(idx)}, "
                f"×{prof.median_growth.iloc[c]:.2f}",
                transform=ax.transAxes, fontsize=7.4, color=color,
                fontweight="bold", va="top", linespacing=1.3)
        ax.set_ylim(-0.8, 0.9)

    if gdf is not None:
        axm = fig.add_subplot(gs[ntraj_rows, :])
        g = gdf.copy()
        g["fips"] = g["GEOID"].astype(str).str.zfill(5)
        g = g.merge(lab.rename("cluster"), left_on="fips",
                    right_index=True, how="left")
        cont = g[~g["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]
        cont = cont.to_crs(5070)
        cont.plot(ax=axm, color="#f0efe9", edgecolor="white", linewidth=0.15)
        for c in range(k):
            cont[cont.cluster == c].plot(ax=axm, color=PAL[c % len(PAL)],
                                         edgecolor="white", linewidth=0.15)
        axm.set_axis_off()
        axm.set_title(f"({chr(97 + k)})", fontsize=8.5, loc="left",
                      fontweight="normal", color=theme.INK)
    fig.tight_layout()
    save_fig(fig, "fig07_county_clusters")
    plt.close(fig)


if __name__ == "__main__":
    main()
