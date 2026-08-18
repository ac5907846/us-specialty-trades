"""Spatial statistics: Moran's I and local indicators (LISA) of contractor
growth, with a growth choropleth and cluster map."""
import numpy as np
import pandas as pd

from config import load_cbp, save_table, save_fig, DIR_GEO, TRADE_CODES
import theme

plt = theme.apply()

EXCL = ["02", "15", "60", "66", "69", "72", "78"]


def growth_frame(cbp, trade, y0=2017, y1=2023):
    codes = [k for k, v in TRADE_CODES.items() if v == trade]
    d0 = cbp[(cbp.year == y0) & cbp.naics.isin(codes)].groupby("fips").est.sum()
    d1 = cbp[(cbp.year == y1) & cbp.naics.isin(codes)].groupby("fips").est.sum()
    df = pd.DataFrame({"e0": d0, "e1": d1}).dropna()
    df = df[df.e0 >= 3]
    df["growth"] = np.log(df.e1.clip(lower=1)) - np.log(df.e0)
    return df


def main():
    import geopandas as gpd
    from libpysal.weights import Queen
    from esda.moran import Moran, Moran_Local

    cbp = load_cbp()
    shp = sorted(DIR_GEO.glob("cb_*_us_county_20m.zip"))[-1]
    gdf = gpd.read_file(f"zip://{shp}")
    gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    gdf = gdf[~gdf["STATEFP"].isin(EXCL)].to_crs(5070)

    rows = []
    lisa_maps = {}
    for trade in ("electrical", "plumbing_hvac"):
        df = growth_frame(cbp, trade)
        g = gdf.merge(df, left_on="fips", right_index=True, how="inner")
        w = Queen.from_dataframe(g, use_index=False, silence_warnings=True)
        w.transform = "r"
        mi = Moran(g["growth"].values, w, permutations=9999)
        rows.append(dict(trade=trade, moran_I=mi.I, p_sim=mi.p_sim,
                         n=len(g)))
        ml = Moran_Local(g["growth"].values, w, permutations=9999, seed=42)
        g["lisa_q"] = ml.q
        g["lisa_sig"] = ml.p_sim < 0.05
        lisa_maps[trade] = g
    save_table(pd.DataFrame(rows).set_index("trade"), "tab_moran")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.6))
    for a in axes:
        a.set_anchor("W")

    g = lisa_maps["electrical"]
    ax = axes[0]
    gdf.plot(ax=ax, color=theme.WASH, edgecolor="white", linewidth=0.12)
    v = g["growth"].clip(-0.6, 0.6)
    g.plot(ax=ax, column=v, cmap=theme.DIV, vmin=-0.6, vmax=0.6,
           edgecolor="white", linewidth=0.12, legend=True,
           legend_kwds={"shrink": 0.55,
                        "label": "log change in establishments"})
    ax.set_axis_off()
    ax.set_title("(a)", fontsize=8.5, loc="left", fontweight="normal",
                 color=theme.INK)

    ax = axes[1]
    gdf.plot(ax=ax, color=theme.WASH, edgecolor="white", linewidth=0.12)
    colors = {1: theme.ROSE, 2: theme.PETROL_TINTS[0],
              3: theme.PETROL, 4: theme.ROSE_TINTS[0]}
    names = {1: "High-High (hot spot)", 2: "Low-High",
             3: "Low-Low (cold spot)", 4: "High-Low"}
    sig = g[g.lisa_sig]
    for q, c in colors.items():
        sub = sig[sig.lisa_q == q]
        if len(sub):
            sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, names.values(), loc="lower left", fontsize=7.5,
              frameon=False)
    ax.set_axis_off()
    ax.set_title("(b)", fontsize=8.5, loc="left", fontweight="normal",
                 color=theme.INK)

    fig.tight_layout()
    save_fig(fig, "fig06_growth_maps")
    plt.close(fig)


if __name__ == "__main__":
    main()
