"""Download county population estimates, building permits, cartographic
boundaries, and (best effort) QCEW industry wage files."""
import time
import urllib.request
import zipfile

from config import DIR_POP, DIR_BPS, DIR_GEO, ensure_dirs

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36",
      "Accept": "text/csv,text/plain,*/*"}


def get(url, dest, retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(dest, "wb") as f:
                f.write(r.read())
            print(f"  [ok] {dest.name}")
            return True
        except Exception as e:
            print(f"  [try {i}] {dest.name} -> {e}")
            time.sleep(2 * i)
    return False


def qcew(inds=("238210", "238220"), years=range(1998, 2025)):
    for y in years:
        outdir = DIR_POP.parent / "qcew"
        outdir.mkdir(exist_ok=True)
        need = [i for i in inds
                if not (outdir / f"qcew_{i}_{y}.csv").exists()]
        if not need:
            continue
        for i in list(need):
            if get(f"https://data.bls.gov/cew/data/api/{y}/A/industry/{i}.csv",
                   outdir / f"qcew_{i}_{y}.csv", retries=1):
                need.remove(i)
        if not need:
            continue
        zdest = outdir / f"_{y}.zip"
        if not get(f"https://data.bls.gov/cew/data/files/{y}/csv/"
                   f"{y}_annual_by_industry.zip", zdest, retries=2):
            continue
        try:
            with zipfile.ZipFile(zdest) as z:
                for i in need:
                    members = [m for m in z.namelist()
                               if i in m and m.endswith(".csv")]
                    if members:
                        with z.open(members[0]) as src, \
                                open(outdir / f"qcew_{i}_{y}.csv", "wb") as f:
                            f.write(src.read())
                        print(f"  [extract] qcew_{i}_{y}.csv")
        except Exception as e:
            print(f"  [zipfail] {zdest.name} -> {e}")
        finally:
            zdest.unlink(missing_ok=True)


def main():
    ensure_dirs()
    print("== county population estimates ==")
    for url, name in (
        ("https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
         "counties/totals/co-est2024-alldata.csv", "pop_2020s.csv"),
        ("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/"
         "counties/totals/co-est2019-alldata.csv", "pop_2010s.csv"),
        ("https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/"
         "intercensal/county/co-est00int-tot.csv", "pop_2000s.csv"),
    ):
        get(url, DIR_POP / name)

    print("== building permits, county, annual ==")
    for y in range(1998, 2025):
        get(f"https://www2.census.gov/econ/bps/County/co{y}a.txt",
            DIR_BPS / f"bps_co{y}.txt")

    print("== cartographic county boundaries ==")
    get("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
        "cb_2023_us_county_20m.zip", DIR_GEO / "cb_2023_us_county_20m.zip")
    get("https://www2.census.gov/geo/tiger/GENZ2020/shp/"
        "cb_2020_us_county_20m.zip", DIR_GEO / "cb_2020_us_county_20m.zip")

    print("== QCEW (optional; endpoints may be retired) ==")
    qcew()
    print("done.")


if __name__ == "__main__":
    main()
