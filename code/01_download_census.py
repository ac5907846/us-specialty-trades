"""Download the Census sources (BDS, SUSB, Economic Census, CBP) and
pre-filter CBP to construction. Already-downloaded files are skipped, so the
script can be re-run to fill gaps."""
import csv
import gzip
import io
import os
import re
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from config import DIR_BDS, DIR_SUSB, DIR_EC, DIR_CBP, INTERIM, ensure_dirs

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"}


def fetch(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=timeout)


def download(url, dest, max_mb=400, retries=3):
    dest = os.fspath(dest)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 0 and not (dest.endswith(".zip") and size < 5000):
            print(f"  [have] {os.path.basename(dest)}")
            return True
        os.remove(dest)
    for attempt in range(1, retries + 1):
        try:
            with fetch(url) as r:
                total, tmp = 0, dest + ".part"
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(524288)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_mb * 1024 * 1024:
                            raise RuntimeError(f"exceeds {max_mb} MB")
                        f.write(chunk)
                os.replace(tmp, dest)
            print(f"  [ok]   {os.path.basename(dest)}  ({total / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  [try {attempt}] {os.path.basename(dest)} -> {e}")
            time.sleep(2 * attempt)
    return False


def list_dir(url):
    try:
        with fetch(url, timeout=60) as r:
            html = r.read().decode("utf-8", "ignore")
        return [h for h in re.findall(r'href="([^"?/][^"]*)"', html)
                if not h.startswith("http")]
    except Exception as e:
        print(f"  [listfail] {url} -> {e}")
        return []


def do_bds():
    print("\n=== BDS (2023 vintage) ===")
    base = "https://www2.census.gov/programs-surveys/bds/tables/time-series/2023/"
    for f in ("bds2023.csv", "bds2023_sec.csv", "bds2023_st_sec.csv",
              "bds2023_vcn4.csv", "bds2023_vcn4_fa.csv", "bds2023_vcn4_fz.csv"):
        download(base + f, DIR_BDS / f, max_mb=150)


def do_susb():
    print("\n=== SUSB 1998-2022 ===")
    for year in range(1998, 2023):
        got = False
        for sub in (f"https://www2.census.gov/programs-surveys/susb/tables/{year}/",
                    f"https://www2.census.gov/programs-surveys/susb/datasets/{year}/"):
            files = list_dir(sub)
            picks = [f for f in files
                     if re.search(r"(6digit|naics.*(size|detail)|detailedsizes)",
                                  f, re.I)
                     and re.search(r"\.(xlsx?|csv|txt)$", f, re.I)]
            for f in picks:
                download(sub + f, DIR_SUSB / f"susb_{year}_{f}", max_mb=150)
            if picks:
                got = True
                break
        if not got:
            ds = f"https://www2.census.gov/programs-surveys/susb/datasets/{year}/"
            tb = f"https://www2.census.gov/programs-surveys/susb/tables/{year}/"
            for url, fname in (
                (ds + f"us_6digitnaics_detailedsizes_{year}.txt",
                 f"us_6digitnaics_detailedsizes_{year}.txt"),
                (ds + f"us_state_naics_{year}.txt",
                 f"us_state_naics_{year}.txt"),
                (tb + f"us_6digitnaics_{year}.xls",
                 f"us_6digitnaics_{year}.xls"),
                (tb + f"us_6digitnaics_{year}.xlsx",
                 f"us_6digitnaics_{year}.xlsx"),
            ):
                download(url, DIR_SUSB / f"susb_{year}_{fname}", max_mb=150)


def do_ec():
    print("\n=== Economic Census, sector 23 ===")
    for year, tag in ((2022, "EC2223BASIC"), (2017, "EC1723BASIC")):
        base = (f"https://www2.census.gov/programs-surveys/economic-census/"
                f"data/{year}/sector23/")
        files = [f for f in list_dir(base)
                 if re.search(r"(BASIC|KOB|LOCCONS|VALCON)", f, re.I)
                 and f.lower().endswith(".zip")] or [f"{tag}.zip"]
        for f in files:
            download(base + f, DIR_EC / f"ec{year}_{f}", max_mb=200)
    for cand in ("https://www2.census.gov/econ2012/EC/sector23/EC1223SG01.zip",
                 "https://www2.census.gov/econ2012/EC/sector23/EC1223I1.zip"):
        download(cand, DIR_EC / ("ec2012_" + cand.rsplit("/", 1)[1]))
    outdir = DIR_EC / "extracted"
    outdir.mkdir(exist_ok=True)
    for z in sorted(DIR_EC.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(outdir / z.stem)
        except Exception as e:
            print(f"  [unzipfail] {z.name} -> {e}")


def filter_cbp(zpath, year, kind):
    """Keep construction rows plus the all-industry total row."""
    out = INTERIM / f"cbp{year}{kind}_sector23.csv.gz"
    if out.exists():
        return
    try:
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
            with z.open(name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="latin-1",
                                                     newline=""))
                header = next(reader)
                lower = [h.strip().lower().strip('"') for h in header]
                ni = (lower.index("naics") if "naics" in lower
                      else next(i for i, h in enumerate(lower) if "naics" in h))
                with gzip.open(out, "wt", newline="") as g:
                    w = csv.writer(g)
                    w.writerow(header)
                    kept = 0
                    for row in reader:
                        if ni >= len(row):
                            continue
                        code = row[ni].replace("-", "").replace("/", "").strip()
                        if code == "" or code.startswith("23"):
                            w.writerow(row)
                            kept += 1
        print(f"  [filter] cbp {year}{kind}: {kept} rows kept")
    except Exception as e:
        print(f"  [filterfail] {os.path.basename(zpath)} -> {e}")


def do_cbp():
    print("\n=== County Business Patterns 1998-2023 ===")
    jobs = []
    for year in range(1998, 2024):
        yy = f"{year % 100:02d}"
        for kind in ("co", "us"):
            jobs.append((
                f"https://www2.census.gov/programs-surveys/cbp/datasets/"
                f"{year}/cbp{yy}{kind}.zip",
                DIR_CBP / f"cbp{yy}{kind}.zip", year, kind))

    def one(job):
        url, dest, year, kind = job
        if download(url, dest):
            filter_cbp(dest, year, kind)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, jobs))


def main():
    ensure_dirs()
    t0 = time.time()
    do_bds()
    do_susb()
    do_ec()
    do_cbp()
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
