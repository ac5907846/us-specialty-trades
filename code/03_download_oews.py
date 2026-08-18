"""Download BLS OEWS national and state wage percentile files, 1999-2024.

If every request is refused (the BLS front end rate-limits scripted
traffic), re-run once, or download the May national and state XLS zips from
https://www.bls.gov/oes/tables.htm into data/raw/supplementary/oews/ under
their original names (oesm23st.zip etc)."""
import shutil
import subprocess
import time
import urllib.request
import zipfile

from config import DIR_OEWS

DIR_OEWS.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36",
      "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                 "*/*;q=0.8"),
      "Accept-Language": "en-US,en;q=0.9",
      "Referer": "https://www.bls.gov/oes/tables.htm",
      "Connection": "keep-alive"}


def curl_get(url, dest):
    curl = shutil.which("curl")
    if not curl:
        return False
    r = subprocess.run(
        [curl, "-sSLf", "--max-time", "180",
         "-A", UA["User-Agent"],
         "-H", f"Referer: {UA['Referer']}",
         "-H", f"Accept-Language: {UA['Accept-Language']}",
         "-o", str(dest), url],
        capture_output=True, text=True)
    ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 10000
    if not ok and dest.exists():
        dest.unlink()
    return ok


def get(url, dest, retries=2):
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"  [have] {dest.name}")
        return True
    err = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(dest, "wb") as f:
                f.write(r.read())
            if dest.stat().st_size > 10000:
                print(f"  [ok]   {dest.name}  "
                      f"({dest.stat().st_size / 1e6:.1f} MB)")
                return True
            dest.unlink()
            err = "response too small"
        except Exception as e:
            err = e
        if "404" in str(err):
            return False
        time.sleep(2 * i)
    if curl_get(url, dest):
        print(f"  [ok/curl] {dest.name}  "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  [fail] {dest.name} -> {err}")
    if dest.exists():
        dest.unlink()
    return False


def unpack(zpath):
    out = DIR_OEWS / zpath.stem
    if out.exists():
        return
    try:
        with zipfile.ZipFile(zpath) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith((".xls", ".xlsx"))]
            out.mkdir(exist_ok=True)
            for m in members:
                target = out / m.split("/")[-1]
                with z.open(m) as src, open(target, "wb") as f:
                    f.write(src.read())
        print(f"  [unpack] {zpath.stem}: {len(members)} spreadsheets")
    except Exception as e:
        print(f"  [unpackfail] {zpath.name} -> {e}")


def main():
    base = "https://www.bls.gov/oes/special-requests/"
    for year in range(1999, 2025):
        yy = f"{year % 100:02d}"
        for scope in ("nat", "st"):
            got = False
            for name in (f"oesm{yy}{scope}.zip", f"oes{yy}{scope}.zip"):
                dest = DIR_OEWS / name
                if get(base + name, dest):
                    unpack(dest)
                    got = True
                    break
            if not got:
                print(f"  [miss] {year} {scope}")
            time.sleep(1.5)
    n = len(list(DIR_OEWS.glob("*.zip")))
    print(f"done: {n} zip files on disk.")


if __name__ == "__main__":
    main()
