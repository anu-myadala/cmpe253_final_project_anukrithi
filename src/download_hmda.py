"""Download the public HMDA loan-level data for the project cohort.

Pulls one CSV per year from the CFPB/FFIEC HMDA Data Browser API using the filters in
config.yaml (state, home-purchase, originated or denied). The API allows at most two filter
criteria besides year/state, so loan type and the other cohort rules are applied in prepare.py.
No login is needed. The API builds the file on request, so a state-wide query can take minutes.

Run:  python src/download_hmda.py            (both years in config.yaml)
      python src/download_hmda.py --years 2024
"""
import argparse
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / "data" / "raw"


def build_url(year: int) -> str:
    params = {"years": str(year), "states": CFG["state"], **CFG["api_filters"]}
    return CFG["api_base"] + "?" + urllib.parse.urlencode(params, safe=",")


def raw_path(year: int) -> Path:
    return RAW / f"hmda_{CFG['state']}_{year}_purchase_orig_denied.csv"


def download(year: int, retries: int = 3) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = raw_path(year)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} already exists ({out.stat().st_size/1e6:.1f} MB)")
        return out
    url = build_url(year)
    print(f"[get ] {url}")
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            # The API answers with a redirect to a cached file, and it rejects Python's default
            # client, so the download goes through curl (-L follows the redirect).
            subprocess.run(["curl", "-sSL", "--compressed", "--fail", "-o", str(out), url],
                           check=True, timeout=1800)
            total = out.stat().st_size
            rows = sum(1 for _ in open(out, "rb")) - 1
            if rows < 1000:
                raise RuntimeError(f"only {rows} rows came back")
            print(f"[done] {out.name}: {total/1e6:.1f} MB, {rows:,} rows, {time.time()-t0:.0f}s")
            return out
        except Exception as exc:  # network hiccups are common on this API; retry
            print(f"[warn] attempt {attempt} failed: {exc}")
            if out.exists():
                out.unlink()
            time.sleep(10 * attempt)
    sys.exit(f"could not download year {year}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int,
                    default=[CFG["years"]["train_test"], CFG["years"]["drift"]])
    for y in ap.parse_args().years:
        download(y)
