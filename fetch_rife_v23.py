"""Fetch RIFE-v2.3 ncnn weights (one-time setup, mirrors dav2 export).

Downloads nihui's rife-ncnn-vulkan 20221029 release (contains the
rife-v2.3 .param/.bin our port was verified against), extracts just
rife-v2.3/ into ./rife-v2.3/, and VERIFIES contents (fails loudly on
any mismatch — never trust a blind download).
Usage: python3 fetch_rife_v23.py [--dir rife-v2.3]
"""
import os
import sys
import urllib.request
import zipfile

URL = ("https://github.com/nihui/rife-ncnn-vulkan/releases/download/"
       "20221029/rife-ncnn-vulkan-20221029-ubuntu.zip")
WANT = {"flownet.param", "flownet.bin", "contextnet.param",
        "contextnet.bin", "fusionnet.param", "fusionnet.bin"}


def main(d="rife-v2.3"):
    if os.path.isdir(d) and WANT <= set(os.listdir(d)):
        print(f"{d}/ already complete — nothing to do")
        return
    os.makedirs(d, exist_ok=True)
    zpath = "/tmp/rife-ncnn-20221029-ubuntu.zip"
    if not os.path.exists(zpath):
        print("downloading (~30MB)...", flush=True)
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for w in sorted(WANT):
            hit = [n for n in names if n.endswith("rife-v2.3/" + w)]
            if not hit:
                sys.exit(f"error: {w} not in archive (release layout changed?)")
            with z.open(hit[0]) as fh, open(os.path.join(d, w), "wb") as out:
                out.write(fh.read())
    got = set(os.listdir(d))
    missing = WANT - got
    if missing:
        sys.exit(f"error: incomplete after extract: {missing}")
    # content sanity: params parse as ncnn headers
    for p in ("flownet", "contextnet", "fusionnet"):
        head = open(os.path.join(d, p + ".param")).readline().strip()
        assert head == "7767517", f"{p}.param bad magic: {head}"
    print(f"verified {d}/: {sorted(got)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rife-v2.3")
