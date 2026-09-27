"""M2.3 gate: C geo kernels bit-exact vs numpy (0-diff bar).

warp_full (real flow magnitudes incl. >16px + border bands), interp
all RIFE dyadic scales (0.125..8), sigmoid wide-range (+-30).
Usage: python3 c_port/rife/run_geo_test.py
"""
import os
import struct
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_core as S
import rife_int as I
import rife_cal as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "test_geo")
TMP = "/tmp/opencode/cport"

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def run(op, payload):
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([BIN, op, f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        return fh.read()


def wtrip(s, e, z):
    return s.tobytes() + e.tobytes() + z.tobytes()


def rtrip(raw, off, n, sh):
    s = np.frombuffer(raw[off:off + n], dtype=np.int8).reshape(*sh)
    e = np.frombuffer(raw[off + n:off + 5 * n], dtype=np.int32).reshape(*sh)
    z = np.frombuffer(raw[off + 5 * n:off + 6 * n], dtype=np.uint8).reshape(*sh)
    return s, e, z


def main():
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(0)
    # warp: image features + LARGE flows (19px class) hitting borders
    H, W, C = 18, 22, 8
    a = (rng.random((H, W, C)) - 0.5) * 4
    s0, e0, z0 = S.encode(a)
    fl = ((rng.random((H, W, 2)) - 0.5) * 38).astype(np.float64)
    fs, fe, fz = S.encode(fl)
    mi, mf, div = CAL.m_of(2.0), 32000, 2
    F = S.to_fixed(s0, e0, z0, mi).reshape(H, W, C)
    Fq = S.to_fixed(fs, fe, fz, mf).reshape(H, W, 2) * div
    ref = I.warp_fixed(F, Fq)
    ro = S.from_fixed(ref.reshape(-1), mi)
    rv = (S.decode(ro[0], ro[1]) * (1 - ro[2])).reshape(H, W, C)
    payload = struct.pack("<7i", H, W, C, mi, mf, div, 0)
    payload += wtrip(s0, e0, z0) + wtrip(fs, fe, fz)
    t = time.time()
    raw = run("warp_full", payload)
    dt = time.time() - t
    H2, W2, C2 = struct.unpack("<3i", raw[:12])
    gs, ge, gz = rtrip(raw, 12, H * W * C, (H, W, C))
    rs, re, rz = S.encode(rv)
    ok = (H2, W2, C2) == (H, W, C) and (gs == rs).all() and (ge == re).all() \
        and (gz == rz).all()
    nd = int((gs != rs).sum() + (ge != re).sum() + (gz != rz).sum())
    check("warp_full 19px+borders", ok, f"C {dt * 1000:.0f}ms diffs={nd}")
    # interp all RIFE scales (k=-3..+3)
    F3 = (rng.random((16, 16, 4)) * 1000).round().astype(np.int64)
    s3, e3, z3 = S.encode(F3.astype(np.float64) / 1000.0)
    mi3 = CAL.m_of(1.0)
    for k, sc in ((-3, 0.125), (-2, 0.25), (-1, 0.5), (1, 2.0), (2, 4.0), (3, 8.0)):
        Ff = S.to_fixed(s3, e3, z3, mi3).reshape(16, 16, 4)
        ref = I.interp_fixed(Ff, sc, sc)
        Ho, Wo = ref.shape[:2]
        ro = S.from_fixed(ref.reshape(-1), mi3)
        rv = (S.decode(ro[0], ro[1]) * (1 - ro[2])).reshape(Ho, Wo, 4)
        payload = struct.pack("<5i", 16, 16, 4, mi3, k)
        payload += wtrip(s3, e3, z3)
        raw = run("interp_full", payload)
        H3, W3 = struct.unpack("<2i", raw[:8])
        gs, ge, gz = rtrip(raw, 8, Ho * Wo * 4, (Ho, Wo, 4))
        rs, re, rz = S.encode(rv)
        ok = (H3, W3) == (Ho, Wo) and (gs == rs).all() and (ge == re).all() \
            and (gz == rz).all()
        nd = int((gs != rs).sum() + (ge != re).sum() + (gz != rz).sum())
        check(f"interp_full 2^{k:+d}", ok, f"diffs={nd}")
    # sigmoid wide range (+-30, span exits both sides)
    xs = np.concatenate([rng.uniform(-30, 30, 2000),
                         [-30., 30., 0., 1., -1., 8., -8., 16.5]])
    ss, se, sz = S.encode(xs)
    I.AUDIT.update({"sig_clip": 0, "sig_span": 0, "sig_n": 0})
    ref = I.sigmoid_int((ss, se, sz))
    payload = struct.pack("<i", xs.size) + wtrip(ss, se, sz)
    raw = run("sigmoid", payload)
    gs, ge, gz = rtrip(raw, 0, xs.size, (xs.size,))
    ok = (gs == ref[0]).all() and (ge == ref[1]).all() and (gz == ref[2]).all()
    nd = int((gs != ref[0]).sum() + (ge != ref[1]).sum() + (gz != ref[2]).sum())
    check("sigmoid +-30", ok, f"diffs={nd} auditsig={I.AUDIT}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
