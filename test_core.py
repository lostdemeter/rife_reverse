"""Fast numpy-only gates (no torch, no model files, seconds).

Covers the geometric core: lattice roundtrip, trunc division signs,
integer bit_length edges, LUT determinism. The heavy gates are the
demo itself (int-vs-float >=40dB) and `make -C c_port test`.
Usage: python3 test_core.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_core as S

FAIL = []


def check(tag, cond):
    print(f"{tag}: {'OK' if cond else 'FAIL'}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(0)
    # lattice encode/decode roundtrip (values land back on lattice)
    x = (rng.random(2000) - 0.5) * 8
    s, e, z = S.encode(x)
    v = S.decode(s, e) * (1 - z)
    check("lattice-roundtrip", float(np.abs(v - x).max()) / max(1e-12, float(np.abs(x).max())) < 1e-3)
    # trunc division both signs (C semantics — floor is a bug here)
    a = np.array([7, -7, 0, -1, 1 << 40, -(1 << 40)], dtype=np.int64)
    b = np.array([2, 2, 7, 7, 3, 3], dtype=np.int64)
    check("tdiv-pos", bool((S.tdiv(a, b) == np.array([3, -3, 0, 0, 366503875925, -366503875925])).all()))
    check("tdiv-negdiv", bool((S.tdiv(a, -b) == -np.array([3, -3, 0, 0, 366503875925, -366503875925])).all()))
    # integer bit_length edges incl. full int64 range
    edge = np.array([0, 1, 2, 3, 2**15, 2**31, 2**32, 2**52, 2**53, 2**62, 2**63 - 1],
                    dtype=np.int64)
    ref = np.array([int(v).bit_length() for v in edge])
    check("bit_length", bool((S.bit_length_int(edge) == ref).all()))
    # LUT determinism: same build twice (proves frozen formulas)
    f1 = S.L_FRAC().copy()
    S._t.clear()
    f2 = S.L_FRAC()
    check("lut-deterministic", bool((f1 == f2).all()))
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
