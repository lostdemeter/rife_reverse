"""M2.3 gate: C arith kernels bit-exact vs numpy (0-diff bar).

tmul/prelu/add/sub/rsub/mul/clip/pool/neg on edge-heavy data.
Usage: python3 c_port/rife/run_arith_test.py
"""
import os
import struct
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_core as S
import rife_int as I
import rife_exec as EI
import rife_cal as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "test_arith")
TMP = "/tmp/opencode/cport"

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def run(op, n, m, *arrs):
    payload = struct.pack("<6i", n, m, 0, 0, 0, 0)
    for (s, e, z) in arrs:
        payload += s.tobytes() + e.tobytes() + z.tobytes()
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([BIN, op, f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        raw = fh.read()
    s = np.frombuffer(raw[:n], dtype=np.int8)
    e = np.frombuffer(raw[n:5 * n], dtype=np.int32)
    z = np.frombuffer(raw[5 * n:6 * n], dtype=np.uint8)
    return s, e, z


def main():
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(1)
    n = 2000
    a = (rng.random(n) - 0.5) * 6
    b = (rng.random(n) - 0.5) * 6
    s0, e0, z0 = S.encode(a)
    s1, e1, z1 = S.encode(b)
    m = CAL.m_of(3.0)
    A, B = (s0, e0, z0), (s1, e1, z1)
    from rife_int import tmul as _tmul
    ref = _tmul(A, B)
    got = run("tmul", n, m, A, B)
    check("tmul", all((g == r).all() for g, r in zip(got, ref)))
    sl = S.encode(np.full(n, -0.37))
    ref = I.prelu_int(A, sl)
    got = run("prelu", n, m, A, sl)
    check("prelu", all((g == r).all() for g, r in zip(got, ref)))
    for op, nm in ((0, "add"), (1, "sub"), (7, "rsub")):
        ref = EI._bin2(op, A, m, B, m, m)
        got = run(nm, n, m, A, B)
        check(f"bin2 {nm}", all((g == r).all() for g, r in zip(got, ref)))
    ref = EI._bin2(2, A, m, B, m, m)
    got = run("mul", n, m, A, B)
    check("bin2 mul", all((g == r).all() for g, r in zip(got, ref)))
    lo, hi = S.encode(np.full(1, -0.5)), S.encode(np.full(1, 0.5))
    ref = EI._clip_int(A, -0.5, 0.5)
    # clip driver takes lo/hi as 1-elem triples after main array
    payload = struct.pack("<6i", n, m, 0, 0, 0, 0)
    payload += s0.tobytes() + e0.tobytes() + z0.tobytes()
    payload += lo[0].tobytes() + lo[1].tobytes() + lo[2].tobytes()
    payload += hi[0].tobytes() + hi[1].tobytes() + hi[2].tobytes()
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([BIN, "clip", f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        raw = fh.read()
    got = (np.frombuffer(raw[:n], dtype=np.int8),
           np.frombuffer(raw[n:5 * n], dtype=np.int32),
           np.frombuffer(raw[5 * n:6 * n], dtype=np.uint8))
    check("clip", all((g == r).all() for g, r in zip(got, ref)))
    H, W, C = 6, 7, 4
    ap = (rng.random((H, W, C)) - 0.5) * 2
    sp, ep, zp = S.encode(ap)
    F = S.to_fixed(sp, ep, zp, m).reshape(H, W, C)
    tot = F.sum(axis=(0, 1))
    ro = S.from_fixed(S.tdiv(tot, H * W), m)
    ref = (ro[0].reshape(1, 1, C), ro[1].reshape(1, 1, C), ro[2].reshape(1, 1, C))
    payload = struct.pack("<6i", H * W * C, m, H, W, C, 0)
    payload += sp.tobytes() + ep.tobytes() + zp.tobytes()
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([BIN, "pool", f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        raw = fh.read()
    got = (np.frombuffer(raw[:C], dtype=np.int8).reshape(1, 1, C),
           np.frombuffer(raw[C:5 * C], dtype=np.int32).reshape(1, 1, C),
           np.frombuffer(raw[5 * C:6 * C], dtype=np.uint8).reshape(1, 1, C))
    check("pool", all((g == r).all() for g, r in zip(got, ref)))
    ref = ((-s0.astype(np.int16)).astype(np.int8), e0, z0)
    got = run("neg", n, m, A)
    check("neg", all((g == r).all() for g, r in zip(got, ref)))
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
