"""M2.1 gate: C bridge kernels bit-exact vs numpy (0-diff bar).

Generates edge-heavy vectors, runs test_bridge binary, asserts zero
diffs. One FAIL blocks everything downstream.
Usage: python3 c_port/rife/run_bridge_test.py
"""
import os
import struct
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_core as S

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "test_bridge")
TMP = "/tmp/opencode/cport"


def run(op, payload):
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([BIN, op, f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        return fh.read()


FAIL = []


def check(tag, cond):
    print(f"{tag}: {'OK' if cond else 'FAIL'}")
    if not cond:
        FAIL.append(tag)


def main():
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(0)
    # edge-heavy triples: zeros, saturation corners (e>>m, e<<m), signs
    n = 3000
    s = rng.integers(-1, 2, size=n).astype(np.int8)
    s[s == 0] = 1
    e = np.clip(rng.integers(28000, 36000, size=n), 0, 65535).astype(np.int32)
    z = (rng.random(n) < 0.05).astype(np.uint8)
    s[:6] = [1, -1, 1, -1, 1, 0]
    e[:6] = [0, 0, 65535, 65535, 32768, 0]
    z[:6] = [0, 0, 0, 0, 0, 1]
    m = 32000
    payload = struct.pack("<ii", n, m) + s.tobytes() + e.tobytes() + z.tobytes()
    got = run("to_fixed", payload)
    want = S.to_fixed(s, e, z, m).tobytes()
    check("to_fixed", got == want)
    # from_fixed edges: 0, +-1, powers of 2 +-1, near 2^53, big Wang
    edge = [0, 1, -1, 2, -2, (1 << 14) - 1, 1 << 14, (1 << 14) + 1,
            (1 << 18) - 1, 1 << 18, (1 << 40) + 12345, -(1 << 40) - 12345,
            (1 << 52) - 1, 1 << 52, (1 << 53) - 1]
    q = np.concatenate([np.array(edge, dtype=np.int64),
                        rng.integers(-(1 << 50), 1 << 50, size=n).astype(np.int64)])
    payload = struct.pack("<ii", q.size, m) + q.tobytes()
    got = run("from_fixed", payload)
    so, eo, zo = S.from_fixed(q, m)
    want = so.tobytes() + eo.tobytes() + zo.tobytes()
    check("from_fixed", got == want)
    # tdiv incl. negatives (C / is NOT trunc — the reason tdiv exists)
    a = np.concatenate([np.array([7, -7, 0, -1, 1 << 40, -(1 << 40)], dtype=np.int64),
                        rng.integers(-10**6, 10**6, size=n).astype(np.int64)])
    b = np.concatenate([np.array([2, 2, 7, 7, 3, 3], dtype=np.int64),
                        rng.integers(1, 1000, size=n).astype(np.int64)])
    bb = b[:len(a)].copy()
    bb[::2] *= -1  # mixed divisor signs (trunc-vs-floor colony)
    payload = struct.pack("<ii", a.size, 0) + a.tobytes() + bb.tobytes()
    got = run("tdiv", payload)
    want = S.tdiv(a, bb).tobytes()
    check("tdiv", got == want)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
