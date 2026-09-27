"""M2.2 gate: C phi_conv bit-exact vs student_emu (0-diff bar).

Shapes: tiny s1/s2 + real RIFE geometries (conv32@72x88 s1/s2,
conv64@36x44, conv256@18x22, conv512@9x11 with bias). Timing printed.
Usage: python3 c_port/rife/run_conv_test.py
"""
import os
import struct
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_core as S
import rife_cal as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "test_conv")
TMP = "/tmp/opencode/cport"

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def one(H, Wd, Ci, Co, st, bias, seed):
    rng = np.random.default_rng(seed)
    a = (rng.random((H, Wd, Ci)) - 0.5) * 2
    s0, e0, z0 = S.encode(a)
    W = (rng.random((Co, Ci, 3, 3)) - 0.5) * 0.2
    ws, we, wz = S.encode(W.transpose(0, 2, 3, 1))
    m = CAL.m_of(0.5)
    if bias:
        b = (rng.random(Co) - 0.5) * 0.4
        bs, be, bz = S.encode(b)
        Wb = (bs, be, bz)
    else:
        Wb = None
    t = time.time()
    # NB: numpy phi_conv Wb is a (s,e,z) tuple (cf. _WB in exec_int)
    ref = S.phi_conv(s0, e0, z0, {"s": ws, "e": we, "z": wz}, m, st,
                     Wb=(bs, be, bz) if bias else None)
    payload = struct.pack("<9i", H, Wd, Ci, Co, 3, st, 1, m, int(bias))
    payload += s0.tobytes() + e0.tobytes() + z0.tobytes()
    payload += ws.tobytes() + we.tobytes() + wz.tobytes()
    if bias:
        payload += bs.tobytes() + be.tobytes() + bz.tobytes()
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    t = time.time()
    r = subprocess.run([BIN, f"{TMP}/in.bin", f"{TMP}/out.bin"],
                       capture_output=True, text=True)
    dt = time.time() - t
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        raw = fh.read()
    Ho, Wo = struct.unpack("<2i", raw[:8])
    no = Ho * Wo * Co
    gs = np.frombuffer(raw[8:8 + no], dtype=np.int8).reshape(Ho, Wo, Co)
    ge = np.frombuffer(raw[8 + no:8 + 5 * no], dtype=np.int32).reshape(Ho, Wo, Co)
    gz = np.frombuffer(raw[8 + 5 * no:8 + 6 * no], dtype=np.uint8).reshape(Ho, Wo, Co)
    ok = (gs == ref[0]).all() and (ge == ref[1]).all() and (gz == ref[2]).all()
    nd = int((gs != ref[0]).sum() + (ge != ref[1]).sum() + (gz != ref[2]).sum())
    check(f"conv {Ci}->{Co} {H}x{Wd}s{st}{'+b' if bias else ''}", ok,
          f"C {dt * 1000:.0f}ms diffs={nd}")


def main():
    os.makedirs(TMP, exist_ok=True)
    one(8, 8, 4, 6, 1, False, 0)
    one(9, 11, 8, 8, 2, True, 1)
    one(72, 88, 32, 32, 1, True, 2)
    one(36, 44, 32, 32, 2, True, 3)
    one(36, 44, 64, 64, 1, True, 4)
    one(18, 22, 256, 256, 1, True, 5)
    one(9, 11, 512, 512, 1, True, 6)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
