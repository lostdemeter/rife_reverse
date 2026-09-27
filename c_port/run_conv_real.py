"""M2.2 sanity: C phi_conv on REAL video tensors (not uniform random).

Dumps real input triples + weights + bias + m for production layers
(CIF pair f_012/f_014), runs test_conv binary, asserts 0 diffs vs
numpy on the same data. Real activations stress sparsity, dynamic
range, and saturation corners that random fills miss.
Layers: contextnet Conv_0 (sparse image feats), flownet Conv_2
(stride-2 entry), fusionnet Conv_46 (deep 512ch).
Usage: python3 c_port/rife/run_conv_real.py
"""
import os
import struct
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rife_ncnn
import rife_ref
import phi_core as S
import rife_cal as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "test_conv")
TMP = "/tmp/opencode/cport"
RIFE = "/tmp/opencode/rife-bin/rife-ncnn-vulkan-20221029-ubuntu/rife-v2.3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dump_run(net, layer_name, feeds, tag):
    L = rife_ncnn.parse_param(f"{RIFE}/{net}.param")
    Wf, _ = rife_ncnn.load_bin(f"{RIFE}/{net}.bin", L)
    M = CAL.calibrate(L, Wf, feeds)
    lay = next(x for x in L if x["name"] == layer_name)
    p = lay["p"]
    kh, st, pad = p[1], p.get(3, 1), p.get(4, 0)
    bf = rife_ref.execute(L, Wf, feeds)
    a = bf[lay["ins"][0]]
    s0, e0, z0 = S.encode(a.astype(np.float64).transpose(1, 2, 0))
    W = Wf[layer_name + ".weight"]
    Co, Ci = W.shape[0], W.shape[1]
    ws, we, wz = S.encode(W.astype(np.float64).transpose(0, 2, 3, 1))
    m = M[layer_name]
    bkey = layer_name + ".bias"
    hasb = bkey in Wf
    if hasb:
        bs, be, bz = S.encode(Wf[bkey].astype(np.float64))
    H, Wd, _ = s0.shape
    ref = S.phi_conv(s0, e0, z0, {"s": ws, "e": we, "z": wz}, m, st,
                     Wb=(bs, be, bz) if hasb else None)
    payload = struct.pack("<9i", H, Wd, Ci, Co, kh, st, pad, m, int(hasb))
    payload += s0.tobytes() + e0.tobytes() + z0.tobytes()
    payload += ws.tobytes() + we.tobytes() + wz.tobytes()
    if hasb:
        payload += bs.tobytes() + be.tobytes() + bz.tobytes()
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    import time
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
    # distribution stats proving this isn't random-fill territory
    print(f"  in: max|e-BIAS| {(np.abs(e0.astype(int) - S.BIAS)).max()}, "
          f"zeros {(z0 > 0).mean() * 100:.1f}%, amax {np.abs(a).max():.3f}")
    check(tag, ok, f"C {dt * 1000:.0f}ms diffs={nd}")


def main():
    from PIL import Image
    os.makedirs(TMP, exist_ok=True)
    fr = f"{ROOT}/samples"
    im0 = np.asarray(Image.open(f"{fr}/f_012.png").convert("RGB"))
    im1 = np.asarray(Image.open(f"{fr}/f_014.png").convert("RGB"))

    def preproc(im):
        x = im.astype(np.float32) / 255.0
        H, W, _ = x.shape
        Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
        out = np.zeros((3, Hp, Wp), np.float32)
        out[:, :H, :W] = x.transpose(2, 0, 1)
        return out

    p0, p1 = preproc(im0), preproc(im1)
    fl = rife_ncnn.parse_param(f"{RIFE}/flownet.param")
    Wfl, _ = rife_ncnn.load_bin(f"{RIFE}/flownet.bin", fl)
    flow = np.ascontiguousarray(
        rife_ref.execute(fl, Wfl, {"input0": p0, "input1": p1})["flow"])
    dump_run("contextnet", "Conv_0", {"input.1": p0, "flow.0": flow[:2]},
             "REAL ctx Conv_0 3->32 full-res")
    dump_run("flownet", "Conv_2", {"input0": p0, "input1": p1},
             "REAL flow Conv_2 s2 entry")
    cl = rife_ncnn.parse_param(f"{RIFE}/contextnet.param")
    Wcl, _ = rife_ncnn.load_bin(f"{RIFE}/contextnet.bin", cl)
    c0 = rife_ref.execute(cl, Wcl, {"input.1": p0, "flow.0": flow[:2]})
    c1 = rife_ref.execute(cl, Wcl, {"input.1": p1, "flow.0": flow[2:]})
    ul = rife_ncnn.parse_param(f"{RIFE}/fusionnet.param")
    Wul, _ = rife_ncnn.load_bin(f"{RIFE}/fusionnet.bin", ul)
    ff = {"img0": p0, "img1": p1, "flow": np.ascontiguousarray(flow)}
    for i, k in enumerate(("f1", "f2", "f3", "f4")):
        ff[str(3 + i)] = np.ascontiguousarray(c0[k])
        ff[str(7 + i)] = np.ascontiguousarray(c1[k])
    dump_run("fusionnet", "Conv_46", ff, "REAL fus Conv_46 deep")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
