"""M2.3 gate: STAGED full-contextnet parity — every math layer through
C binaries on real data (f_012 identical pair), moves in numpy (exact
by construction, no arithmetic). Compares EVERY blob vs numpy exec_int,
0-diff bar. (The C executor itself is M2.4; staging gates the kernels
on production wiring first.)
Usage: python3 c_port/rife/run_stage_ctx.py
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
import rife_exec

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp/opencode/cport"
RIFE = "/tmp/opencode/rife-bin/rife-ncnn-vulkan-20221029-ubuntu/rife-v2.3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAIL = []


def runb(binary, op, payload):
    with open(f"{TMP}/in.bin", "wb") as fh:
        fh.write(payload)
    r = subprocess.run([os.path.join(HERE, binary), op, f"{TMP}/in.bin",
                        f"{TMP}/out.bin"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(f"{TMP}/out.bin", "rb") as fh:
        return fh.read()


def wtrip(s, e, z):
    return (np.ascontiguousarray(s).tobytes() + np.ascontiguousarray(e).tobytes()
            + np.ascontiguousarray(z).tobytes())


def rtrip(raw, off, n):
    s = np.frombuffer(raw[off:off + n], dtype=np.int8)
    e = np.frombuffer(raw[off + n:off + 5 * n], dtype=np.int32)
    z = np.frombuffer(raw[off + 5 * n:off + 6 * n], dtype=np.uint8)
    return s, e, z


def main():
    from PIL import Image
    layers = rife_ncnn.parse_param(f"{RIFE}/contextnet.param")
    Wf, _ = rife_ncnn.load_bin(f"{RIFE}/contextnet.bin", layers)
    Wt = {}
    for k, v in Wf.items():
        a = v.astype(np.float64)
        if a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        s, e, z = S.encode(a)
        Wt[k] = {"s": s, "e": e, "z": z}
    fr = f"{ROOT}/samples"
    im0 = np.asarray(Image.open(f"{fr}/f_012.png").convert("RGB"))

    def preproc(im):
        x = im.astype(np.float32) / 255.0
        H, W, _ = x.shape
        Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
        out = np.zeros((3, Hp, Wp), np.float32)
        out[:, :H, :W] = x.transpose(2, 0, 1)
        return out

    p0 = preproc(im0)
    fl = rife_ncnn.parse_param(f"{RIFE}/flownet.param")
    Wfl, _ = rife_ncnn.load_bin(f"{RIFE}/flownet.bin", fl)
    f0 = np.ascontiguousarray(
        rife_ref.execute(fl, Wfl, {"input0": p0, "input1": p0})["flow"][:2])
    M = CAL.calibrate(layers, Wf, {"input.1": p0, "flow.0": f0})

    def triples(a):
        s, e, z = S.encode(np.asarray(a, dtype=np.float64))
        return s, e, z

    feeds = {"input.1": triples(p0), "flow.0": triples(f0)}
    bi = rife_exec.execute_int(layers, Wt, M, feeds)
    # staged run: blobs as HWC triples (mirror execute_int entry _chw2hwc)
    def chw2hwc(t):
        return (np.ascontiguousarray(t[0].transpose(1, 2, 0)),
                np.ascontiguousarray(t[1].transpose(1, 2, 0)),
                np.ascontiguousarray(t[2].transpose(1, 2, 0)))
    bs = {k: chw2hwc(v) for k, v in feeds.items()}

    def shape_of(t):
        return t[0].shape

    def c_conv(name, ins, m_acc, Wb, stride, pad, kh):
        s, e, z = ins
        H, Wd, Ci = s.shape
        Co = Wt[name + ".weight"]["s"].shape[0]
        w = Wt[name + ".weight"]
        payload = struct.pack("<9i", H, Wd, Ci, Co, kh, stride, pad, m_acc,
                              int(Wb is not None))
        payload += wtrip(s, e, z) + wtrip(w["s"], w["e"], w["z"])
        if Wb is not None:
            payload += wtrip(*Wb)
        # test_conv takes (in out), no op arg
        with open(f"{TMP}/in.bin", "wb") as fh:
            fh.write(payload)
        r = subprocess.run([os.path.join(HERE, "test_conv"), f"{TMP}/in.bin",
                            f"{TMP}/out.bin"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(f"{TMP}/out.bin", "rb") as fh:
            raw = fh.read()
        Ho, Wo = struct.unpack("<2i", raw[:8])
        no = Ho * Wo * Co
        return rtrip(raw, 8, no) + ((Ho, Wo, Co),)

    nbad, ntot = 0, 0
    for L in layers:
        t, p, name = L["type"], L["p"], L["name"]
        if all(k in bs for k in L["outs"]):
            continue
        ins = [bs[k] for k in L["ins"]]
        if t == "Input":
            pass
        elif t == "Split":
            for k in L["outs"]:
                bs[k] = ins[0]
        elif t == "Concat":
            ax = {0: 2, 1: 0, 2: 1}[p.get(0, 0)]
            bs[L["outs"][0]] = (np.concatenate([v[0] for v in ins], ax),
                                np.concatenate([v[1] for v in ins], ax).astype(np.int32),
                                np.concatenate([v[2] for v in ins], ax))
        elif t == "Crop":
            st, en, ax = p[-23309], p[-23310], p[-23311]
            mp = {0: 2, 1: 0, 2: 1}
            sl = [slice(None)] * 3
            dims = (ins[0][0].shape[2], ins[0][0].shape[0], ins[0][0].shape[1])
            for s_, e_, a_ in zip(st, en, ax):
                n_ = dims[a_]
                s_ = 0 if s_ == -233 else (s_ if s_ >= 0 else n_ + s_)
                e_ = n_ if e_ == -233 else (e_ if e_ > 0 else n_ + e_)
                sl[mp[a_]] = slice(s_, min(n_, e_))
            tt = tuple(sl)
            bs[L["outs"][0]] = tuple(np.ascontiguousarray(v[tt]) for v in ins[0])
        elif t == "Convolution":
            kh, stride, pad = p[1], p.get(3, 1), p.get(4, 0)
            wkey = name + ".weight"
            Co = Wt[wkey]["s"].shape[0]
            bkey = name + ".bias"
            if bkey in Wt:
                d = Wt[bkey]
                Wb = (d["s"], d["e"], d["z"])
            else:
                Wb = None
            s, e, z, sh = c_conv(name, ins[0], M[name], Wb, stride, pad, kh)
            s = s.reshape(*sh)
            e = e.reshape(*sh)
            z = z.reshape(*sh)
            # fused act via C (mirror _act leaky/sigmoid)
            if 9 in p:
                import rife_ref as _R
                kind = _R.ACT9.get(p[9])
                if kind == "leaky":
                    v = p[-23310]
                    slope = float(v[0]) if isinstance(v, list) else float(v)
                    ss, se, sz = S.encode(np.full(sh, slope))
                    n_ = int(np.prod(sh))
                    payload = struct.pack("<6i", n_, 0, 0, 0, 0, 0)
                    payload += wtrip(s.reshape(-1), e.reshape(-1), z.reshape(-1))
                    payload += wtrip(np.full(n_, ss.flat[0], np.int8),
                                     np.full(n_, se.flat[0], np.int32),
                                     np.zeros(n_, np.uint8))
                    raw = runb("test_arith", "prelu", payload)
                    s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
                elif kind == "sigmoid":
                    n_ = int(np.prod(sh))
                    payload = struct.pack("<i", n_)
                    payload += wtrip(s.reshape(-1), e.reshape(-1), z.reshape(-1))
                    raw = runb("test_geo", "sigmoid", payload)
                    s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "PReLU":
            sl = Wt[name + ".slope"]
            sh = shape_of(ins[0])
            C = sh[2]
            if sl["s"].size == 1:
                zs = (np.full(sh, sl["s"].flat[0], np.int8),
                      np.full(sh, sl["e"].flat[0], np.int32), np.zeros(sh, np.uint8))
            else:
                zs = (np.broadcast_to(sl["s"].reshape(1, 1, C), sh).copy().astype(np.int8),
                      np.broadcast_to(sl["e"].reshape(1, 1, C), sh).copy().astype(np.int32),
                      np.zeros(sh, np.uint8))
            n_ = int(np.prod(sh))
            payload = struct.pack("<6i", n_, 0, 0, 0, 0, 0)
            payload += wtrip(ins[0][0].reshape(-1), ins[0][1].reshape(-1),
                             ins[0][2].reshape(-1))
            payload += wtrip(zs[0].reshape(-1), zs[1].reshape(-1), zs[2].reshape(-1))
            raw = runb("test_arith", "prelu", payload)
            s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "BinaryOp":
            op = p.get(0, 0)
            if p.get(1, 0) == 1 and len(ins) == 1:
                sc = float(p[2])
                cs, ce, cz = S.encode(np.atleast_1d(np.asarray(sc, dtype=np.float64)))
                sh = shape_of(ins[0])
                n_ = int(np.prod(sh))
                c = (np.full(n_, cs.flat[0], np.int8), np.full(n_, ce.flat[0], np.int32),
                     np.zeros(n_, np.uint8))
                m_out = M["cov:" + L["ins"][0]]
                nm = {0: "add", 1: "sub", 2: "mul", 7: "rsub"}[op]
                a = ins[0]
            else:
                a, c0 = ins
                sh = shape_of(a)
                n_ = int(np.prod(sh))
                c = (c0[0].reshape(-1), c0[1].reshape(-1), c0[2].reshape(-1))
                m_out = max(M["cov:" + L["ins"][0]], M["cov:" + L["ins"][1]])
                nm = {0: "add", 1: "sub", 2: "mul", 7: "rsub"}[op]
            payload = struct.pack("<6i", n_, m_out, 0, 0, 0, 0)
            payload += wtrip(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1))
            payload += wtrip(*c)
            raw = runb("test_arith", nm, payload)
            s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "Pooling":
            m_in = M["cov:" + L["ins"][0]]
            a = ins[0]
            sh = shape_of(a)
            H, W, C = sh
            n_ = H * W * C
            payload = struct.pack("<6i", n_, m_in, H, W, C, 0)
            payload += wtrip(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1))
            raw = runb("test_arith", "pool", payload)
            s, e, z = rtrip(raw, 0, C)
            bs[L["outs"][0]] = (s.reshape(1, 1, C), e.reshape(1, 1, C), z.reshape(1, 1, C))
        elif t == "Interp":
            m_in = M["cov:" + L["ins"][0]]
            a = ins[0]
            H, W, C = shape_of(a)
            sc = float(p[1])
            k = {0.125: -3, 0.25: -2, 0.5: -1, 1.0: 0, 2.0: 1, 4.0: 2, 8.0: 3}[sc]
            assert float(p[2]) == sc
            n_ = H * W * C
            payload = struct.pack("<5i", H, W, C, m_in, k)
            payload += wtrip(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1))
            raw = runb("test_geo", "interp_full", payload)
            Ho, Wo = struct.unpack("<2i", raw[:8])
            no = Ho * Wo * C
            s, e, z = (x.reshape(Ho, Wo, C) for x in rtrip(raw, 8, no))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "PixelShuffle":
            C = ins[0][0].shape[2]
            H, W = ins[0][0].shape[0], ins[0][0].shape[1]
            s, e, z = (v.transpose(2, 0, 1) for v in ins[0])
            G = s.shape[0] // 4
            oo = lambda v: np.ascontiguousarray(
                v.reshape(G, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(G, H * 2, W * 2)).transpose(1, 2, 0)
            bs[L["outs"][0]] = (oo(s), oo(e.astype(np.int64)).astype(np.int32), oo(z))
        elif t == "UnaryOp":
            assert p.get(0) == 1
            bs[L["outs"][0]] = ((-ins[0][0].astype(np.int16)).astype(np.int8),
                                ins[0][1], ins[0][2])
        elif t == "Clip":
            lo, hi = float(p.get(0, -1e9)), float(p.get(1, 1e9))
            a = ins[0]
            sh = shape_of(a)
            n_ = int(np.prod(sh))
            ls, le, lz = S.encode(np.full(1, lo))
            hs, he, hz = S.encode(np.full(1, hi))
            payload = struct.pack("<6i", n_, 0, 0, 0, 0, 0)
            payload += wtrip(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1))
            payload += wtrip(ls, le, lz) + wtrip(hs, he, hz)
            raw = runb("test_arith", "clip", payload)
            s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "rife.Warp":
            f, fl = ins
            H, W, C = shape_of(f)
            mi = M["cov:" + L["ins"][0]]
            mf = M.get("flow_m", rife_exec.M_FLOW)
            div = M.get("flow_div", 1)
            payload = struct.pack("<7i", H, W, C, mi, mf, div, 0)
            payload += wtrip(f[0].reshape(-1), f[1].reshape(-1), f[2].reshape(-1))
            payload += wtrip(fl[0].reshape(-1), fl[1].reshape(-1), fl[2].reshape(-1))
            raw = runb("test_geo", "warp_full", payload)
            H2, W2, C2 = struct.unpack("<3i", raw[:12])
            assert (H2, W2, C2) == (H, W, C)
            s, e, z = (x.reshape(H, W, C) for x in rtrip(raw, 12, H * W * C))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        elif t == "Deconvolution":
            raise SystemExit("contextnet has no deconv; M2.4 covers it")
        elif t == "Sigmoid":
            a = ins[0]
            sh = shape_of(a)
            n_ = int(np.prod(sh))
            payload = struct.pack("<i", n_)
            payload += wtrip(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1))
            raw = runb("test_geo", "sigmoid", payload)
            s, e, z = (x.reshape(*sh) for x in rtrip(raw, 0, n_))
            bs[L["outs"][0]] = (np.ascontiguousarray(s), np.ascontiguousarray(e),
                                np.ascontiguousarray(z))
        else:
            raise SystemExit(f"unmapped {t}")
    for L in layers:
        for k in L["outs"]:
            if k not in bi or k not in bs:
                continue
            ntot += 1
            a, b = bi[k], bs[k]
            if a[0].shape != b[0].shape or not (bool((a[0] == b[0]).all())
                    and bool((a[1] == b[1]).all()) and bool((a[2] == b[2]).all())):
                nbad += 1
                print(f"DIFF {L['name']} {k}", flush=True)
    print(f"STAGE_CTX: {ntot - nbad}/{ntot} blobs exact",
          "GO" if nbad == 0 else "NO-GO", flush=True)


if __name__ == "__main__":
    main()
