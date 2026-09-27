"""Interpolate one frame between two images with the geometric RIFE.

Runs the float reference AND the integer datapath, saves the int mid,
prints int-vs-float fidelity (expect >=40dB — the Phase-1 gate).
Backends: numpy integer emu (default, CPU, no torch needed) or
--torch for the bit-exact CUDA backend (same values, ~60x faster).
Usage:
  python3 fetch_rife_v23.py            # one-time weights (auto also)
  python3 demo_interp.py samples/f_012.png samples/f_014.png captures/mid.png
  python3 demo_interp.py a.png b.png captures/mid.png --torch
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BAR_DB = 40.0


def preproc(im):
    import numpy as np
    x = im.astype(np.float32) / 255.0
    H, W, _ = x.shape
    Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    out = np.zeros((3, Hp, Wp), np.float32)
    out[:, :H, :W] = x.transpose(2, 0, 1)
    return out, (H, W)


def main(a_path, b_path, out_path, torch_backend=False):
    import numpy as np
    from PIL import Image
    import rife_ncnn as ncnn_io
    import rife_ref as ref_exec
    import phi_core as S
    import rife_cal as CAL

    R = os.environ.get("RIFE_DIR", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rife-v2.3"))
    if not os.path.isdir(R):
        import fetch_rife_v23
        fetch_rife_v23.main(R)
    nets = {}
    for n in ("flownet", "contextnet", "fusionnet"):
        L = ncnn_io.parse_param(f"{R}/{n}.param")
        Wf, _ = ncnn_io.load_bin(f"{R}/{n}.bin", L)
        Wt = {}
        for k, v in Wf.items():
            a = v.astype(np.float64)
            if a.ndim == 4:
                a = a.transpose(0, 2, 3, 1)  # OIHW -> OHWI layout bridge
            s, e, z = S.encode(a)
            Wt[k] = {"s": s, "e": e, "z": z}
        nets[n] = (L, Wf, Wt)
    fl, Wfl, Wtfl = nets["flownet"]
    cl, Wcl, Wtcl = nets["contextnet"]
    ul, Wul, Wtul = nets["fusionnet"]

    def triples(a):
        s, e, z = S.encode(np.asarray(a, dtype=np.float64))
        return s, e, z

    p0, (H, W) = preproc(np.asarray(Image.open(a_path).convert("RGB")))
    p1, _ = preproc(np.asarray(Image.open(b_path).convert("RGB")))
    flow = np.ascontiguousarray(
        ref_exec.execute(fl, Wfl, {"input0": p0, "input1": p1})["flow"])
    c0 = ref_exec.execute(cl, Wcl, {"input.1": p0, "flow.0":
                                    np.ascontiguousarray(flow[:2])})
    c1 = ref_exec.execute(cl, Wcl, {"input.1": p1, "flow.0":
                                    np.ascontiguousarray(flow[2:])})
    ffeeds = {"img0": p0, "img1": p1, "flow": np.ascontiguousarray(flow)}
    for i, k in enumerate(("f1", "f2", "f3", "f4")):
        ffeeds[str(3 + i)] = np.ascontiguousarray(c0[k])
        ffeeds[str(7 + i)] = np.ascontiguousarray(c1[k])
    mid_float = ref_exec.execute(ul, Wul, ffeeds)["output"][:, :H, :W]
    Mfl = CAL.calibrate(fl, Wfl, {"input0": p0, "input1": p1})
    Mcl0 = CAL.calibrate(cl, Wcl, {"input.1": p0, "flow.0": flow[:2]})
    Mcl1 = CAL.calibrate(cl, Wcl, {"input.1": p1, "flow.0": flow[2:]})
    Mul = CAL.calibrate(ul, Wul, ffeeds)
    if torch_backend:
        import torch
        import exec_torch as ET

        def hwc2chw(t):
            return (t[0].permute(2, 0, 1).contiguous(),
                    t[1].permute(2, 0, 1).contiguous(),
                    t[2].permute(2, 0, 1).contiguous())

        bifl = ET.execute_torch(fl, Wtfl, Mfl, {"input0": triples(p0),
                                                "input1": triples(p1)})
        torch.cuda.synchronize()
        flow_t = hwc2chw(bifl["flow"])
        f0t = tuple(x[:2].contiguous() for x in flow_t)
        f1t = tuple(x[2:].contiguous() for x in flow_t)
        bic0 = ET.execute_torch(cl, Wtcl, Mcl0, {"input.1": triples(p0), "flow.0": f0t})
        bic1 = ET.execute_torch(cl, Wtcl, Mcl1, {"input.1": triples(p1), "flow.0": f1t})
        torch.cuda.synchronize()
        ifeeds = {"img0": triples(p0), "img1": triples(p1), "flow": flow_t}
        for i, k in enumerate(("f1", "f2", "f3", "f4")):
            ifeeds[str(3 + i)] = hwc2chw(bic0[k])
            ifeeds[str(7 + i)] = hwc2chw(bic1[k])
        mid = ET.execute_torch(ul, Wtul, Mul, ifeeds)["output"]
        torch.cuda.synchronize()
        s, e, z = (x.cpu().numpy() for x in mid)
        mid_int = (S.decode(s.transpose(2, 0, 1), e.transpose(2, 0, 1))
                   * (1 - z.transpose(2, 0, 1))).astype(np.float64)[:, :H, :W]
    else:
        import rife_exec as exec_int

        def dec(t):
            s, e, z = t
            return (S.decode(s.transpose(2, 0, 1), e.transpose(2, 0, 1))
                    * (1 - z.transpose(2, 0, 1))).astype(np.float64)

        bifl = exec_int.execute_int(fl, Wtfl, Mfl, {"input0": triples(p0),
                                                   "input1": triples(p1)})
        flow_i = np.ascontiguousarray(dec(bifl["flow"]))

        def hwc2chw_n(t):
            return (t[0].transpose(2, 0, 1).copy(), t[1].transpose(2, 0, 1).copy(),
                    t[2].transpose(2, 0, 1).copy())

        bic0 = exec_int.execute_int(cl, Wtcl, Mcl0, {"input.1": triples(p0),
                                                     "flow.0": triples(flow_i[:2])})
        bic1 = exec_int.execute_int(cl, Wtcl, Mcl1, {"input.1": triples(p1),
                                                     "flow.0": triples(flow_i[2:])})
        ifeeds = {"img0": triples(p0), "img1": triples(p1), "flow": triples(flow_i)}
        for i, k in enumerate(("f1", "f2", "f3", "f4")):
            ifeeds[str(3 + i)] = hwc2chw_n(bic0[k])
            ifeeds[str(7 + i)] = hwc2chw_n(bic1[k])
        mid_int = dec(exec_int.execute_int(ul, Wtul, Mul, ifeeds)["output"])[:, :H, :W]
    a, v = mid_float.astype(np.float64), mid_int
    mse = float(((v - a) ** 2).mean())
    db = 10 * np.log10(1.0 / mse)
    print(f"int-vs-float {db:.1f}dB (bar >={BAR_DB}) — "
          f"{'GO' if db >= BAR_DB else 'NO-GO'}", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray((np.clip(v, 0, 1).transpose(1, 2, 0) * 255 + 0.5
                     ).astype(np.uint8)).save(out_path)
    print("saved", out_path)
    if db < BAR_DB:
        sys.exit(1)


if __name__ == "__main__":
    tb = "--torch" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--torch"]
    if len(args) != 3:
        sys.exit("usage: demo_interp.py <img0> <img1> <outmid> [--torch]")
    main(*args, torch_backend=tb)
