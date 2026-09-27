"""Numpy reference executor for the RIFE ncnn subset (Phase 0).

Env hypothesis flags (sweep cells): RIFE_SHUF=DCR, RIFE_IPT=1.

Semantics ported 1:1 from verified sources (see header notes per op).
Layout (C,H,W) float32. Shape-traced: every blob's shape logged; channel
mismatches fail loud at the op (no silent broadcast).
"""
import math
import os
import numpy as np

ACT9 = {2: "leaky", 4: "sigmoid"}  # 9=2 LeakyReLU(slope), 9=4 Sigmoid; gate arbitrates
NETNAME = "?"  # set by caller per net (hypothesis scoping)

UNARY = {0: "abs", 1: "neg", 2: "floor", 3: "ceil", 4: "square", 5: "sqrt",
         6: "rsq", 7: "exp", 8: "log", 9: "sin", 10: "cos", 11: "tan"}


def apply_fused_act(x, L):
    p = L["p"]
    if 9 not in p:
        return x
    code, name = p[9], L["name"]
    kind = ACT9.get(code)
    if kind is None:
        raise SystemExit(f"error: unknown fused activation 9={code} at {name}")
    if kind == "leaky01":
        return np.where(x >= 0, x, x * np.float32(0.1)).astype(np.float32)
    if kind == "leaky":
        slope = 0.1
        if -23310 in p:
            v = p[-23310]
            slope = float(v[0]) if isinstance(v, list) else float(v)
        return np.where(x >= 0, x, x * np.float32(slope)).astype(np.float32)
    if kind == "relu":
        return np.where(x >= 0, x, 0).astype(np.float32)
    if kind == "elu":
        a = 1.0
        if -23310 in p:
            v = p[-23310]
            a = float(v[0]) if isinstance(v, list) else float(v)
        return np.where(x >= 0, x, a * (np.exp(x.astype(np.float64)) - 1)).astype(np.float32)
    if kind == "sigmoid":
        return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    raise SystemExit(f"error: unhandled fused act {kind}")


def conv2d(x, W, b, stride=1, pad=0):
    Cout, Cin, kh, kw = W.shape
    C, H, Wd = x.shape
    assert C == Cin, f"conv ch {C} != {Cin}"
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kw) // stride + 1
    out = np.zeros((Cout, Ho, Wo), np.float32)
    # NOTE: columns need stride too (a missing *stride here broke every
    # stride-2 conv while stride-1 stayed exact; caught by torch unit test).
    rr = (np.arange(Ho) * stride)[:, None] + np.zeros(Wo, np.int64)[None, :]
    cc = (np.arange(Wo) * stride)[None, :] + np.zeros(Ho, np.int64)[:, None]
    for ky in range(kh):
        for kx in range(kw):
            P = xp[:, rr + ky, cc + kx]  # (Cin,Ho,Wo) broadcast fancy grids
            out += (P.transpose(1, 2, 0) @ W[:, :, ky, kx].T).transpose(2, 0, 1)
    if b is not None:
        out += b.reshape(-1, 1, 1)
    return out.astype(np.float32)


def deconv2d(x, W, b, stride=2, pad=1):
    # transpose-conv via scatter-add (reference clarity over speed)
    Cout, Cin, kh, kw = W.shape
    C, H, Wd = x.shape
    assert C == Cin
    Ho, Wo = (H - 1) * stride - 2 * pad + kh, (Wd - 1) * stride - 2 * pad + kw
    out = np.zeros((Cout, Ho, Wo), np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            acc = np.zeros(Cout)
            for ky in range(kh):
                iy = oy + pad - ky
                if iy % stride or (iy // stride) >= H or iy < 0:
                    continue
                for kx in range(kw):
                    ix = ox + pad - kx
                    if ix % stride or (ix // stride) >= Wd or ix < 0:
                        continue
                    acc += W[:, :, ky, kx] @ x[:, iy // stride, ix // stride]
            out[:, oy, ox] = acc
    if b is not None:
        out += b.reshape(-1, 1, 1)
    return out.astype(np.float32)


def interp_bilinear(x, sy, sx):
    # default: align_corners=False half-pixel. RIFE_ALIGN_UP/DOWN select
    # align_corners=True per direction (ncnn default ambiguous; gate decides).
    import os as _os
    C, H, W = x.shape
    Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)
    up = sy > 1 or sx > 1
    ac = (_os.environ.get("RIFE_ALIGN_UP") == "1" and up) or \
         (_os.environ.get("RIFE_ALIGN_DOWN") == "1" and not up)
    if ac:
        gy, gx = np.meshgrid(np.linspace(0, H - 1, Ho), np.linspace(0, W - 1, Wo), indexing="ij")
    else:
        gy, gx = np.meshgrid((np.arange(Ho) + 0.5) / sy - 0.5, (np.arange(Wo) + 0.5) / sx - 0.5, indexing="ij")
    return sample_bilinear(x, gx, gy)


def sample_bilinear(x, gx, gy):
    C, H, W = x.shape
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    ax = np.clip(gx - x0, 0, 1)
    ay = np.clip(gy - y0, 0, 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y0 + 1, 0, H - 1)
    return (x[:, y0c, x0c] * (1 - ax) * (1 - ay) +
            x[:, y0c, x1c] * ax * (1 - ay) +
            x[:, y1c, x0c] * (1 - ax) * ay +
            x[:, y1c, x1c] * ax * ay).astype(np.float32)


def warp_nihui(img, flow):
    """nihui warp.cpp EXACT: sample = p + flow, floor UNclamped for alpha,
    indices clamped (border replicate). NOTE: differs from warp_int which
    clips coords first — border-only divergence, C port must use THIS form."""
    C, H, W = img.shape
    assert flow.shape[0] == 2
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    sx = gx + flow[0].astype(np.float64)
    sy = gy + flow[1].astype(np.float64)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    ax = np.clip(sx - x0, 0, 1)
    ay = np.clip(sy - y0, 0, 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y0 + 1, 0, H - 1)
    return (img[:, y0c, x0c] * (1 - ax) * (1 - ay) +
            img[:, y0c, x1c] * ax * (1 - ay) +
            img[:, y1c, x0c] * (1 - ax) * ay +
            img[:, y1c, x1c] * ax * ay).astype(np.float32)


def execute(layers, weights, feeds, trace=None):
    # feeds pre-populate blobs; layers whose outputs are ALL already present
    # are SKIPPED (ncnn Extractor.input() override semantics — load-bearing:
    # rife.cpp feeds intermediate blob "flow.0" to override the Neg on side0).
    blobs = dict(feeds)
    execute.last = None
    for L in layers:
        execute.last = L["name"]  # debuggability: failing op names itself
        if all(k in blobs for k in L["outs"]):
            if trace is not None:
                trace.append((L["name"], L["type"] + " (override-skip)", tuple(blobs[L["outs"][0]].shape), 0.0))
            continue
        t, p, name = L["type"], L["p"], L["name"]
        ins = [blobs[k] for k in L["ins"]]
        if t == "Input":
            pass  # already in feeds
        elif t == "Split":
            for k in L["outs"]:
                blobs[k] = ins[0]
        elif t == "Concat":
            axis = p.get(0, 0)
            assert axis == 0, f"{name}: concat axis {axis} != channel"
            blobs[L["outs"][0]] = np.concatenate(ins, axis=0)
        elif t == "Crop":
            # numpy-style starts/ends/axes (-23309/-23310/-23311)
            st, en, ax = p.get(-23309), p.get(-23310), p.get(-23311)
            assert isinstance(st, list) and isinstance(en, list), f"{name}: crop form"
            axes = ax if isinstance(ax, list) else list(range(len(st)))
            sl = [slice(None)] * 3  # (C,H,W): axis0=C,1=H,2=W
            dims = ins[0].shape
            for s, e, a in zip(st, en, axes):
                n = dims[a]
                s = 0 if s == -233 else (s if s >= 0 else n + s)
                e = n if e == -233 else (e if e > 0 else n + e)
                sl[a] = slice(s, min(n, e))
            blobs[L["outs"][0]] = np.ascontiguousarray(ins[0][tuple(sl)])
        elif t == "Convolution":
            kh = p[1]
            stride = p.get(3, 1)
            pad = p.get(4, 0)
            b = weights.get(name + ".bias")
            blobs[L["outs"][0]] = apply_fused_act(
                conv2d(ins[0], weights[name + ".weight"], b, stride, pad), L)
        elif t == "InnerProduct":
            b = weights.get(name + ".bias")
            Wip = weights[name + ".weight"]
            if os.environ.get("RIFE_IPT") == "1":
                v = Wip.T @ ins[0].ravel()
            else:
                v = Wip @ ins[0].ravel()
            if b is not None:
                v = v + b
            blobs[L["outs"][0]] = apply_fused_act(v.astype(np.float32).reshape(-1, 1, 1), L)
        elif t == "PReLU":
            s = weights[name + ".slope"]
            x = ins[0]
            if s.size == 1:
                blobs[L["outs"][0]] = np.where(x >= 0, x, x * s[0]).astype(np.float32)
            else:
                assert s.size == x.shape[0], f"{name}: prelu slopes {s.size} vs ch {x.shape[0]}"
                blobs[L["outs"][0]] = np.where(x >= 0, x, x * s.reshape(-1, 1, 1)).astype(np.float32)
        elif t == "BinaryOp":
            op = p.get(0, 0)
            if p.get(1, 0) == 1 and len(ins) == 1:
                sc = float(p[2])
                a = ins[0]
                blobs[L["outs"][0]] = {(0): a + sc, (1): a - sc, (2): a * sc,
                                       (7): sc - a}.get(op, None).astype(np.float32) \
                    if op in (0, 1, 2, 7) else _die(name, f"scalar op {op}")
            else:
                assert len(ins) == 2, f"{name}: binaryop inputs"
                a, c = ins
                blobs[L["outs"][0]] = {(0): a + c, (1): a - c, (2): a * c,
                                       (7): c - a}.get(op, None).astype(np.float32) \
                    if op in (0, 1, 2, 7) else _die(name, f"op {op}")
        elif t == "Pooling":
            assert p.get(0) == 1 and p.get(4) == 1, f"{name}: only global-avg supported, got {p}"
            blobs[L["outs"][0]] = ins[0].mean(axis=(1, 2), keepdims=True).astype(np.float32)
        elif t == "Interp":
            assert p.get(0) == 2, f"{name}: only bilinear interp, got {p}"
            blobs[L["outs"][0]] = interp_bilinear(ins[0], float(p[1]), float(p[2]))
        elif t == "PixelShuffle":
            assert p.get(0) == 2, f"{name}: only x2 shuffle"
            C, H, W = ins[0].shape
            assert C % 4 == 0
            if os.environ.get("RIFE_SHUF") == "DCR":
                # alternative channel interleave (which one matches ncnn?)
                # out[g,2h+a,2w+b] = in[(a*2+b)*G+g]: gather over output
                # order (g outer, ab inner). (First attempt had axes
                # transposed = identity; caught by unit test.)
                G = C // 4
                ab = np.arange(4)
                gather = (ab[None, :] * G + np.arange(G)[:, None]).reshape(-1)
                blobs[L["outs"][0]] = np.ascontiguousarray(
                    ins[0].reshape(C // 4, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(C // 4, H * 2, W * 2) * 0
                    + ins[0][gather].reshape(G, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(G, H * 2, W * 2))
            else:
                blobs[L["outs"][0]] = np.ascontiguousarray(
                    ins[0].reshape(C // 4, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(C // 4, H * 2, W * 2))
        elif t == "UnaryOp":
            code = p.get(0)
            x = ins[0]
            fn = {0: (lambda v: np.abs(v)), 1: (lambda v: -v)}.get(code)
            if fn is None:
                raise SystemExit(f"error: UnaryOp code {code} at {name}")
            blobs[L["outs"][0]] = fn(x).astype(np.float32)
        elif t == "Clip":
            blobs[L["outs"][0]] = np.clip(ins[0], float(p.get(0, -1e9)), float(p.get(1, 1e9))).astype(np.float32)
        elif t == "rife.Warp":
            f2 = ins[1]
            if os.environ.get("RIFE_WNEG") == "1":
                f2 = -f2  # hypothesis: sampling convention flipped (blobs intact)
            if os.environ.get("RIFE_WHALF") == "1" and NETNAME in ("contextnet", "fusionnet"):
                # hypothesis: FINAL-midpoint warps use halved fields.
                # Flownet-internal coarse-to-fine warps stay full (PWC-style).
                f2 = f2 / 2.0
            _ws = os.environ.get("RIFE_WSCALE")
            if _ws is not None and NETNAME in ("contextnet", "fusionnet"):
                # scale sweep for systematic warp-magnitude error
                f2 = f2 * float(_ws)
            blobs[L["outs"][0]] = warp_nihui(ins[0], f2)
        elif t == "Deconvolution":
            stride, pad = p.get(3, 1), p.get(4, 0)
            b = weights.get(name + ".bias")
            blobs[L["outs"][0]] = apply_fused_act(
                deconv2d(ins[0], weights[name + ".weight"], b, stride, pad), L)
        elif t == "Eltwise":
            raise SystemExit(f"error: Eltwise at {name} (v4.6-only op; default-rife port stops here)")
        elif t == "Sigmoid":
            blobs[L["outs"][0]] = (1.0 / (1.0 + np.exp(-ins[0].astype(np.float64)))).astype(np.float32)
        else:
            raise SystemExit(f"error: unmapped op {t} at {name} (coverage gap — see ledger)")
        if os.environ.get("RIFE_FP16STORE") == "1" and t != "Input" and L["outs"][0] in blobs:
            # emulate binary's fp16 blob storage (store+reload each op)
            blobs[L["outs"][0]] = blobs[L["outs"][0]].astype(np.float16).astype(np.float32)
        if trace is not None:
            o = blobs[L["outs"][0]]
            trace.append((name, t, tuple(o.shape), float(np.abs(o).max())))
    return blobs


def _die(name, msg):
    raise SystemExit(f"error: {name}: {msg}")
