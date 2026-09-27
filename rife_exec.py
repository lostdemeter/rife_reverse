"""Integer executor mirroring ref_exec dispatch (Phase 1).

LAYOUT: HWC triples everywhere in here (phi_conv native). Feeds arrive
CHW-float like ref_exec; converted once at entry (exact transpose).
Outputs transposed back for comparison. Convention split documented:
ref_exec = CHW/float, exec_int = HWC/triples, both internally proven.

Scales: M["<layer>"] m_acc; M["cov:<blob>"] m_cov; M_FLOW for flows
(q18-as-q14 convention, ~0.02% documented). BinaryOp add/sub at
max(ma,mc); mul exact. Sigmoid U=1 + audit. Clip via exact int compare.
"""
import math
import numpy as np

import phi_core as S
import rife_int as I

M_FLOW = S.BIAS + int(round(S.K * math.log(16.0) / S.LN_PHI))

WARP_AUDIT = {"flow_over_U16": 0}

_CONST_CACHE = {}


def _m_cov(M, blob):
    return M["cov:" + blob]


def _WD(Wt, name, suffix):
    d = Wt[name + "." + suffix]
    return {"s": d["s"], "e": d["e"], "z": d["z"]}


def _WB(Wt, name):
    # bias triples or None (RIFE convs mostly biased; student has none)
    key = name + ".bias"
    if key not in Wt:
        return None
    d = Wt[key]
    return (d["s"], d["e"], d["z"])


def _const_like(sh, arr):
    # scalars only (array constants use _slope_triples). atleast_1d because
    # 0-d arrays reject boolean item assignment (numpy 2). Cached by value:
    # S.encode uses float log (setup-class; must not run per-frame in loop).
    key = (tuple(sh), float(arr))
    if key not in _CONST_CACHE:
        s, e, z = S.encode(np.atleast_1d(np.asarray(arr, dtype=np.float64)))
        assert s.size == 1, f"non-scalar const {arr}"
        _CONST_CACHE[key] = (
            np.full(sh, s.flat[0], np.int8), np.full(sh, e.flat[0], np.int32),
            np.zeros(sh, np.uint8))
    return _CONST_CACHE[key]


def _slope_triples(sl, shp):
    # sl: {"s","e","z"} slope arrays, scalar or (C,); broadcast to (H,W,C)
    C = shp[2]
    if sl["s"].size == 1:
        return (np.full(shp, sl["s"].flat[0], np.int8),
                np.full(shp, sl["e"].flat[0], np.int32),
                np.zeros(shp, np.uint8))
    assert sl["s"].size == C, f"slope count {sl['s'].size} vs C {C}"
    return (np.broadcast_to(sl["s"].reshape(1, 1, C), shp).copy().astype(np.int8),
            np.broadcast_to(sl["e"].reshape(1, 1, C), shp).copy().astype(np.int32),
            np.zeros(shp, np.uint8))


def _cmp_gt(a, b):
    az, bz = a[2].astype(bool), b[2].astype(bool)
    out = np.zeros(a[0].shape, bool)
    live = ~az & ~bz
    same = live & (a[0] == b[0])
    out |= same & (a[1] > b[1]) & (a[0] > 0)
    out |= same & (a[1] < b[1]) & (a[0] < 0)
    out |= live & (a[0] > b[0])
    out |= ~az & bz & (a[0] > 0)
    return out


def _clip_int(t, lo, hi):
    lo_t = _const_like(t[0].shape, lo)
    hi_t = _const_like(t[0].shape, hi)
    s = np.where(_cmp_gt(lo_t, t), lo_t[0], t[0])
    e = np.where(_cmp_gt(lo_t, t), lo_t[1], t[1]).astype(np.int32)
    z = np.where(_cmp_gt(lo_t, t), lo_t[2], t[2]).astype(np.uint8)
    s2 = np.where(_cmp_gt((s, e, z), hi_t), hi_t[0], s)
    e2 = np.where(_cmp_gt((s, e, z), hi_t), hi_t[1], e).astype(np.int32)
    z2 = np.where(_cmp_gt((s, e, z), hi_t), hi_t[2], z).astype(np.uint8)
    return s2.astype(np.int8), e2, z2


def _bin2(op, a, ma, c, mc, m_out):
    if op == 2:
        s = (a[0].astype(np.int16) * c[0].astype(np.int16)).astype(np.int8)
        e = np.clip(a[1].astype(np.int64) + c[1].astype(np.int64) - S.BIAS,
                    0, S.NLVL - 1).astype(np.int32)
        return s, e, (a[2] | c[2]).astype(np.uint8)
    fa = S.to_fixed(a[0], a[1], a[2], m_out)
    fc = S.to_fixed(c[0], c[1], c[2], m_out)
    if op == 0:
        q = fa + fc
    elif op == 1:
        q = fa - fc
    elif op == 7:
        q = fc - fa
    else:
        raise SystemExit(f"error: binary op {op}")
    return S.from_fixed(q, m_out)


def _act(L, o):
    import rife_ref
    p = L["p"]
    if 9 not in p:
        return o
    code = p[9]
    kind = rife_ref.ACT9.get(code)
    if kind == "leaky":
        v = p[-23310]
        slope = float(v[0]) if isinstance(v, list) else float(v)
        return I.prelu_int(o, _const_like(o[0].shape, slope))
    if kind == "sigmoid":
        o2 = I.sigmoid_int(o)
        sh = o[0].shape
        return o2[0].reshape(sh), o2[1].reshape(sh), o2[2].reshape(sh)
    raise SystemExit(f"error: fused act {kind}")


def _chw2hwc(t):
    return t[0].transpose(1, 2, 0), t[1].transpose(1, 2, 0), t[2].transpose(1, 2, 0)


def _hwc2chw(t):
    return t[0].transpose(2, 0, 1), t[1].transpose(2, 0, 1), t[2].transpose(2, 0, 1)


def execute_int(layers, Wt, M, feeds_chw):
    # entry: CHW triples -> HWC; everything below is HWC.
    blobs = {k: _chw2hwc(v) for k, v in feeds_chw.items()}
    execute_int.last = None
    for L in layers:
        execute_int.last = L["name"]
        t, p, name = L["type"], L["p"], L["name"]
        if all(k in blobs for k in L["outs"]):
            continue  # override semantics (mirrors ref_exec)
        ins = [blobs[k] for k in L["ins"]]
        if t == "Input":
            pass
        elif t == "Split":
            for k in L["outs"]:
                blobs[k] = ins[0]
        elif t == "Concat":
            ax = {0: 2, 1: 0, 2: 1}[p.get(0, 0)]
            blobs[L["outs"][0]] = (
                np.concatenate([v[0] for v in ins], ax),
                np.concatenate([v[1] for v in ins], ax).astype(np.int32),
                np.concatenate([v[2] for v in ins], ax))
        elif t == "Crop":
            st, en, ax = p[-23309], p[-23310], p[-23311]
            mp = {0: 2, 1: 0, 2: 1}  # ncnn (C,H,W) axes -> HWC axes
            sl = [slice(None)] * 3
            dims = (ins[0][0].shape[2], ins[0][0].shape[0], ins[0][0].shape[1])
            for s, e, a in zip(st, en, ax):
                n = dims[a]
                s = 0 if s == -233 else (s if s >= 0 else n + s)
                e = n if e == -233 else (e if e > 0 else n + e)
                sl[mp[a]] = slice(s, min(n, e))
            tt = tuple(sl)
            blobs[L["outs"][0]] = tuple(np.ascontiguousarray(v[tt]) for v in ins[0])
        elif t == "Convolution":
            kh, stride, pad = p[1], p.get(3, 1), p.get(4, 0)
            assert pad == kh // 2, f"{name}: pad {pad}"
            blobs[L["outs"][0]] = _act(
                L, S.phi_conv(ins[0][0], ins[0][1], ins[0][2],
                              _WD(Wt, name, "weight"), M[name], stride,
                              Wb=_WB(Wt, name)))
        elif t == "InnerProduct":
            Wd = _WD(Wt, name, "weight")
            # ravel in CHW order (matches float mirror) via transpose
            c = _hwc2chw(ins[0])
            o = I.ip_int(c[0], c[1], c[2], Wd, M[name])
            oo = (o[0].reshape(-1, 1, 1), o[1].reshape(-1, 1, 1), o[2].reshape(-1, 1, 1))
            blobs[L["outs"][0]] = _act(L, oo)
        elif t == "PReLU":
            sl = Wt[name + ".slope"]
            blobs[L["outs"][0]] = I.prelu_int(ins[0], _slope_triples(sl, ins[0][0].shape))
        elif t == "BinaryOp":
            op = p.get(0, 0)
            if p.get(1, 0) == 1 and len(ins) == 1:
                sc = float(p[2])
                b = _const_like(ins[0][0].shape, sc)
                ma = _m_cov(M, L["ins"][0])
                blobs[L["outs"][0]] = _bin2(op, ins[0], ma, b, ma, ma)
            else:
                a, c = ins
                ma, mc = _m_cov(M, L["ins"][0]), _m_cov(M, L["ins"][1])
                blobs[L["outs"][0]] = _bin2(op, a, ma, c, mc, max(ma, mc))
        elif t == "Pooling":
            F = S.to_fixed(ins[0][0], ins[0][1], ins[0][2], _m_cov(M, L["ins"][0])
                           ).reshape(ins[0][0].shape)
            H, W, C = F.shape
            tot = F.sum(axis=(0, 1))
            o = S.from_fixed(S.tdiv(tot, H * W), _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(1, 1, C), o[1].reshape(1, 1, C),
                                   o[2].reshape(1, 1, C))
        elif t == "Interp":
            _sh = ins[0][0].shape
            assert len(_sh) == 3, f"{name}: interp input rank {_sh}"
            F = S.to_fixed(ins[0][0], ins[0][1], ins[0][2], _m_cov(M, L["ins"][0])
                           ).reshape(_sh)
            H0, W0 = F.shape[0], F.shape[1]
            Ho, Wo = max(int(round(H0 * float(p[1]))), 1), max(int(round(W0 * float(p[2]))), 1)
            C0 = F.shape[2]
            # NB: interp preserves fixed-point units -> decode at INPUT scale
            o = S.from_fixed(I.interp_fixed(F, float(p[1]), float(p[2])).reshape(-1),
                             _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(Ho, Wo, C0), o[1].reshape(Ho, Wo, C0),
                                   o[2].reshape(Ho, Wo, C0))
        elif t == "PixelShuffle":
            C = ins[0][0].shape[2]
            H, W = ins[0][0].shape[0], ins[0][0].shape[1]
            s, e, z = (v.transpose(2, 0, 1) for v in ins[0])  # HWC->CHW
            G = s.shape[0] // 4
            oo = lambda v: np.ascontiguousarray(
                v.reshape(G, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(G, H * 2, W * 2)
                ).transpose(1, 2, 0)
            blobs[L["outs"][0]] = (oo(s), oo(e.astype(np.int64)).astype(np.int32), oo(z))
        elif t == "UnaryOp":
            assert p.get(0) == 1, f"{name}: only Neg"
            blobs[L["outs"][0]] = ((-ins[0][0].astype(np.int16)).astype(np.int8),
                                   ins[0][1], ins[0][2])
        elif t == "Clip":
            blobs[L["outs"][0]] = _clip_int(ins[0], float(p.get(0, -1e9)), float(p.get(1, 1e9)))
        elif t == "rife.Warp":
            f, fl = ins
            F = S.to_fixed(f[0], f[1], f[2], _m_cov(M, L["ins"][0])).reshape(f[0].shape)
            # flow UNIT: calibrate-sized power-of-two covering scale.
            # to_fixed gives 2^-18 @ U; warp wants 2^-14 px = v*2^14,
            # i.e. q *= U/16 = div (EXACT multiply, no rounding).
            # div=1 reproduces the legacy M_FLOW path bit-exactly.
            _mfl = M.get("flow_m", M_FLOW)
            _div = M.get("flow_div", 1)
            Fq = (S.to_fixed(fl[0], fl[1], fl[2], _mfl).reshape(fl[0].shape)
                  .astype(np.int64) * np.int64(_div))
            if bool(((fl[1].astype(np.int64) > _mfl) & (~fl[2].astype(bool))).any()):
                WARP_AUDIT["flow_over_U16"] += 1
            H0, W0, C0 = F.shape
            # NB: warp output is resampled image -> decode at IMAGE (ins[0]) scale
            o = S.from_fixed(I.warp_fixed(F, Fq).reshape(-1), _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(H0, W0, C0), o[1].reshape(H0, W0, C0),
                                   o[2].reshape(H0, W0, C0))
        elif t == "Deconvolution":
            stride, pad = p.get(3, 1), p.get(4, 0)
            blobs[L["outs"][0]] = _act(
                L, I.deconv_int(ins[0][0], ins[0][1], ins[0][2],
                                _WD(Wt, name, "weight"), M[name], stride, pad,
                                Wb=_WB(Wt, name)))
        elif t == "Sigmoid":
            o = I.sigmoid_int(ins[0])
            sh = ins[0][0].shape
            blobs[L["outs"][0]] = (o[0].reshape(sh), o[1].reshape(sh), o[2].reshape(sh))
        else:
            raise SystemExit(f"error: unmapped op {t} at {name}")
    return blobs


WARP_AUDIT = {"flow_over_U16": 0}
