"""Torch-CUDA executor mirroring exec_int dispatch (bit-exact target).

Same structure/layouts (HWC triples), same M dict, same op semantics;
only the substrate changes (numpy -> torch int64 on CUDA). Every op
used here is gated by torch_parity.py (bit-exact vs numpy mirrors).
Feeds/Wt arrive as numpy triples (same as execute_int) and move to
device once at entry. Scalar consts encode via S.encode (setup-cost
only, cached by value).
"""
import numpy as np
import torch

import phi_core as S
import torch_emu as T

DEV = T.DEV
M_FLOW = None  # set from rife_exec at import (single source of truth)

WARP_AUDIT = {"flow_over_U16": 0}

_CONST_CACHE = {}


def _m_cov(M, blob):
    return M["cov:" + blob]


def _tv(t):
    return tuple(torch.as_tensor(x, device=DEV) if isinstance(x, np.ndarray)
                 else (x.to(DEV) if isinstance(x, torch.Tensor) else x)
                 for x in t)


def _twd(d):
    return {k: (torch.as_tensor(v, device=DEV) if isinstance(v, np.ndarray)
                else (v.to(DEV) if isinstance(v, torch.Tensor) else v))
            for k, v in d.items()}


def _WD(Wt, name, suffix):
    return _twd(Wt[name + "." + suffix])


def _WB(Wt, name):
    key = name + ".bias"
    if key not in Wt:
        return None
    d = Wt[key]
    return (d["s"].to(DEV) if isinstance(d["s"], torch.Tensor)
            else torch.as_tensor(d["s"], device=DEV),
            d["e"].to(DEV) if isinstance(d["e"], torch.Tensor)
            else torch.as_tensor(d["e"], device=DEV),
            d["z"].to(DEV) if isinstance(d["z"], torch.Tensor)
            else torch.as_tensor(d["z"], device=DEV))


def _const_like(sh, arr):
    key = (tuple(sh), float(arr))
    if key not in _CONST_CACHE:
        s, e, z = S.encode(np.atleast_1d(np.asarray(arr, dtype=np.float64)))
        assert s.size == 1, f"non-scalar const {arr}"
        _CONST_CACHE[key] = (
            torch.full(sh, int(s.flat[0]), dtype=torch.int8, device=DEV),
            torch.full(sh, int(e.flat[0]), dtype=torch.int32, device=DEV),
            torch.zeros(sh, dtype=torch.uint8, device=DEV))
    return _CONST_CACHE[key]


def _slope_triples(sl, shp):
    C = shp[2]
    s = sl["s"].to(DEV) if isinstance(sl["s"], torch.Tensor) \
        else torch.as_tensor(np.asarray(sl["s"]), device=DEV)
    e = sl["e"].to(DEV) if isinstance(sl["e"], torch.Tensor) \
        else torch.as_tensor(np.asarray(sl["e"]), device=DEV)
    if s.numel() == 1:
        return (torch.full(shp, int(s.flatten()[0]), dtype=torch.int8, device=DEV),
                torch.full(shp, int(e.flatten()[0]), dtype=torch.int32, device=DEV),
                torch.zeros(shp, dtype=torch.uint8, device=DEV))
    assert s.numel() == C
    return (s.reshape(1, 1, C).expand(shp).contiguous().to(torch.int8),
            e.reshape(1, 1, C).expand(shp).contiguous().to(torch.int32),
            torch.zeros(shp, dtype=torch.uint8, device=DEV))


def _act(L, o, M):
    import rife_ref
    p = L["p"]
    if 9 not in p:
        return o
    code = p[9]
    kind = rife_ref.ACT9.get(code)
    if kind == "leaky":
        v = p[-23310]
        slope = float(v[0]) if isinstance(v, list) else float(v)
        return T.prelu_int(o, _const_like(o[0].shape, slope))
    if kind == "sigmoid":
        o2 = T.sigmoid_int(o)
        sh = o[0].shape
        return o2[0].reshape(sh), o2[1].reshape(sh), o2[2].reshape(sh)
    raise SystemExit(f"error: fused act {kind}")


def _chw2hwc(t):
    t = _tv(t)
    return t[0].permute(1, 2, 0).contiguous(), t[1].permute(1, 2, 0).contiguous(), \
        t[2].permute(1, 2, 0).contiguous()


def hwc2chw(t):
    """Device-side HWC->CHW (chain handoffs stay GPU-resident; the numpy
    hwc2chw in e2e_int bounces through CPU and must not be used here)."""
    return t[0].permute(2, 0, 1).contiguous(), t[1].permute(2, 0, 1).contiguous(), \
        t[2].permute(2, 0, 1).contiguous()


def execute_torch(layers, Wt, M, feeds_chw):
    blobs = {k: _chw2hwc(v) for k, v in feeds_chw.items()}
    execute_torch.last = None
    for L in layers:
        execute_torch.last = L["name"]
        t, p, name = L["type"], L["p"], L["name"]
        if all(k in blobs for k in L["outs"]):
            continue
        ins = [blobs[k] for k in L["ins"]]
        if t == "Input":
            pass
        elif t == "Split":
            for k in L["outs"]:
                blobs[k] = ins[0]
        elif t == "Concat":
            ax = {0: 2, 1: 0, 2: 1}[p.get(0, 0)]
            blobs[L["outs"][0]] = (
                torch.cat([v[0] for v in ins], ax),
                torch.cat([v[1] for v in ins], ax).to(torch.int32),
                torch.cat([v[2] for v in ins], ax))
        elif t == "Crop":
            st, en, ax = p[-23309], p[-23310], p[-23311]
            mp = {0: 2, 1: 0, 2: 1}
            sl = [slice(None)] * 3
            dims = (ins[0][0].shape[2], ins[0][0].shape[0], ins[0][0].shape[1])
            for s, e, a in zip(st, en, ax):
                n = dims[a]
                s = 0 if s == -233 else (s if s >= 0 else n + s)
                e = n if e == -233 else (e if e > 0 else n + e)
                sl[mp[a]] = slice(s, min(n, e))
            tt = tuple(sl)
            blobs[L["outs"][0]] = tuple(v[tt].contiguous() for v in ins[0])
        elif t == "Convolution":
            kh, stride, pad = p[1], p.get(3, 1), p.get(4, 0)
            assert pad == kh // 2, f"{name}: pad {pad}"
            blobs[L["outs"][0]] = _act(
                L, T.phi_conv(ins[0][0], ins[0][1], ins[0][2],
                              _WD(Wt, name, "weight"), M[name], stride,
                              Wb=_WB(Wt, name)), M)
        elif t == "PReLU":
            sl = Wt[name + ".slope"]
            blobs[L["outs"][0]] = T.prelu_int(
                ins[0], _slope_triples(sl, tuple(ins[0][0].shape)))
        elif t == "BinaryOp":
            op = p.get(0, 0)
            if p.get(1, 0) == 1 and len(ins) == 1:
                sc = float(p[2])
                b = _const_like(tuple(ins[0][0].shape), sc)
                blobs[L["outs"][0]] = T.bin2(op, ins[0], b, _m_cov(M, L["ins"][0]))
            else:
                a, c = ins
                m_out = max(_m_cov(M, L["ins"][0]), _m_cov(M, L["ins"][1]))
                blobs[L["outs"][0]] = T.bin2(op, a, c, m_out)
        elif t == "Pooling":
            F = T.to_fixed(ins[0][0], ins[0][1], ins[0][2],
                           _m_cov(M, L["ins"][0])).reshape(ins[0][0].shape)
            H, W, C = F.shape
            tot = F.sum(dim=(0, 1))
            o = T.from_fixed(T.tdiv(tot, H * W), _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(1, 1, C), o[1].reshape(1, 1, C),
                                   o[2].reshape(1, 1, C))
        elif t == "Interp":
            _sh = tuple(ins[0][0].shape)
            assert len(_sh) == 3, f"{name}: interp input rank {_sh}"
            F = T.to_fixed(ins[0][0], ins[0][1], ins[0][2],
                           _m_cov(M, L["ins"][0])).reshape(_sh)
            H0, W0 = F.shape[0], F.shape[1]
            Ho, Wo = max(int(round(H0 * float(p[1]))), 1), \
                max(int(round(W0 * float(p[2]))), 1)
            C0 = F.shape[2]
            o = T.from_fixed(T.interp_fixed(F, float(p[1]), float(p[2])
                                            ).reshape(-1), _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(Ho, Wo, C0),
                                   o[1].reshape(Ho, Wo, C0),
                                   o[2].reshape(Ho, Wo, C0))
        elif t == "PixelShuffle":
            C = ins[0][0].shape[2]
            H, W = ins[0][0].shape[0], ins[0][0].shape[1]
            s, e, z = (v.permute(2, 0, 1).contiguous() for v in ins[0])
            G = s.shape[0] // 4
            oo = lambda v: v.reshape(G, 2, 2, H, W
                                    ).permute(0, 3, 1, 4, 2
                                              ).reshape(G, H * 2, W * 2
                                                        ).permute(1, 2, 0).contiguous()
            blobs[L["outs"][0]] = (oo(s), oo(e.to(torch.int64)).to(torch.int32),
                                   oo(z))
        elif t == "UnaryOp":
            assert p.get(0) == 1, f"{name}: only Neg"
            blobs[L["outs"][0]] = ((-ins[0][0].to(torch.int16)).to(torch.int8),
                                   ins[0][1], ins[0][2])
        elif t == "Clip":
            blobs[L["outs"][0]] = T.clip_int(
                ins[0], _const_like(tuple(ins[0][0].shape), float(p.get(0, -1e9))),
                _const_like(tuple(ins[0][0].shape), float(p.get(1, 1e9))))
        elif t == "rife.Warp":
            f, fl = ins
            F = T.to_fixed(f[0], f[1], f[2], _m_cov(M, L["ins"][0])).reshape(f[0].shape)
            _mfl = M.get("flow_m", M_FLOW)
            _div = M.get("flow_div", 1)
            Fq = (T.to_fixed(fl[0], fl[1], fl[2], _mfl).reshape(fl[0].shape)
                  .to(torch.int64) * int(_div))
            if bool(((fl[1].to(torch.int64) > _mfl) & (~fl[2].to(torch.bool))).any()):
                WARP_AUDIT["flow_over_U16"] += 1
            H0, W0, C0 = F.shape
            o = T.from_fixed(T.warp_fixed(F, Fq).reshape(-1), _m_cov(M, L["ins"][0]))
            blobs[L["outs"][0]] = (o[0].reshape(H0, W0, C0),
                                   o[1].reshape(H0, W0, C0),
                                   o[2].reshape(H0, W0, C0))
        elif t == "Deconvolution":
            stride, pad = p.get(3, 1), p.get(4, 0)
            blobs[L["outs"][0]] = _act(
                L, T.deconv_int(ins[0][0], ins[0][1], ins[0][2],
                                _WD(Wt, name, "weight"), M[name], stride, pad,
                                Wb=_WB(Wt, name)), M)
        elif t == "Sigmoid":
            o = T.sigmoid_int(ins[0])
            sh = tuple(ins[0][0].shape)
            blobs[L["outs"][0]] = (o[0].reshape(sh), o[1].reshape(sh), o[2].reshape(sh))
        else:
            raise SystemExit(f"error: unmapped op {t} at {name}")
    return blobs


import rife_exec as _EI
M_FLOW = _EI.M_FLOW
