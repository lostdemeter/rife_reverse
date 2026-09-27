"""Torch (CUDA) port of the phi integer datapath — BIT-EXACT vs numpy emu.

All arithmetic is integer (int64); sums are order-free so restructured
kernels (unfold/matmul) match the numpy mirrors bit-for-bit. Float
appears ONLY in from_fixed's frexp (torch.frexp == np.frexp semantics,
verified in torch_parity.py) and offline LUT loads (same frozen .npy).
Non-exact op => must fail torch_parity before use in exec_torch.
"""
import os
import numpy as np
import torch

BIAS, K, NLVL = 32768, 512, 65536
FRAC_CAP, FIXED_F = 13312, 18
LUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_T = {}


def _load(name):
    if name not in _T:
        for d in (LUTDIR,):
            p = os.path.join(d, name + ".npy")
            if os.path.exists(p):
                _T[name] = torch.from_numpy(np.load(p)).to(DEV)
                break
        else:
            raise FileNotFoundError(name)
    return _T[name]


def L_FRAC():
    return _load("frac_lut")


def L_COARSE():
    return _load("coarse_lut")


def L_FINE():
    return _load("fine_lut")


def sig_lut():
    return _load("sig_lut")


def sigx_lut():
    return _load("sigx_lut")


def bit_length_int(a):
    """Integer bit_length (floor(log2)+1, 0 for a=0) — pure integer.

    Replaces frexp (which needs float64/FPU). Compare-shift cascade,
    exact over full int64 (frexp was exact only < 2^53 — this is
    strictly more general). Gated by torch_parity (edge values).
    """
    bl = torch.zeros(a.shape, dtype=torch.int64, device=a.device)
    x = a
    for sh in (32, 16, 8, 4, 2, 1):
        big = (x >> sh) > 0
        bl = bl + big.to(torch.int64) * sh
        x = torch.where(big, x >> sh, x)
    return bl + (a > 0).to(torch.int64)


def to_fixed(s, e, z, m):
    """(s,e,z) triples -> int64 2^-18 @ U(m). Exact mirror of phi_core."""
    F = L_FRAC()
    base = F[0]
    d = int(m) - e.to(torch.int64)
    zz = z.to(torch.bool)
    out = torch.where(zz, torch.zeros((), dtype=torch.int64, device=DEV),
            torch.where(d < 0, s.to(torch.int64) * base,
              torch.where(d > FRAC_CAP, torch.zeros((), dtype=torch.int64, device=DEV),
                s.to(torch.int64) * F[torch.clamp(d, 0, FRAC_CAP)])))
    return out


def tdiv(a, b):
    """Truncating division (C semantics)."""
    return torch.div(a.to(torch.int64),
                     torch.as_tensor(b, dtype=torch.int64, device=a.device),
                     rounding_mode="trunc")


def from_fixed(q, m):
    """int64 2^-18 -> triples. Exact mirror (integer bit-length path)."""
    q = q.to(torch.int64)
    z = (q == 0)
    a = torch.abs(q)
    s = torch.where(q > 0, torch.ones((), dtype=torch.int8, device=DEV),
                    torch.full((), -1, dtype=torch.int8, device=DEV))
    bl = bit_length_int(a)
    nz = a > 0
    shift = bl - 15
    mant = torch.where(shift >= 0, a >> torch.clamp(shift, min=0),
                       a << torch.clamp(-shift, min=0))
    t = shift + 14 - FIXED_F
    idx = t + 64
    C, FN = L_COARSE(), L_FINE()
    e = (int(m) + C[torch.clamp(idx, 0, 192)]
         + FN[torch.clamp(mant - 16384, 0, 16383)])
    e = torch.clamp(e, 0, 65535).to(torch.int32)
    return s, e, z.to(torch.uint8)


def tmul(a, b):
    s = (a[0].to(torch.int16) * b[0].to(torch.int16)).to(torch.int8)
    e = torch.clamp(a[1].to(torch.int64) + b[1].to(torch.int64) - BIAS,
                    0, NLVL - 1).to(torch.int32)
    return s, e, (a[2] | b[2]).to(torch.uint8)


def sigmoid_int(t, audit=None):
    X = sigx_lut()
    G = sig_lut()
    SPAN = 16 * 16384
    e = torch.clamp(t[1].to(torch.int64), 0, 65535)
    x14 = torch.where(t[2].to(torch.bool), torch.zeros((), dtype=torch.int64, device=DEV),
                      t[0].to(torch.int64) * X[e])
    idx = x14 + SPAN
    y14 = torch.where(x14 < -SPAN, torch.zeros((), dtype=torch.int64, device=DEV),
            torch.where(x14 > SPAN, torch.full((), 16384, dtype=torch.int64, device=DEV),
                        G[torch.clamp(idx, 0, len(G) - 1)]))
    return from_fixed(y14 * np.int64(16), BIAS)


def prelu_int(t, slope):
    neg = (t[0].to(torch.int16) < 0) & (~t[2].to(torch.bool))
    ps, pe, pz = tmul(t, slope)
    return (torch.where(neg, ps, t[0]),
            torch.where(neg, pe, t[1]).to(torch.int32),
            torch.where(neg, pz, t[2]).to(torch.uint8))


def bilinear_sample_fixed(F, X, Y):
    """Exact mirror of rife_int.bilinear_sample_fixed (edge replicate)."""
    H, W, C = F.shape
    ONE = 1 << 14
    Xc = torch.clamp(X, 0, (W - 1) << 14)
    Yc = torch.clamp(Y, 0, (H - 1) << 14)
    xi, xf = Xc >> 14, Xc & (ONE - 1)
    yi, yf = Yc >> 14, Yc & (ONE - 1)
    xi1 = torch.minimum(xi + 1, torch.as_tensor(W - 1, device=DEV))
    yi1 = torch.minimum(yi + 1, torch.as_tensor(H - 1, device=DEV))
    tot = (F[yi, xi] * (ONE - xf)[..., None] * (ONE - yf)[..., None]
           + F[yi, xi1] * xf[..., None] * (ONE - yf)[..., None]
           + F[yi1, xi] * (ONE - xf)[..., None] * yf[..., None]
           + F[yi1, xi1] * xf[..., None] * yf[..., None]) >> 28
    return tot


def warp_fixed(feat_q, flow_q):
    H, W, C = feat_q.shape
    gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.int64, device=DEV),
                            torch.arange(W, dtype=torch.int64, device=DEV),
                            indexing="ij")
    X = (gx << 14)[..., None] + flow_q[..., 0:1]
    Y = (gy << 14)[..., None] + flow_q[..., 1:2]
    return bilinear_sample_fixed(feat_q, X[..., 0], Y[..., 0])


def _coord(n_out, s):
    o = torch.arange(n_out, dtype=torch.int64, device=DEV)
    for k in range(1, 14):
        if s == 2.0 ** -k:
            return ((2 * o + 1) << (13 + k)) - 8192
        if s == 2.0 ** k:
            return (((2 * o + 1) - (1 << k)) << (13 - k))
    if s == 1.0:
        return o << 14
    num = ((2 * o + 1) << 14)
    den = int(round(2.0 / s)) if s != 0 else 1
    return torch.div(num, torch.as_tensor(den, device=DEV), rounding_mode="trunc")


def interp_fixed(feat_q, sy, sx):
    """Exact mirror of rife_int.interp_fixed (all dyadic scales)."""
    H, W, C = feat_q.shape
    Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)
    X = _coord(Wo, sx)
    Y = _coord(Ho, sy)
    return bilinear_sample_fixed(feat_q, X[None, :].expand(Ho, Wo),
                                 Y[:, None].expand(Ho, Wo))


def deconv_int(s, e, z, Wd, m_out, stride=2, pad=1, Wb=None, co_chunk=32):
    """Bit-exact mirror of rife_int.deconv_int (scatter-add over bridge).

    Same math, restructured: per (ky,kx), all-co/all-ci product triples
    -> to_fixed -> sum ci -> index_add into acc. Integer-exact.
    """
    H, W, Cin = s.shape
    Cout, kh = Wd["s"].shape[0], Wd["s"].shape[1]
    Ho, Wo = (H - 1) * stride - 2 * pad + kh, (W - 1) * stride - 2 * pad + kh
    acc = torch.zeros((Ho, Wo, Cout), dtype=torch.int64, device=DEV)
    if Wb is not None:
        acc = acc + to_fixed(Wb[0], Wb[1], Wb[2], m_out).reshape(1, 1, -1)
    si, ei, zi = s.to(torch.int64), e.to(torch.int64), z.to(torch.int64)
    for ky in range(kh):
        for kx in range(kh):
            ys = ky - pad + torch.arange(H, device=DEV) * stride
            xs = kx - pad + torch.arange(W, device=DEV) * stride
            oky = (ys >= 0) & (ys < Ho)
            okx = (xs >= 0) & (xs < Wo)
            if not bool(oky.any()) or not bool(okx.any()):
                continue
            ih = torch.arange(H, device=DEV)[oky]
            iw = torch.arange(W, device=DEV)[okx]
            oy = ky - pad + ih * stride
            ox = kx - pad + iw * stride
            a_s = si[ih][:, iw]          # (nh, nw, Ci)
            a_e = ei[ih][:, iw]
            a_z = zi[ih][:, iw]
            nh, nw = len(ih), len(iw)
            for c0 in range(0, Cout, co_chunk):
                c1 = min(Cout, c0 + co_chunk)
                w_s = Wd["s"][c0:c1, ky, kx, :]      # (cc, Ci)
                w_e = Wd["e"][c0:c1, ky, kx, :].to(torch.int64)
                w_z = Wd["z"][c0:c1, ky, kx, :]
                # (nh,nw,Ci,cc): a (nh,nw,Ci,1) x W.T (1,1,Ci,cc), sum Ci
                q_s = (a_s[..., None].to(torch.int16)
                       * w_s.t()[None, None].to(torch.int16)).to(torch.int8)
                q_e = torch.clamp(a_e[..., None] + w_e.t()[None, None]
                                  - BIAS, 0, NLVL - 1)
                q_z = (a_z[..., None] | w_z.t()[None, None])
                sub = to_fixed(q_s.reshape(-1), q_e.reshape(-1).to(torch.int32),
                               q_z.reshape(-1), m_out
                               ).reshape(nh, nw, -1, c1 - c0).sum(2)
                # combined advanced index (in-place); chained [] would copy.
                # channel slice required: co-chunks tile Cout (cf. phi_conv
                # column-slice lesson).
                acc[oy[:, None], ox, c0:c1] += sub
    so, eo, zo = from_fixed(acc.reshape(-1), m_out)
    return (so.reshape(Ho, Wo, Cout), eo.reshape(Ho, Wo, Cout),
            zo.reshape(Ho, Wo, Cout))


def cmp_gt(a, b):
    az, bz = a[2].to(torch.bool), b[2].to(torch.bool)
    out = torch.zeros(a[0].shape, dtype=torch.bool, device=DEV)
    live = ~az & ~bz
    same = live & (a[0] == b[0])
    out |= same & (a[1] > b[1]) & (a[0] > 0)
    out |= same & (a[1] < b[1]) & (a[0] < 0)
    out |= live & (a[0] > b[0])
    out |= ~az & bz & (a[0] > 0)
    return out


def clip_int(t, lo_t, hi_t):
    s = torch.where(cmp_gt(lo_t, t), lo_t[0], t[0])
    e = torch.where(cmp_gt(lo_t, t), lo_t[1], t[1]).to(torch.int32)
    z = torch.where(cmp_gt(lo_t, t), lo_t[2], t[2]).to(torch.uint8)
    s2 = torch.where(cmp_gt((s, e, z), hi_t), hi_t[0], s)
    e2 = torch.where(cmp_gt((s, e, z), hi_t), hi_t[1], e).to(torch.int32)
    z2 = torch.where(cmp_gt((s, e, z), hi_t), hi_t[2], z).to(torch.uint8)
    return s2.to(torch.int8), e2, z2


def bin2(op, a, c, m_out):
    """_bin2 mirror (ma/mc/m_out folded by caller convention: exec always
    converts both sides to m_out first for add/sub; MUL is scale-free)."""
    if op == 2:
        s = (a[0].to(torch.int16) * c[0].to(torch.int16)).to(torch.int8)
        e = torch.clamp(a[1].to(torch.int64) + c[1].to(torch.int64) - BIAS,
                        0, NLVL - 1).to(torch.int32)
        return s, e, (a[2] | c[2]).to(torch.uint8)
    fa = to_fixed(a[0], a[1], a[2], m_out)
    fc = to_fixed(c[0], c[1], c[2], m_out)
    if op == 0:
        q = fa + fc
    elif op == 1:
        q = fa - fc
    elif op == 7:
        q = fc - fa
    else:
        raise SystemExit(f"error: binary op {op}")
    return from_fixed(q, m_out)


_IM2COL = {}


def _im2col_index(Hp, Wp, Ho, Wo, kh, stride):
    """Frozen gather table for im2col (pure-integer runtime).

    F.unfold lacks int64 kernels, and the old float64 detour used the FPU.
    This table is built once per geometry (numpy, setup) and cached; the
    runtime path is a single integer gather. Values identical to unfold.
    idx[o, ky*kh+kx] = flat offset of tap (ky,kx) for output pixel o
    (row-major), matching numpy tap order (ky,kx,ci).
    """
    key = (Hp, Wp, Ho, Wo, kh, stride)
    if key not in _IM2COL:
        oy, ox = np.meshgrid(np.arange(Ho) * stride, np.arange(Wo) * stride,
                             indexing="ij")
        base = (oy * Wp + ox).reshape(-1, 1)
        taps = np.array([[ky * Wp + kx for ky in range(kh) for kx in range(kh)]],
                        dtype=np.int64)
        _IM2COL[key] = torch.as_tensor(base + taps, dtype=torch.int64, device=DEV)
    return _IM2COL[key]


def phi_conv(s, e, z, W, m_out, stride=1, Wb=None, co_chunk=32, n_tile=256):
    """Bit-exact torch mirror of phi_core.phi_conv (see that docstring).

    Same math, restructured: pad once, frozen-gather patches, per-tile
    product triples -> to_fixed -> exact int64 sum over taps -> from_fixed.
    Integer sums are order-free => bit-exact vs the numpy triple loop.
    Tiling (co_chunk x n_tile) bounds memory; tiling never changes values.
    Layouts: s,e,z (H,W,Ci); W dict (Cout,kh,kh,Cin); returns (Ho,Wo,Cout).
    Zero float in the loop: im2col is a frozen integer gather (no unfold
    float64 detour); see from_fixed for the frexp-free bit-length path.
    """
    H, Wd, Cin = s.shape
    Cout, kh = W["s"].shape[0], W["s"].shape[1]
    assert Cin == W["s"].shape[3]
    pad = kh // 2 if kh == 3 else 0
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kh) // stride + 1
    N = Ho * Wo
    import torch.nn.functional as Fn
    ps = Fn.pad(s.to(torch.int64), (0, 0, pad, pad, pad, pad)).to(torch.int8)
    pe = Fn.pad(e.to(torch.int64), (0, 0, pad, pad, pad, pad)).to(torch.int32)
    pz = Fn.pad(z.to(torch.int64), (0, 0, pad, pad, pad, pad)).to(torch.uint8)
    # NOTE: numpy pads s/e/z with 0 (pad-region contributes via d>FRAC_CAP->0
    # in to_fixed since z=0 but e=0 => d=m huge). Replicated here exactly.
    idx = _im2col_index(H + 2 * pad, Wd + 2 * pad, Ho, Wo, kh, stride)
    TAPS = kh * kh * Cin
    Ps = ps.reshape(-1, Cin)[idx].reshape(N, TAPS)
    Pe = pe.reshape(-1, Cin)[idx].reshape(N, TAPS)
    Pz = pz.reshape(-1, Cin)[idx].reshape(N, TAPS)
    Ws = W["s"].reshape(Cout, TAPS)
    We = W["e"].reshape(Cout, TAPS).to(torch.int64)
    Wz = W["z"].reshape(Cout, TAPS)
    out_s = torch.empty((N, Cout), dtype=torch.int8, device=DEV)
    out_e = torch.empty((N, Cout), dtype=torch.int32, device=DEV)
    out_z = torch.empty((N, Cout), dtype=torch.uint8, device=DEV)
    for c0 in range(0, Cout, co_chunk):
        c1 = min(Cout, c0 + co_chunk)
        cc = c1 - c0
        for n0 in range(0, N, n_tile):
            n1 = min(N, n0 + n_tile)
            # (Nt, taps, cc): Ps (Nt,T,1) x W.T (1,T,cc)
            q_s = (Ps[n0:n1][:, :, None].to(torch.int16)
                   * Ws[c0:c1].t()[None, :, :].to(torch.int16)).to(torch.int8)
            q_e = torch.clamp(Pe[n0:n1][:, :, None].to(torch.int64)
                              + We[c0:c1].t()[None, :, :] - BIAS, 0, NLVL - 1)
            q_z = (Pz[n0:n1][:, :, None] | Wz[c0:c1].t()[None, :, :])
            # NOTE (measured 2026-09-27): int32 products + promote-at-sum
            # was tried and REVERTED (extra cast launches cost more than
            # the bandwidth saves at these shapes: 62s vs 57s/720p pair).
            # Traffic wall stands; fused kernels (Phase 2) are the answer.
            acc = to_fixed(q_s.reshape(-1), q_e.reshape(-1).to(torch.int32),
                           q_z.reshape(-1), m_out).reshape(n1 - n0, -1, cc).sum(1)
            if Wb is not None:
                acc = acc + to_fixed(Wb[0], Wb[1], Wb[2], m_out
                                     ).reshape(1, -1)[:, c0:c1]
            so, eo, zo = from_fixed(acc.reshape(-1), m_out)
            out_s[n0:n1, c0:c1] = so.reshape(n1 - n0, cc)
            out_e[n0:n1, c0:c1] = eo.reshape(n1 - n0, cc)
            out_z[n0:n1, c0:c1] = zo.reshape(n1 - n0, cc)
    return (out_s.reshape(Ho, Wo, Cout), out_e.reshape(Ho, Wo, Cout),
            out_z.reshape(Ho, Wo, Cout))
