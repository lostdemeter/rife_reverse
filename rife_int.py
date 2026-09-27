"""Phase-1 integer datapath for rife-v2.3 (mirrors ref_exec op-for-op).

Representation: triples (s:int8, e:int32, z:uint8) between ops; fixed-point
int64 inside MACs/warps, via student_emu bridge (to_fixed/from_fixed/tdiv,
phi_conv). Two-scale calibration (M dict): m_acc covers max single PRODUCT
per conv/IP; m_cov covers max|value| wherever triples are DECODED
(match/warp/sigmoid inputs). Sigmoid is the SOLE bounded op (U=1 absolute,
span +-16, violations AUDITED); PReLU/Leaky are exact (scale-free).
Warp/interp share bilinear_sample_fixed (nihui-exact form, proven equal
to warp_int everywhere: out-of-range taps clamp to one pixel, weights sum
to 1 in both forms).
"""
import math
import os
import numpy as np

import phi_core as S

SIG_SPAN = 16 * 16384
_SIG = None
_SIGX = None


def sigx_lut():
    """EXPACT (exact triples -> 2^-14): round(lattice(e)*2^14) for every
    e in 0..65535, lattice(e) = PHI^((e-BIAS)/K). Lets sigmoid_int accept
    ANY pre-activation range with zero encode saturation (the old
    to_fixed-at-U1 head clipped |x|>1 to +-1). Same load-or-build frozen
    pattern as sig_lut/L_FRAC."""
    global _SIGX
    if _SIGX is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts", "sigx_lut.npy")
        try:
            _SIGX = np.load(p)
        except Exception:
            _SIGX = np.round(np.power(
                S.PHI, (np.arange(65536, dtype=np.float64) - S.BIAS) / S.K
                ) * 16384).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, _SIGX)
            except Exception:
                pass
    return _SIGX


def sig_lut():
    """SIGMOID_LUT (mirrors gen_luts GELU pattern): round(sig(x)*2^14),
    x in 2^-14 over [-16,16]. Asymptotes exact (0 / 16384)."""
    global _SIG
    if _SIG is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts", "sig_lut.npy")
        try:
            _SIG = np.load(p)
        except Exception:
            k = np.arange(2 * 16 * 16384 + 1, dtype=np.float64)
            x = (k - 16 * 16384) / 16384.0
            _SIG = np.round(1.0 / (1.0 + np.exp(-x)) * 16384).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, _SIG)
            except Exception:
                pass
    return _SIG


AUDIT = {"sig_clip": 0, "sig_span": 0, "sig_n": 0}


def tshape(t):
    return t[0].shape


def tmul(a, b):
    """Exact triple multiply (sign-XOR + exp-add). Zero-aware."""
    s = (a[0].astype(np.int16) * b[0].astype(np.int16)).astype(np.int8)
    e = np.clip(a[1].astype(np.int64) + b[1].astype(np.int64) - S.BIAS,
                0, S.NLVL - 1).astype(np.int32)
    z = (a[2] | b[2]).astype(np.uint8)
    return s, e, z


def const_triple(v, shape):
    return S.encode(np.full(shape, v, np.float64))


def sigmoid_int(t):
    """Sigmoid exact-to-lattice: x14 via EXPACT gather (no saturation at
    any range) -> LUT over span +-16, asymptotes outside. sig_clip is
    structurally 0 now (nothing CAN clip); sig_span counts LUT-range
    exits (still must read 0 for fidelity)."""
    X = sigx_lut()
    e = np.clip(t[1].astype(np.int64), 0, 65535)
    x14 = np.where(t[2].astype(bool), 0, t[0].astype(np.int64) * X[e])
    AUDIT["sig_n"] += int(x14.size)
    AUDIT["sig_span"] += int((np.abs(x14) > SIG_SPAN).sum())
    G = sig_lut()
    idx = x14 + SIG_SPAN
    y14 = np.where(x14 < -SIG_SPAN, 0, np.where(
        x14 > SIG_SPAN, 16384, G[np.clip(idx, 0, len(G) - 1)]))
    return S.from_fixed(y14 * np.int64(16), S.BIAS)


def prelu_int(t, slope):
    """PReLU exact: y = x if x>=0 (sign bit) else x*slope_triple."""
    neg = (t[0].astype(np.int16) < 0) & (~t[2].astype(bool))
    ps, pe, pz = tmul(t, slope)
    s = np.where(neg, ps, t[0])
    e = np.where(neg, pe, t[1]).astype(np.int32)
    z = np.where(neg, pz, t[2]).astype(np.uint8)
    return s, e, z


def bilinear_sample_fixed(F, X, Y):
    """F int64 (H,W,C); X,Y int64 coords in 2^-14 px. Integer bilinear
    with edge replicate. Returns int64 (same fixed unit as F)."""
    H, W, C = F.shape
    ONE = 1 << 14
    Xc = np.clip(X, 0, (W - 1) << 14)
    Yc = np.clip(Y, 0, (H - 1) << 14)
    xi, xf = Xc >> 14, Xc & (ONE - 1)
    yi, yf = Yc >> 14, Yc & (ONE - 1)
    xi1 = np.minimum(xi + 1, W - 1)
    yi1 = np.minimum(yi + 1, H - 1)
    g = lambda YY_, XX_: F[YY_, XX_, np.arange(C)[None, None, :]]
    tot = (g(yi[..., None], xi[..., None]) * (ONE - xf)[..., None] * (ONE - yf)[..., None]
           + g(yi[..., None], xi1[..., None]) * xf[..., None] * (ONE - yf)[..., None]
           + g(yi1[..., None], xi[..., None]) * (ONE - xf)[..., None] * yf[..., None]
           + g(yi1[..., None], xi1[..., None]) * xf[..., None] * yf[..., None]) >> 28
    return tot


def warp_fixed(feat_q, flow_q):
    """feat int64 (H,W,C) fixed; flow int64 (H,W,2) 2^-14. Samples at p + F
    (NIHUI convention, verified from warp.cpp source — NOTE opposite sign
    from softproj.warp_int's p - F! RIFE carries both signs explicitly via
    Neg ops, so callers pass flows as-is; do NOT pre-negate here)."""
    H, W, C = feat_q.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.int64), np.arange(H, dtype=np.int64))
    X = (gx << 14)[..., None] + flow_q[..., 0:1]
    Y = (gy << 14)[..., None] + flow_q[..., 1:2]
    return bilinear_sample_fixed(feat_q, X[..., 0], Y[..., 0])


def warp_triples_int(s, e, z, flow_q, m):
    F = S.to_fixed(s, e, z, m).reshape(s.shape)
    return S.from_fixed(warp_fixed(F, flow_q).reshape(-1), m)


def interp_fixed(feat_q, sy, sx):
    """Bilinear resample int64 fixed field. Sample coords in 2^-14 px,
    EXACT for all dyadic scales 2^-k / 2^k (RIFE uses 0.125/0.25/0.5/2/4):
      down 2^-k: X[o] = (2o+1)*2^(13+k) - 2^13   (shift, exact)
      up   2^k : X[o] = ((2o+1) - 2^k)*2^(13-k)  (shift, exact; negative
                 o=0 coords clip downstream = edge replicate, same as the
                 old hardcoded 2/4 cases, torch-verified).
    Non-dyadic scales fall back to truncating division (noted approx)."""
    H, W, C = feat_q.shape
    Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)

    def coord(n_out, s):
        o = np.arange(n_out, dtype=np.int64)
        for k in range(1, 14):
            if s == 2.0 ** -k:
                return ((2 * o + 1) << (13 + k)) - 8192
            if s == 2.0 ** k:
                return (((2 * o + 1) - (1 << k)) << (13 - k))
        if s == 1.0:
            return (o << 14) + 0 * o
        # generic fallback (truncation toward zero, noted)
        num = ((2 * o + 1) << 14)
        den = int(round(2 * s * 16384))
        q = np.abs(num) // den
        return np.where(num < 0, -q, q) - 8192

    XX, YY = np.meshgrid(coord(Wo, sx), coord(Ho, sy))
    return bilinear_sample_fixed(feat_q, XX, YY)


def avgpool_int(s, e, z, m):
    F = S.to_fixed(s, e, z, m).reshape(s.shape)
    H, W, C = F.shape
    tot = F.sum(axis=(0, 1))  # (C,) int64 exact
    mean = S.tdiv(tot, H * W)  # truncating (noted 1-LSB-class approximation)
    return S.from_fixed(mean, m)


def ip_int(s, e, z, Wd, m_out, Wb=None):
    """InnerProduct as dense MAC over raveled triples (+optional bias)."""
    N = int(np.prod(s.shape))
    acc = np.zeros((Wd["s"].shape[0],), np.int64)
    if Wb is not None:
        acc += S.to_fixed(Wb[0], Wb[1], Wb[2], m_out)
    fs, fe, fz = s.reshape(-1), e.reshape(-1), z.reshape(-1)
    for j in range(Wd["s"].shape[0]):
        ps = (fs.astype(np.int16) * Wd["s"][j].astype(np.int16)).astype(np.int8)
        pe = np.clip(fe.astype(np.int64) + Wd["e"][j].astype(np.int64) - S.BIAS,
                     0, S.NLVL - 1).astype(np.int32)
        pz = (fz | Wd["z"][j]).astype(np.uint8)
        acc[j] = S.to_fixed(ps, pe, pz, m_out).sum()
    return S.from_fixed(acc, m_out)


def deconv_int(s, e, z, Wd, m_out, stride=2, pad=1, Wb=None):
    """Transpose-conv via scatter-add over bridge (reference-slow).
    out[oy,ox] += W[ky,kx]*in[iy,ix] for oy = iy*s - pad + ky
    (verified vs torch conv_transpose2d exactly)."""
    H, W, Cin = s.shape
    Cout = Wd["s"].shape[0]
    kh = Wd["s"].shape[1]
    Ho, Wo = (H - 1) * stride - 2 * pad + kh, (W - 1) * stride - 2 * pad + kh
    acc = np.zeros((Ho, Wo, Cout), np.int64)
    # NOTE 2026-09-26: bias must ADD per-co AFTER the scatter (an earlier
    # form pre-added then overwrote per-co, silently dropping deconv bias;
    # float adds it, and RIFE deconv biases reach 0.29).
    bo = (S.to_fixed(Wb[0], Wb[1], Wb[2], m_out).reshape(-1)
          if Wb is not None else None)
    for co in range(Cout):
        A = np.zeros((Ho, Wo), np.int64)
        for ky in range(kh):
            for kx in range(kh):
                ys = ky - pad + np.arange(H) * stride
                xs = kx - pad + np.arange(W) * stride
                ok = (ys >= 0) & (ys < Ho)
                okx = (xs >= 0) & (xs < Wo)
                if not ok.any() or not okx.any():
                    continue
                sub = np.zeros((ok.sum(), okx.sum()), np.int64)
                ih = np.arange(H)[ok]
                iw = np.arange(W)[okx]
                oy = ky - pad + ih * stride
                ox = kx - pad + iw * stride
                for ci in range(Cin):
                    a_s = s[np.ix_(ih, iw, [ci])].reshape(-1)
                    a_e = e[np.ix_(ih, iw, [ci])].reshape(-1)
                    a_z = z[np.ix_(ih, iw, [ci])].reshape(-1)
                    wse = int(Wd["s"][co, ky, kx, ci])
                    wee = int(Wd["e"][co, ky, kx, ci])
                    wze = int(Wd["z"][co, ky, kx, ci])
                    ps_ = (a_s.astype(np.int16) * np.int16(wse)).astype(np.int8)
                    pe_ = np.clip(a_e.astype(np.int64) + wee - S.BIAS, 0, S.NLVL - 1).astype(np.int32)
                    pz_ = (a_z | np.int8(wze)).astype(np.uint8)
                    sub += S.to_fixed(ps_, pe_, pz_, m_out).reshape(ok.sum(), okx.sum())
                A[np.ix_(oy, ox)] += sub
        acc[:, :, co] = A + (bo[co] if bo is not None else 0)
    so, eo, zo = S.from_fixed(acc.reshape(-1), m_out)
    return so.reshape(Ho, Wo, Cout), eo.reshape(Ho, Wo, Cout), zo.reshape(Ho, Wo, Cout)
