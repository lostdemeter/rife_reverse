"""M1: geometric flow student, integer-emu forward at init weights.

Implements M1_SPEC.md exactly. Bridge/LUT formulas are verbatim ports of
dav2_reverse c_port/{phi_int.c,gen_luts.py} (see provenance header in each
function). Float mirror (float_forward) shares structure for G0 parity.
"""
import math
import os
import time
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
K, BIAS, NLVL = 512, 32768, 65536
DMAX, FRAC_CAP, FIXED_F = 4096, 13312, 18
GELU_SPAN = 8 * 16384
LUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SEED = 7

_t = {}


def _load(name, build):
    if name not in _t:
        p = os.path.join(LUTDIR, name + ".npy")
        try:
            _t[name] = np.load(p)
        except Exception:
            _t[name] = build()
            try:
                np.save(p, _t[name])
            except Exception:
                pass
    return _t[name]


def L_FRAC():
    return _load("frac_lut", lambda: np.round(
        np.power(PHI, -np.arange(13313) / K) * 2 ** 18).astype(np.int64))


def L_COARSE():
    return _load("coarse_lut", lambda: np.round(
        np.arange(-64, 129) * K * math.log(2) / LN_PHI).astype(np.int64))


def L_FINE():
    return _load("fine_lut", lambda: np.round(
        K * np.log((16384 + np.arange(16384)) / 16384) / LN_PHI).astype(np.int64))


def L_GELU():
    def build():
        k = np.arange(2 * 8 * 16384 + 1, dtype=np.float64)
        x = (k - 8 * 16384) / 16384.0
        from math import erf  # noqa
        v = 0.5 * x * (1 + np.vectorize(lambda t: math.erf(t / math.sqrt(2)))(x))
        return np.round(v * 16384).astype(np.int64)
    return _load("gelu_lut", build)


def L_EXP():
    from softproj import exp_lut
    return exp_lut()


# ---- triples ----
def encode(a):
    a = np.asarray(a, dtype=np.float64)
    s = np.sign(a).astype(np.int8)
    s[s == 0] = 1
    e = np.round(K * np.log(np.abs(a) + 1e-15) / LN_PHI).astype(np.int64) + BIAS
    return s, np.clip(e, 0, NLVL - 1).astype(np.int32), np.zeros(a.shape, np.uint8)


def decode(s, e):
    return s.astype(np.float64) * np.power(PHI, (e.astype(np.float64) - BIAS) / K)


def to_fixed(s, e, z, m):
    """Port of phi_to_fixed: (s,e,z) at scale m -> int64 2^-18 units."""
    d = np.asarray(m, dtype=np.int64) - e.astype(np.int64)
    F = L_FRAC()
    base = F[0]
    out = np.where(z.astype(bool), 0, np.where(
        d < 0, s.astype(np.int64) * base, np.where(
            d > FRAC_CAP, 0, s.astype(np.int64) * F[np.clip(d, 0, FRAC_CAP)])))
    return out.astype(np.int64)


def from_fixed(q, m):
    """Port of phi_from_fixed(q, m, f=18). q int64 2^-18 -> triples."""
    q = np.asarray(q, dtype=np.int64)
    z = (q == 0)
    a = np.abs(q)
    return _from_fixed_inner(q, z, a, m)


def bit_length_int(a):
    """Integer bit_length (floor(log2)+1, 0 for a=0) — pure integer.

    Replaces the frexp path (float64/FPU). Compare-shift cascade, exact
    over full int64 (frexp was exact only < 2^53). The <2^53 accumulator
    discipline still stands (C-port range), but the conversion itself
    no longer needs the FPU.
    """
    a = np.asarray(a, dtype=np.int64)
    bl = np.zeros(a.shape, np.int64)
    x = a
    for sh in (32, 16, 8, 4, 2, 1):
        big = (x >> sh) > 0
        bl = bl + big.astype(np.int64) * sh
        x = np.where(big, x >> sh, x)
    return bl + (a > 0).astype(np.int64)


def _from_fixed_inner(q, z, a, m):
    s = np.where(q > 0, 1, -1).astype(np.int8)
    assert int(a.max(initial=0)) < 2 ** 53, "accum exceeds frexp-exact range"
    bl = bit_length_int(a)
    nz = a > 0
    shift = bl - 15
    mant = np.where(shift >= 0, a >> np.maximum(shift, 0),
                    a << np.maximum(-shift, 0))
    t = shift + 14 - FIXED_F
    idx = t + 64
    assert int(idx[nz].min(initial=0)) >= 0 and int(idx[nz].max(initial=0)) <= 192, \
        f"COARSE index OOB (t+64 range violated): min={idx[nz].min(initial=0)} max={idx[nz].max(initial=0)}"
    C, FN = L_COARSE(), L_FINE()
    e = np.asarray(m, dtype=np.int64) + C[np.clip(idx, 0, 192)] + FN[np.clip(mant - 16384, 0, 16383)]
    e = np.clip(e, 0, 65535).astype(np.int32)
    return s, e, z.astype(np.uint8)


def tdiv(a, b):
    # True truncation toward zero, BOTH signs (C semantics). NOTE: an
    # earlier form did abs(a)//b (wrong for b<0 — floor, not trunc);
    # all historical uses had b>0 so behavior there is unchanged.
    # torch (rounding_mode="trunc") and C tdiv() already truncate.
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q)


AUDIT = {"gelu_clip_in": 0, "gelu_span_out": 0, "gelu_n": 0}

# M1d: U_GELU is ARCHITECTURAL (not calibrated). gelu_U(v) = U·gelu(v/U)
# has identical expressivity to raw gelu (weights absorb the scale) while
# the integer LUT stays faithful: bridge clips |v|>U_m, LUT spans ±8 in
# units of v/U_m. U=8 ⇒ faithful window |v|≤8 (span headroom unused but
# harmless). Training uses gelu_8 exactly (torch twin) + L_bound beyond
# 6.4. Fallback if bound fights learning: LayerNorm before gelu
# (integer-mappable via geo_int precedent) — measurable criterion:
# maxpre pinned at bound while EPE stalls.
U_GELU = 8.0
M_GELU = BIAS + int(round(K * math.log(U_GELU) / LN_PHI))


def gelu_triples(s, e, z):
    """gelu_8 (M1d discipline): fixed scale m=M_GELU, absolute semantics
    identical to torch twin's U*gelu(x/U). Pre-gelu acts must satisfy
    |.|≤8 (bridge clips beyond — COUNTED, unfaithful)."""
    m = M_GELU
    q18 = to_fixed(s, e, z, m)
    d = m - e.astype(np.int64)
    AUDIT["gelu_clip_in"] += int(((d < 0) & (~z.astype(bool))).sum())
    x14 = tdiv(q18, 16)
    AUDIT["gelu_span_out"] += int((np.abs(x14) > GELU_SPAN).sum())
    AUDIT["gelu_n"] += int(x14.size)
    G = L_GELU()
    idx = x14 + GELU_SPAN
    y14 = np.where(x14 < -GELU_SPAN, 0, np.where(x14 > GELU_SPAN, x14, G[np.clip(idx, 0, len(G) - 1)]))
    return from_fixed(y14 * np.int64(16), m)


# ---- conv ----
def im2col_idx(H, W, kh=3, stride=1, pad=1):
    Hp = H + 2 * pad
    yy, xx = np.meshgrid(np.arange(Hp), np.arange(Wp := W + 2 * pad), indexing="ij")
    oy, ox = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    Ho, Wo = oy.shape
    cols = []
    for ky in range(kh):
        for kx in range(kh):
            cols.append((oy + ky) * Wp + (ox + kx))
    return (Hp, Wp), np.stack(cols, axis=-1).reshape(-1, kh * kh), (Ho, Wo)


def _patch(Ap, Ho, Wo, stride, ky, kx):
    # NOTE 2026-09-26: columns need stride too (missing *stride broke all
    # stride-2 convs while s1 stayed exact; int+float shared the bug so M1
    # parity stayed green on wrong math — cross-validate against torch!).
    rr = (np.arange(Ho) * stride)[:, None] + np.zeros(Wo, np.int64)[None, :] + ky
    cc = (np.arange(Wo) * stride)[None, :] + np.zeros(Ho, np.int64)[:, None] + kx
    return Ap[rr, cc]


def phi_conv(s, e, z, W, m_out, stride=1, out_ch_chunk=8, Wb=None):
    """phi-conv with OPTIONAL bias triples Wb (RIFE port need; student has
    none). Bias added in fixed domain at m_out (exact, like any addend)."""
    """kh×kh phi-conv (kh=3 s1/s2 pad1, or kh=1). s,e,z: (H,W,Cin).
    W: dict(s,e,z) each (Cout,kh,kh,Cin). Returns triples (Ho,Wo,Cout).
    Bridge: products are absolute triples (XOR sign, e1+e2-BIAS);
    accumulated via to_fixed(., m_out) — m_out is the accumulator scale."""
    H, Wd, Cin = s.shape
    Cout, kh, _, CinW = W["s"].shape
    assert Cin == CinW
    pad = kh // 2 if kh == 3 else 0
    (Hp, Wp), _, _ = im2col_idx(H, Wd, kh, stride, pad)
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kh) // stride + 1
    N = Ho * Wo
    rH = (np.arange(Ho) * stride)[:, None]
    cW = (np.arange(Wo) * stride)[None, :]
    ps = np.pad(s, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    pe = np.pad(e.astype(np.int64), ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    pz = np.pad(z, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    taps = [(ky, kx, ci) for ky in range(kh) for kx in range(kh) for ci in range(Cin)]
    out_s = np.empty((N, Cout), np.int8)
    out_e = np.empty((N, Cout), np.int32)
    out_z = np.empty((N, Cout), np.uint8)
    for c0 in range(0, Cout, out_ch_chunk):
        c1 = min(Cout, c0 + out_ch_chunk)
        acc = np.zeros((N, c1 - c0), np.int64)
        if Wb is not None:
            # bias in fixed domain at m_out, broadcast over N pixels
            acc += to_fixed(Wb[0], Wb[1], Wb[2], m_out).reshape(1, -1)[:, c0:c1]
        for (ky, kx, ci) in taps:
            a_s = ps[rH + ky, cW + kx, ci].reshape(-1)
            a_e = pe[rH + ky, cW + kx, ci].reshape(-1)
            a_z = pz[rH + ky, cW + kx, ci].reshape(-1)
            for j, co in enumerate(range(c0, c1)):
                ws = int(W["s"][co, ky, kx, ci])
                we = int(W["e"][co, ky, kx, ci])
                wz = int(W["z"][co, ky, kx, ci])
                ps_ = (a_s.astype(np.int16) * ws).astype(np.int8)
                pe_ = np.clip(a_e + we - BIAS, 0, NLVL - 1).astype(np.int32)
                pz_ = (a_z | wz).astype(np.uint8)
                acc[:, j] += to_fixed(ps_, pe_, pz_, m_out)
        so, eo, zo = from_fixed(acc, m_out)
        out_s[:, c0:c1] = so.reshape(N, c1 - c0)
        out_e[:, c0:c1] = eo.reshape(N, c1 - c0)
        out_z[:, c0:c1] = zo.reshape(N, c1 - c0)
    return (out_s.reshape(Ho, Wo, Cout), out_e.reshape(Ho, Wo, Cout),
            out_z.reshape(Ho, Wo, Cout))


# NOTE on clamping: product exponents are clipped to [0,65535] before the
# bridge (C port clamps in phi_add/phi_head_predict the same way); m_out is
# the accumulator scale per M1-S2.


def warp_triples(s, e, z, flow_q, m):
    """Backward-warp triples by flow (2^-14): to_fixed -> int bilinear ->
    from_fixed at same m. flow points toward the other frame; caller passes
    half-flow so samples land at mid positions (warp_int convention)."""
    H, W, C = s.shape
    F = to_fixed(s, e, z, m).reshape(H, W, C)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.int64), np.arange(H, dtype=np.int64))
    X = np.clip((gx << 14)[..., None] - flow_q[..., 0:1], 0, (W - 1) << 14)
    Y = np.clip((gy << 14)[..., None] - flow_q[..., 1:2], 0, (H - 1) << 14)
    xi, xf = X >> 14, X & 16383
    yi, yf = Y >> 14, Y & 16383
    xi1 = np.minimum(xi + 1, W - 1)
    yi1 = np.minimum(yi + 1, H - 1)
    g = lambda YY, XX: F[YY, XX, np.arange(C)[None, None, :]]
    tot = (g(yi, xi) * (16384 - xf) * (16384 - yf)
           + g(yi, xi1) * xf * (16384 - yf)
           + g(yi1, xi) * (16384 - xf) * yf
           + g(yi1, xi1) * xf * yf) >> 28
    qs, qe, qz = from_fixed(tot.reshape(-1), m)
    return qs.reshape(H, W, C), qe.reshape(H, W, C), qz.reshape(H, W, C)


def match81(sA, eA, zA, sB, eB, zB, m, radius=4):
    """SAD cost volume over ±radius window. Fixed-point exact. -> (H,W,(2r+1)^2) int64."""
    H, W, C = sA.shape
    A = to_fixed(sA, eA, zA, m)
    B = to_fixed(sB, eB, zB, m)
    Bp = np.pad(B.reshape(H, W, C), ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    n = 2 * radius + 1
    cost = np.empty((H, W, n * n), np.int64)
    k = 0
    base = A.reshape(H, W, C)
    for dy in range(n):
        for dx in range(n):
            cost[:, :, k] = np.abs(base - Bp[dy:dy + H, dx:dx + W]).sum(axis=-1)
            k += 1
    return cost


def up2(flow):
    return np.repeat(np.repeat(flow, 2, axis=0), 2, axis=1)


def tdiv2(a):
    a = np.asarray(a, dtype=np.int64)
    q = np.abs(a) // 2
    return np.where(a < 0, -q, q)


# ---- weights (M1f: PER-ITERATION heads; shared M scales) ----
N_ITERS = 3  # M1e: unrolled refinement iters per level (M1f: unshared heads)


def _layer_defs():
    defs = [
        ("stem0", 3, 3, 16, 1), ("ref0", 16, 3, 16, 1),
        ("down0", 16, 3, 24, 2), ("ref1", 24, 3, 24, 1),
        ("down1", 24, 3, 32, 2), ("ref2", 32, 3, 32, 1),
    ]
    for lk, ch in (("h0", 81), ("h1", 81), ("h2", 289)):
        for it in range(N_ITERS):
            defs.append((f"{lk}.it{it}.a", ch, 1, 32, 1))
            defs.append((f"{lk}.it{it}.m", 32, 3, 32, 1))
            defs.append((f"{lk}.it{it}.b", 32, 1, 2, 1))
    return defs


LAYER_DEFS = _layer_defs()
RADII = {"h2": 8, "h1": 4, "h0": 4}  # L2 wide for range (M1b finding)
M_OUT = {"stem0": BIAS, "down0": BIAS + 256, "stem1": BIAS + 256,
         "down1": BIAS + 512, "stem2": BIAS + 512,
         "h0.a": BIAS + 2048, "h0.b": BIAS + 2048,
         "h1.a": BIAS + 2048, "h1.b": BIAS + 2048,
         "h2.a": BIAS + 2048, "h2.b": BIAS + 2048}
# NOTE (M1 calibration finding): the static table above is a FALLBACK only.
# At init, activation magnitudes are unknown; blind scales saturate the
# bridge (d<0 clip) or waste precision. calibrate() below measures typical
# |v| per tensor on a float init pass and sets m = BIAS + round(512·ln(typ)/lnφ)
# so typical d≈0. Flow-head m_out are FORCED equal (adds mix levels).
# Calibrated m's freeze into the artifact. Static table used only if
# calibration is skipped (parity still holds; precision may not).


def unit_of(m):
    return PHI ** ((m - BIAS) / K)


def init_weights(seed=SEED, gain=1.0):
    # M1 calibration finding #2 (Xavier needs no help) + #5 (bound):
    # stems/down keep gain 1.0; head-a/mixer use 0.15 so init pre-gelu
    # acts land ≤1 (the integer-gelu bound). Recorded init design, not
    # quality tuning (no quality bar at M1; training moves weights under
    # the L_bound penalty).
    gains = {"stem": 1.0, "down": 1.0, "a": 0.15, "m": 0.15, "b": 1.0}
    rng = np.random.default_rng(seed)
    W = {}
    for name, cin, kh, cout, _ in LAYER_DEFS:
        role = name.split(".")[-1] if "." in name else ("down" if name.startswith("down") else "stem")
        g = gains.get(role, 1.0) * gain
        bound = math.sqrt(6.0 / (cin * kh * kh + cout)) * g
        f = rng.uniform(-bound, bound, size=(cout, kh, kh, cin))
        if name.endswith(".b"):  # zero-flow start: exact-zero flag (M1-S4)
            s = np.ones_like(f, np.int8)
            e_ = np.zeros_like(f, np.int32)
            z = np.ones_like(f, np.uint8)
        else:
            s, e_, z = encode(f)
        W[name] = {"s": s, "e": e_, "z": z, "f": f}
    return W


def save_artifact(W, M, path="artifact.npz"):
    kw = {"seed": np.array([SEED]), "gain": np.array([1.0])}
    for name, d in W.items():
        kw[name + ".s"] = d["s"]
        kw[name + ".e"] = d["e"]
        kw[name + ".z"] = d["z"]
    for k, v in M.items():
        kw["m__" + k] = np.array([v], dtype=np.int64)
    np.savez(path, **kw)


def stem(s, e, z, W, M, name, stride=1):
    return phi_conv(s, e, z, {k: W[name][k] for k in "sez"}, M[name], stride)


def head(cost, W, M, prefix, mkey, m_cost, nch):
    # prefix: weight key (per-iteration, M1f); mkey: scale key (shared
    # across iterations — magnitudes similar; parity verifies).
    cs, ce, cz = from_fixed(cost.reshape(-1), m_cost)
    H, Wd, _ = cost.shape
    cs, ce, cz = cs.reshape(H, Wd, nch), ce.reshape(H, Wd, nch), cz.reshape(H, Wd, nch)
    a = phi_conv(cs, ce, cz, {k: W[prefix + ".a"][k] for k in "sez"}, M[mkey + ".a"])
    g = gelu_triples(a[0], a[1], a[2])
    m = phi_conv(g[0], g[1], g[2], {k: W[prefix + ".m"][k] for k in "sez"}, M[mkey + ".m"])
    g2 = gelu_triples(m[0], m[1], m[2])
    return phi_conv(g2[0], g2[1], g2[2], {k: W[prefix + ".b"][k] for k in "sez"}, M[mkey + ".b"])


def normalize(rgb):
    return (rgb.astype(np.float64) / 255.0 - MEAN) / STD


def to_flow14(h, m_hb):
    """Head-output triples -> int64 2^-14 flow in head-scale units.

    to_fixed gives q18 with q18·2^-18 ≈ v/U; the same quantity in 2^-14
    units is q18/16 (C truncation). NOT ×16 (M1 calibration finding)."""
    q = to_fixed(h[0], h[1], h[2], m_hb)
    return tdiv(q, 16)


def refine_level(fA, fB, flow_init, W, M, lkey, fkey, timings=None):
    """K residual updates at one level. flow_init: int64 2^-14 (upsampled
    coarser flow, or zeros at L2). Shared head weights every iteration
    (no new params; C port = loop). Returns refined int64 flow."""
    import time as _t
    r = RADII[lkey]
    nch = (2 * r + 1) ** 2
    flow = flow_init
    for it in range(N_ITERS):
        t = _t.perf_counter()
        wB = warp_triples(*fB, tdiv2(flow), M["cov." + fkey])
        c = match81(*fA, *wB, M["cov." + fkey], radius=r)
        h = head(c, W, M, f"{lkey}.it{it}", lkey, M[lkey + ".cost"], nch)
        d = to_flow14(h, M[lkey + ".b"]).reshape(c.shape[:2] + (2,))
        flow = flow + d
        if timings is not None:
            timings[lkey] = timings.get(lkey, 0.0) + (_t.perf_counter() - t)
    return flow


def forward(fA, fB, W, M, timings=None):
    """fA/fB uint8 RGB. Returns int64 flow (H,W,2) 2^-14 in head-scale
    units (all head m_out forced equal, so levels mix validly; absolute
    px calibration is M3's job). M1-S3 dataflow + M1e refinement."""
    def rec(k, v):
        if timings is not None:
            timings[k] = timings.get(k, 0.0) + v
    t = time.perf_counter()
    tA = encode(normalize(fA))
    tB = encode(normalize(fB))
    rec("encode", time.perf_counter() - t)
    feats = {}
    for tag, tt in (("A", tA), ("B", tB)):
        t = time.perf_counter()
        r0 = stem(*stem(*tt, W, M, "stem0"), W, M, "ref0")
        g0 = gelu_triples(*r0)
        d0 = stem(*g0, W, M, "down0", stride=2)
        r1 = stem(*d0, W, M, "ref1")
        g1 = gelu_triples(*r1)
        d1 = stem(*g1, W, M, "down1", stride=2)
        r2 = stem(*d1, W, M, "ref2")
        s2 = gelu_triples(*r2)
        feats[tag] = (g0, g1, s2)
        rec("stem", time.perf_counter() - t)
    # L2 (coarsest): index 2 in feats tuples; flow starts at exact zero
    c2shape = feats["A"][2][0].shape[:2]
    flows = {2: refine_level(feats["A"][2], feats["B"][2],
                            np.zeros(c2shape + (2,), np.int64),
                            W, M, "h2", "s2", timings)}
    # L1
    f1 = up2(flows[2])
    flows[1] = refine_level(feats["A"][1], feats["B"][1], f1, W, M, "h1", "s1", timings)
    # L0
    f0 = up2(flows[1])
    flows[0] = refine_level(feats["A"][0], feats["B"][0], f0, W, M, "h0", "s0", timings)
    return flows[0]


# ---- float mirror (parity target, NOT shipped) ----
# Scale discipline (mirrors the bridge exactly):
# - conv preserves real values (products exact, accum labeled by m_out but
#   from_fixed inverts the label) ⇒ float conv is RAW, no scaling.
# - gelu at scale m: int computes gelu on 2^-14 units of v/U_m and labels
#   back ⇒ float: gelu(v·2^14/U_m)·U_m·2^-14.
# - match cost int64 = 2^18·C_raw/U_feat; from_fixed(., m_cost) labels it
#   ×U_cost ⇒ float head input = C_raw·U_cost/U_feat (gain G_Lk).
# - head out triples have value V; int flow14·2^-14 = V/U_hb ⇒ float flow
#   = V/U_hb. All head m_out forced equal ⇒ levels mix validly.
def float_forward(fA, fB, W, M):
    """Same dataflow in float64 from decoded weights. Returns flow in
    head-scale units (compare int_out/16384)."""
    F = {n: decode(d["s"], d["e"]).astype(np.float64) * (1 - d["z"]) for n, d in W.items()}
    U = {k: unit_of(v) for k, v in M.items()}
    XnA = normalize(fA)
    XnB = normalize(fB)

    def conv_do(A, name, stride=1):
        Kh = F[name].shape[1]
        pad = Kh // 2 if Kh == 3 else 0
        Ap = np.pad(A, ((pad, pad), (pad, pad), (0, 0)))
        H, Wd = A.shape[:2]
        Ho, Wo = (H + 2 * pad - Kh) // stride + 1, (Wd + 2 * pad - Kh) // stride + 1
        out = np.zeros((Ho, Wo, F[name].shape[0]))
        for ky in range(Kh):
            for kx in range(Kh):
                P = _patch(Ap, Ho, Wo, stride, ky, kx)
                out += P @ F[name][:, ky, kx, :].T
        return out

    UG = 8.0  # must equal U_GELU / torch twin gelu8 (M1d discipline)

    def gelu_f(A):
        # gelu_8: exact mirror of int side (and torch twin)
        return UG * 0.5 * (A / UG) * (1 + np.vectorize(lambda t: math.erf(t / math.sqrt(2)))(A / UG))

    def warp_f(Fm, fl):
        import cv2
        H, Wd = Fm.shape[:2]
        gx, gy = np.meshgrid(np.arange(Wd, dtype=np.float64), np.arange(H, dtype=np.float64))
        outs = []
        for c in range(Fm.shape[2]):
            outs.append(cv2.remap(Fm[:, :, c], (gx - fl[..., 0]).astype(np.float32),
                                  (gy - fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
        return np.stack(outs, -1)

    def match_f(A, B, r=4):
        H, Wd, C = A.shape
        Bp = np.pad(B, ((r, r), (r, r), (0, 0)), mode="edge")
        n = 2 * r + 1
        cost = np.empty((H, Wd, n * n))
        k = 0
        for dy in range(n):
            for dx in range(n):
                cost[:, :, k] = np.abs(A - Bp[dy:dy + H, dx:dx + Wd]).sum(-1)
                k += 1
        return cost

    feats = {}
    for tag, Xn in (("A", XnA), ("B", XnB)):
        s0 = gelu_f(conv_do(conv_do(Xn, "stem0"), "ref0"))
        d0 = conv_do(s0, "down0", 2)
        s1 = gelu_f(conv_do(d0, "ref1"))
        d1 = conv_do(s1, "down1", 2)
        s2 = gelu_f(conv_do(d1, "ref2"))
        feats[tag] = (s0, s1, s2)

    def head_f(cost_raw, prefix, mkey):
        # prefix: weight key (per-iteration); mkey: scale key (shared)
        G = U[mkey + ".cost"] / U["cov." + {"h0": "s0", "h1": "s1", "h2": "s2"}[mkey]]
        a = conv_do(cost_raw * G, prefix + ".a")
        m = conv_do(gelu_f(a), prefix + ".m")
        return conv_do(gelu_f(m), prefix + ".b") / U[mkey + ".b"]

    def refine_f(fA_, fB_, flow_init, lkey, fkey):
        r = RADII[lkey]
        flow = flow_init
        for it in range(N_ITERS):
            wB = warp_f(fB_, flow / 2)
            c = match_f(fA_, wB, r=r)
            flow = flow + head_f(c, f"{lkey}.it{it}", lkey)
        return flow

    fA2, fB2 = feats["A"][2], feats["B"][2]
    fl2 = refine_f(fA2, fB2, np.zeros(fA2.shape[:2] + (2,)), "h2", "s2")
    f1 = np.repeat(np.repeat(fl2, 2, 0), 2, 1)
    fl1 = refine_f(feats["A"][1], feats["B"][1], f1, "h1", "s1")
    f0 = np.repeat(np.repeat(fl1, 2, 0), 2, 1)
    return refine_f(feats["A"][0], feats["B"][0], f0, "h0", "s0")


def calibrate(W, fA, fB, margin=512):
    """Two-scale calibration (M1 finding #3).

    Every fixed-domain crossing declares its range:
    - m_acc[conv]: covers max single PRODUCT max|a_in|·max|w| (sums never
      clip in int64; only per-product to_fixed can). Accumulator scale.
    - m_cov[tensor]: covers max|value| of a triple tensor wherever it is
      DECODED (match/warp/gelu inputs). Prevents d<0 saturation.
    - m_cost: gain-setting (head-input magnitude), formula below.
    Flow-head m_out forced equal (add-consistency). All frozen to artifact.
    """
    F = {n: decode(d["s"], d["e"]).astype(np.float64) * (1 - d["z"]) for n, d in W.items()}
    maxw = {n: float(np.abs(d["f"]).max()) for n, d in W.items()}

    def conv_raw(A, name, stride=1):
        Kh = F[name].shape[1]
        pad = Kh // 2 if Kh == 3 else 0
        Ap = np.pad(A, ((pad, pad), (pad, pad), (0, 0)))
        H, Wd = A.shape[:2]
        Ho, Wo = (H + 2 * pad - Kh) // stride + 1, (Wd + 2 * pad - Kh) // stride + 1
        out = np.zeros((Ho, Wo, F[name].shape[0]))
        for ky in range(Kh):
            for kx in range(Kh):
                P = _patch(Ap, Ho, Wo, stride, ky, kx)
                out += P @ F[name][:, ky, kx, :].T
        return out

    def m_of(pmax):
        return BIAS + int(round(K * math.log(pmax + 1e-12) / LN_PHI)) + margin

    def gelu8p(A):
        return 8.0 * 0.5 * (A / 8.0) * (1 + np.vectorize(
            lambda t: math.erf(t / math.sqrt(2)))(A / 8.0))

    Xn0, Xn1 = normalize(fA), normalize(fB)
    M = {}
    cov = {}
    cur0, cur1 = Xn0, Xn1
    for name, stride, ckey, gel in [("stem0", 1, None, False), ("ref0", 1, "s0", True),
                               ("down0", 2, "d0", False), ("ref1", 1, "s1", True),
                               ("down1", 2, "d1", False), ("ref2", 1, "s2", True)]:
        nxt0 = conv_raw(cur0, name, stride)
        nxt1 = conv_raw(cur1, name, stride)
        if gel:
            nxt0, nxt1 = gelu8p(nxt0), gelu8p(nxt1)
        amax_in = max(float(np.abs(cur0).max()), float(np.abs(cur1).max()))
        amax_out = max(float(np.abs(nxt0).max()), float(np.abs(nxt1).max()))
        M[name] = m_of(amax_in * maxw[name])  # product-covering accumulator
        if ckey:
            cov[ckey] = m_of(amax_out)        # value-covering decodes
        cur0, cur1 = nxt0, nxt1
    # proxy costs per level (float SAD on calibrated feats)
    fmap = {}
    c0, c1 = Xn0, Xn1
    fmap["s0"] = (gelu8p(conv_raw(conv_raw(c0, "stem0"), "ref0")),
                  gelu8p(conv_raw(conv_raw(c1, "stem0"), "ref0")))
    fmap["d0"] = (conv_raw(fmap["s0"][0], "down0", 2), conv_raw(fmap["s0"][1], "down0", 2))
    fmap["s1"] = (gelu8p(conv_raw(fmap["d0"][0], "ref1")), gelu8p(conv_raw(fmap["d0"][1], "ref1")))
    fmap["d1"] = (conv_raw(fmap["s1"][0], "down1", 2), conv_raw(fmap["s1"][1], "down1", 2))
    fmap["s2"] = (gelu8p(conv_raw(fmap["d1"][0], "ref2")), gelu8p(conv_raw(fmap["d1"][1], "ref2")))
    for lk, fk in (("h0", "s0"), ("h1", "s1"), ("h2", "s2")):
        A, B = fmap[fk]
        H, Wd, C = A.shape
        r = RADII[lk]
        n = 2 * r + 1
        Bp = np.pad(B, ((r, r), (r, r), (0, 0)), mode="edge")
        smax = 0.0
        for dy in range(n):
            for dx in range(n):
                smax = max(smax, float(np.abs(A - Bp[dy:dy + H, dx:dx + Wd]).sum(-1).max()))
        # gain: max head-input value ≈ φ^{margin/K}
        need = math.log(unit_of(cov[fk])) - math.log(smax + 1e-12)
        M[lk + ".cost"] = BIAS + int(round(K * need / LN_PHI)) + margin
        # head-a conv accumulator (product-covering) + pre-gelu coverage
        G = unit_of(M[lk + ".cost"]) / unit_of(cov[fk])
        cost81 = np.stack(
            [np.abs(A - np.pad(B, ((r, r), (r, r), (0, 0)), mode="edge")
                       [dy:dy + H, dx:dx + Wd]).sum(-1)
             for dy in range(n) for dx in range(n)], axis=-1)
        M[lk + ".a"] = m_of(float(np.abs(cost81).max()) * maxw[lk + ".it0.a"])
        ha = conv_raw(G * cost81, lk + ".it0.a")
        M["cov." + lk + "a"] = m_of(float(np.abs(ha).max()))
        # mixer conv accumulator + pre-gelu2 coverage. |gelu|≤1.1|x| bounds
        # mixer inputs by head-a max; exact erf unnecessary for scales.
        M[lk + ".m"] = m_of(float(np.abs(ha).max()) * 1.1 * maxw[lk + ".it0.m"])
        hm = conv_raw(np.clip(ha, 0, None), lk + ".it0.m")
        M["cov." + lk + "m"] = m_of(float(np.abs(hm).max()))
    for k, v in cov.items():
        M["cov." + k] = v
    # flow-head scales forced equal (add-consistency across levels).
    # Head-b outputs are exact-zero at init; inherit max head-a m_acc
    # (M3 recalibrates once Δflow is nonzero).
    mh = max(M["h0.a"], M["h1.a"], M["h2.a"])
    M["h0.b"] = M["h1.b"] = M["h2.b"] = mh
    return M
