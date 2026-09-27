"""Two-scale calibration for the RIFE int port (M1 discipline).

m_acc[layer]: covers max single PRODUCT max|a_in|*max|w|.
m_cov[blob]: covers max|value| wherever triples decode (match/warp/
sigmoid/pool inputs, and every bridge output for precision sanity).
Flow-carrying triples use M_FLOW convention (see exec_int header).
Margin 512 steps (~2.7x) everywhere; frozen into artifact.
"""
import math
import numpy as np

import phi_core as S


def m_of(pmax, margin=512):
    return S.BIAS + int(round(S.K * math.log(pmax + 1e-12) / S.LN_PHI)) + margin


def calibrate(layers, Wfloat, feeds):
    """Wfloat: float weight dict (from rife_ncnn). Returns M dict.
    Runs float reference instrumented for maxima."""
    import rife_ref
    maxact = {}
    orig_exec = rife_ref.execute

    def rec(layers_, weights, feeds_):
        blobs = orig_exec(layers_, weights, feeds_)
        for k, v in blobs.items():
            if isinstance(v, np.ndarray):
                m = float(np.abs(v).max())
                if k not in maxact or m > maxact[k]:
                    maxact[k] = m
        return blobs

    rec(layers, Wfloat, feeds)
    M = {}
    maxw = {}
    for L in layers:
        t, p, name = L["type"], L["p"], L["name"]
        if t in ("Convolution", "InnerProduct", "Deconvolution"):
            key = name + ".weight"
            if key in Wfloat:
                maxw[name] = float(np.abs(Wfloat[key]).max())
    for L in layers:
        for k in L["outs"]:
            if k in maxact:
                M["cov:" + k] = m_of(maxact[k])
    for L in layers:
        t, p, name = L["type"], L["p"], L["name"]
        if t in ("Convolution", "InnerProduct", "Deconvolution") and name in maxw:
            # product bound from max input act feeding this layer
            amax = max([maxact.get(k, 1e-12) for k in L["ins"]] + [1e-12])
            taps = 1
            if t in ("Convolution", "Deconvolution"):
                taps = p[1] * p[1] * 1  # per-tap product (accum never clips)
            M[name] = m_of(amax * maxw[name])
    # ensure every input blob has a scale (feeds)
    for k in feeds:
        if "cov:" + k not in M:
            M["cov:" + k] = m_of(float(np.abs(feeds[k]).max()) + 1e-12)
    # flow UNIT for warp 2^-14 conversion: smallest power-of-two U >=
    # 16 covering max |flow| over all warp flow-inputs (+25% headroom).
    # exec_int multiplies to_fixed(@U) output by div=U/16 (EXACT, since
    # 2^-18@U times U/16 = 2^-14 px). div=1 keeps the legacy U16 path.
    maxflow = 0.0
    for L in layers:
        if L["type"] == "rife.Warp" and len(L["ins"]) > 1:
            for k in L["ins"][1:]:
                maxflow = max(maxflow, maxact.get(k, 0.0))
    U, div = 16, 1
    while U < maxflow * 1.25:
        U *= 2
        div *= 2
    M["flow_U"], M["flow_div"] = U, div
    M["flow_m"] = S.BIAS + int(round(S.K * math.log(float(U)) / S.LN_PHI))
    return M
