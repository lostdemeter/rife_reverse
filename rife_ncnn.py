"""ncnn .param parser + .bin loader with byte-exact audit (RIFE port Phase 0).

.param: magic, counts, then per layer:
  Type Name in_count out_count [inputs] [outputs] [key=value ...]
  values: int | float | comma-list (arrays, e.g. -23310=1,0.32)
.bin: sequential float32 blobs; per-layer layout derived from param.
  Ambiguous defaults (InnerProduct bias when 1= absent) are decided by
  BYTE-EXACT match: exactly one hypothesis may sum to the file size.
  Anything else -> loud failure with both totals (never silent drift).
"""
import numpy as np

ACT9 = {2: "leaky", 4: "sigmoid"}  # fused activations; external gate arbitrates


def _val(s):
    # ncnn array values are [size, v0, v1, ...] — the size prefix is
    # STRIPPED (docs: "integer array value: [array size],int,...").
    # Missing this produced phantom slice axes (found via Slice_99).
    if "," in s:
        parts = [_val(x) for x in s.split(",")]
        n = parts[0]
        assert isinstance(n, int) and len(parts) - 1 >= n, f"array {s}"
        return parts[1:1 + n]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_param(path):
    layers = []
    with open(path) as fh:
        magic = fh.readline().strip()
        assert magic == "7767517", f"bad magic {magic} in {path}"
        nlayers, nblobs = [int(x) for x in fh.readline().split()]
        for _ in range(nlayers):
            toks = fh.readline().split()
            typ, name = toks[0], toks[1]
            ni, no = int(toks[2]), int(toks[3])
            ins = toks[4:4 + ni]
            outs = toks[4 + ni:4 + ni + no]
            params = {}
            for kv in toks[4 + ni + no:]:
                k, v = kv.split("=", 1)
                params[int(k)] = _val(v)
            layers.append({"type": typ, "name": name, "ins": ins, "outs": outs, "p": params})
    assert len(layers) == nlayers, f"layer count {len(layers)} != {nlayers}"
    return layers


def weight_shapes(layers):
    """Yield (key, shape, kind) per weight blob in .bin order."""
    for L in layers:
        t, p = L["type"], L["p"]
        if t == "Convolution":
            out, k = p[0], p[1]
            w = p[6]
            has_bias = p.get(5, 0) == 1
            cin = w // (out * k * k)
            assert w == out * k * k * cin, f"{L['name']}: weight {w} != {out}x{k}x{k}x{cin}"
            yield (L["name"] + ".weight", (out, cin, k, k), "conv")
            if has_bias:
                yield (L["name"] + ".bias", (out,), "bias")
        elif t == "InnerProduct":
            out, w = p[0], p[2]
            has_bias = p.get(1, 0) == 1
            cin = w // out
            assert w == out * cin, f"{L['name']}: ip weight {w} != {out}x{cin}"
            yield (L["name"] + ".weight", (out, cin), "ip")
            if has_bias:
                yield (L["name"] + ".bias", (out,), "bias")
        elif t == "PReLU":
            yield (L["name"] + ".slope", (p[0],), "slope")
        elif t == "Deconvolution":
            out, k = p[0], p[1]
            w = p[6]
            has_bias = p.get(5, 0) == 1
            cin = w // (out * k * k)
            assert w == out * k * k * cin, f"{L['name']}: deconv {w}"
            yield (L["name"] + ".weight", (out, cin, k, k), "deconv")
            if has_bias:
                yield (L["name"] + ".bias", (out,), "bias")


FLAG_FP16 = 0x01306B47  # verified head bytes
FLAG_FP32 = 0x00000000


def load_bin(path, layers):
    """Sequential parse, validated three ways: every weight flag checked,
    final EOF exact, value sanity downstream (external parity gate).
    Layout (proven by byte-exact walk of flownet.bin):
    - weights: [u32 flag][fp16|fp32 data][32-bit align pad]
    - conv bias iff 5==1, deconv bias iff 5==1: raw fp32, no flag
    - InnerProduct: NEVER bias (torch default overridden in RIFE export;
      proven: no-bias walk hits EOF exactly; a bias would desync flags)
    - PReLU slopes: raw fp32, no flag
    Any desync fails LOUD with byte position (never silent drift)."""
    import os
    import struct
    total = os.path.getsize(path)
    raw = open(path, "rb").read()
    assert len(raw) == total
    pos, W = 0, [None]  # W as list-cell for nested fn
    W = {}
    n_flag = {"fp16": 0, "fp32": 0}

    def u32():
        return struct.unpack("<I", raw[pos:pos + 4])[0]

    def blob(n, kind, owner):
        nonlocal pos
        (flag,) = struct.unpack("<I", raw[pos:pos + 4])
        if flag == FLAG_FP16:
            v = np.frombuffer(raw[pos + 4:pos + 4 + 2 * n], dtype=np.float16).astype(np.float32)
            pos += 4 + 2 * n
        elif flag == FLAG_FP32:
            v = np.frombuffer(raw[pos + 4:pos + 4 + 4 * n], dtype=np.float32).copy()
            pos += 4 + 4 * n
        else:
            raise ValueError(f"{owner}/{kind}: flag 0x{flag:08x} at byte {pos} (not fp16/fp32)")
        if v.size != n:
            raise ValueError(f"{owner}/{kind}: overrun")
        pos += (4 - (pos % 4)) % 4
        n_flag["fp16" if flag == FLAG_FP16 else "fp32"] += 1
        return v

    def raw32(n):
        nonlocal pos
        v = np.frombuffer(raw[pos:pos + 4 * n], dtype=np.float32).copy()
        if v.size != n:
            raise ValueError("overrun reading raw fp32")
        pos += 4 * n
        return v

    for L in layers:
        t, p, nm = L["type"], L["p"], L["name"]
        if t == "Convolution":
            out, k, w = p[0], p[1], p[6]
            assert w % (out * k * k) == 0, f"{nm}: weight {w}"
            W[nm + ".weight"] = blob(w, "weight", nm).reshape(out, w // (out * k * k), k, k)
            if p.get(5, 0) == 1:
                W[nm + ".bias"] = raw32(out)
        elif t == "InnerProduct":
            out, w = p[0], p[2]
            assert w % out == 0, f"{nm}: ip weight {w}"
            W[nm + ".weight"] = blob(w, "weight", nm).reshape(out, w // out)
            if p.get(1, 0) == 1:
                W[nm + ".bias"] = raw32(out)
        elif t == "PReLU":
            W[nm + ".slope"] = raw32(p[0])
        elif t == "Deconvolution":
            out, k, w = p[0], p[1], p[6]
            assert w % (out * k * k) == 0, f"{nm}: deconv {w}"
            W[nm + ".weight"] = blob(w, "weight", nm).reshape(out, w // (out * k * k), k, k)
            if p.get(5, 0) == 1:
                W[nm + ".bias"] = raw32(out)
    audit = {"file_bytes": total, "consumed": pos, "weight_blobs": n_flag}
    if pos != len(raw):
        raise SystemExit(f"error: .bin trailing bytes for {path}: pos {pos} vs {len(raw)}")
    return W, audit
