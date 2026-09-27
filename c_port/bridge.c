/* Bridge kernels — bit-exact mirrors of student_emu.to_fixed /
 * _from_fixed_inner. FP-free throughout (see `make trap`).
 */
#include <stdlib.h>
#include "bridge.h"
#include "luts.h"

audit_t AUDIT = {0, 0, 0};

void to_fixed(const trip_t *t, fq_t *out, int m) {
    int64_t base = LUT_FRAC[0];
    for (int i = 0; i < out->n; i++) {
        if (t[i].z) { out->q[i] = 0; continue; }
        int64_t d = (int64_t)m - (int64_t)t[i].e;
        if (d < 0) out->q[i] = (int64_t)t[i].s * base;
        else if (d > FRAC_CAP) out->q[i] = 0;
        else out->q[i] = (int64_t)t[i].s * LUT_FRAC[d];
    }
    out->m = m;
}

void triples_from_fixed(const fq_t *f, trip_t *out) {
    for (int i = 0; i < f->n; i++) {
        int64_t q = f->q[i];
        int8_t s = q > 0 ? 1 : -1;
        uint64_t a = q >= 0 ? (uint64_t)q : (uint64_t)(-(q + 1)) + 1u;
        int bl = bit_length_int((int64_t)a);
        int shift = bl - 15;
        uint64_t mant = shift >= 0 ? (a >> (shift > 0 ? shift : 0))
                                   : (a << (-shift));
        int t = shift + 14 - FIXED_F;
        int idx = t + 64;
        int ic = idx < 0 ? 0 : (idx > 192 ? 192 : idx);
        int64_t fi = (int64_t)mant - 16384;
        if (fi < 0) fi = 0;
        if (fi > 16383) fi = 16383;
        int64_t e = (int64_t)f->m + LUT_COARSE[ic] + LUT_FINE[fi];
        if (e < 0) e = 0;
        if (e > 65535) e = 65535;
        out[i].s = s;
        out[i].e = (int32_t)e;
        out[i].z = (q == 0);
    }
}

void fq_rescale(const fq_t *f, int m2, fq_t *out) {
    /* exact-value rescale via triples roundtrip ( documents intent;
     * fused kernels will do this in one pass — same values). */
    trip_t *tmp = (trip_t *)malloc(sizeof(trip_t) * f->n);
    triples_from_fixed(f, tmp);
    out->n = f->n;
    to_fixed(tmp, out, m2);
    free(tmp);
}

void fq_add(const fq_t *a, const fq_t *b, fq_t *out) {
    assert(a->m == b->m && "unit confusion: rescale explicitly");
    assert(out->n == a->n && out->n == b->n);
    for (int i = 0; i < a->n; i++) out->q[i] = a->q[i] + b->q[i];
    out->m = a->m;
}

void fq_sub(const fq_t *a, const fq_t *b, fq_t *out) {
    assert(a->m == b->m && "unit confusion: rescale explicitly");
    assert(out->n == a->n && out->n == b->n);
    for (int i = 0; i < a->n; i++) out->q[i] = a->q[i] - b->q[i];
    out->m = a->m;
}
