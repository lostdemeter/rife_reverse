/* Elementwise kernels — bit-exact mirrors. FP-free. */
#include <stdlib.h>
#include "arith.h"
#include "bridge.h"

void tmul(const trip_t *a, const trip_t *b, int n, trip_t *out) {
    for (int i = 0; i < n; i++) {
        out[i].s = (int8_t)((int)a[i].s * (int)b[i].s);
        int64_t e = (int64_t)a[i].e + (int64_t)b[i].e - PHI_BIAS;
        if (e < 0) e = 0;
        if (e > 65535) e = 65535;
        out[i].e = (int32_t)e;
        out[i].z = a[i].z | b[i].z;
    }
}

void prelu_int(const trip_t *t, const trip_t *slope, int n, trip_t *out) {
    /* neg = (s<0) & ~z  (mirrors int_emu; slope pre-broadcast by caller) */
    trip_t *tp = (trip_t *)malloc(sizeof(trip_t) * n);
    tmul(t, slope, n, tp);
    for (int i = 0; i < n; i++) {
        int neg = ((int)t[i].s < 0) && !t[i].z;
        out[i] = neg ? tp[i] : t[i];
    }
    free(tp);
}

int cmp_gt_trip(trip_t a, trip_t b) {
    int az = a.z != 0, bz = b.z != 0;
    if (!az && !bz) {
        if (a.s == b.s) {
            /* s==0 (pad zeros, z=0) compares EQUAL regardless of e —
             * faithful to the mirror (same-sign/e rules need s!=0) */
            if (a.s > 0) return a.e > b.e;
            if (a.s < 0) return a.e < b.e;
            return 0;
        }
        return a.s > b.s;
    }
    if (!az && bz) return a.s > 0;
    return 0;
}

void clip_int(const trip_t *t, trip_t lo, trip_t hi, int n, trip_t *out) {
    for (int i = 0; i < n; i++) {
        trip_t v = cmp_gt_trip(lo, t[i]) ? lo : t[i];
        out[i] = cmp_gt_trip(v, hi) ? hi : v;
    }
}

void bin2_add(const fq_t *a, const fq_t *b, fq_t *out) {
    fq_add(a, b, out);
}

void bin2_sub(const fq_t *a, const fq_t *b, fq_t *out) {
    fq_sub(a, b, out);
}

void bin2_rsub(const fq_t *a, const fq_t *b, fq_t *out) {
    assert(a->m == b->m && "unit confusion: rescale explicitly");
    assert(out->n == a->n);
    for (int i = 0; i < a->n; i++) out->q[i] = b->q[i] - a->q[i];
    out->m = a->m;
}

void bin2_mul(const trip_t *a, const trip_t *b, int n, trip_t *out) {
    tmul(a, b, n, out);
}

void pool_avg(const fq_t *f, int H, int W, int C, fq_t *out) {
    /* tot = sum over HW (int64, order-free); tdiv(tot, H*W); same tag */
    for (int c = 0; c < C; c++) {
        int64_t tot = 0;
        for (int i = 0; i < H * W; i++) tot += f->q[i * C + c];
        out->q[c] = tdiv(tot, (int64_t)H * W);
    }
    out->n = C;
    out->m = f->m;
}

void neg_trip(const trip_t *t, int n, trip_t *out) {
    for (int i = 0; i < n; i++) {
        out[i].s = (int8_t)(-(int)t[i].s);
        out[i].e = t[i].e;
        out[i].z = t[i].z;
    }
}
