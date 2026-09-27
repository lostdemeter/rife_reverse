/* Geometric kernels — bit-exact mirrors of int_emu. FP-free. */
#include <stdlib.h>
#include "geo.h"
#include "bridge.h"
#include "luts.h"

#define ONE14 (1 << 14)

void bilinear_sample(const int64_t *F, int H, int W, int C, const int64_t *X,
                     const int64_t *Y, int Ho, int Wo, int64_t *out) {
    int64_t xmax = (int64_t)(W - 1) << 14, ymax = (int64_t)(H - 1) << 14;
    for (int oy = 0; oy < Ho; oy++) {
        for (int ox = 0; ox < Wo; ox++) {
            int64_t xc = X[oy * Wo + ox], yc = Y[oy * Wo + ox];
            if (xc < 0) xc = 0;
            if (xc > xmax) xc = xmax;
            if (yc < 0) yc = 0;
            if (yc > ymax) yc = ymax;
            int64_t xi = xc >> 14, xf = xc & (ONE14 - 1);
            int64_t yi = yc >> 14, yf = yc & (ONE14 - 1);
            int64_t xi1 = xi + 1 < W - 1 ? xi + 1 : W - 1;
            int64_t yi1 = yi + 1 < H - 1 ? yi + 1 : H - 1;
            int64_t wx0 = ONE14 - xf, wy0 = ONE14 - yf;
            for (int c = 0; c < C; c++) {
                int64_t tot =
                    F[(yi * W + xi) * C + c] * wx0 * wy0
                    + F[(yi * W + xi1) * C + c] * xf * wy0
                    + F[(yi1 * W + xi) * C + c] * wx0 * yf
                    + F[(yi1 * W + xi1) * C + c] * xf * yf;
                out[(oy * Wo + ox) * C + c] = tot >> 28;
            }
        }
    }
}

void warp_fixed(const fq_t *feat, int H, int W, int C, const int64_t *flow14,
                fq_t *out) {
    int N = H * W;
    int64_t *X = (int64_t *)malloc(sizeof(int64_t) * N);
    int64_t *Y = (int64_t *)malloc(sizeof(int64_t) * N);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            X[y * W + x] = ((int64_t)x << 14) + flow14[(y * W + x) * 2];
            Y[y * W + x] = ((int64_t)y << 14) + flow14[(y * W + x) * 2 + 1];
        }
    }
    bilinear_sample(feat->q, H, W, C, X, Y, H, W, out->q);
    out->n = N * C;
    out->m = feat->m;
    free(X);
    free(Y);
}

static int64_t coord_dyadic(int o, int k) {
    /* X[o] for scale 2^k, align_corners=False, 2^-14 units. Exact shifts:
     * down (k<0, j=-k): (2o+1)*2^(13+j) - 2^13; up: ((2o+1)-2^k)*2^(13-k).
     * RIFE uses k in [-3,+3] (0.125..8). */
    assert(k >= -13 && k <= 13);
    if (k < 0) {
        int j = -k;
        return ((int64_t)(2 * o + 1) << (13 + j)) - 8192;
    }
    return (((int64_t)(2 * o + 1) - (1 << k)) << (13 - k));
}

void interp_dyadic(const fq_t *f, int H, int W, int C, int k, fq_t *out,
                   int *Ho, int *Wo) {
    int ho, wo;
    if (k >= 0) {
        ho = H << k;
        wo = W << k;
    } else {
        /* RIFE dims are multiples of 32 and |k|<=3: exact division.
         * (Python round() differs on odd sizes — precondition, not bug.) */
        int j = -k;
        assert(((H >> j) << j) == H && ((W >> j) << j) == W);
        ho = H >> j;
        wo = W >> j;
        if (ho < 1) ho = 1;
        if (wo < 1) wo = 1;
    }
    *Ho = ho;
    *Wo = wo;
    int64_t *X = (int64_t *)malloc(sizeof(int64_t) * ho * wo);
    int64_t *Y = (int64_t *)malloc(sizeof(int64_t) * ho * wo);
    for (int ox = 0; ox < wo; ox++) {
        int64_t cx = coord_dyadic(ox, k);
        for (int oy = 0; oy < ho; oy++) X[oy * wo + ox] = cx;
    }
    for (int oy = 0; oy < ho; oy++) {
        int64_t cy = coord_dyadic(oy, k);
        for (int ox = 0; ox < wo; ox++) Y[oy * wo + ox] = cy;
    }
    bilinear_sample(f->q, H, W, C, X, Y, ho, wo, out->q);
    out->n = ho * wo * C;
    out->m = f->m;
    free(X);
    free(Y);
}

void sigmoid_int(const trip_t *t, int n, trip_t *out) {
    int64_t *x14 = (int64_t *)malloc(sizeof(int64_t) * n);
    for (int i = 0; i < n; i++) {
        if (t[i].z) {
            x14[i] = 0;
            continue;
        }
        int e = t[i].e < 0 ? 0 : (t[i].e > 65535 ? 65535 : t[i].e);
        x14[i] = (int64_t)t[i].s * LUT_SIGX[e];
    }
    AUDIT.sig_n += n;
    int64_t SPAN = (int64_t)16 * 16384;
    int64_t *y14 = (int64_t *)malloc(sizeof(int64_t) * n);
    for (int i = 0; i < n; i++) {
        /* audit counts BOTH span exits (mirrors sig_span); asymptotes exact */
        if (x14[i] < -SPAN || x14[i] > SPAN) AUDIT.sig_span++;
        if (x14[i] < -SPAN) y14[i] = 0;
        else if (x14[i] > SPAN) y14[i] = 16384;
        else y14[i] = LUT_SIG[x14[i] + SPAN];
    }
    /* y = y14 * 16 counts @2^-18 @U1 -> triples (mirrors from_fixed) */
    int64_t *q = (int64_t *)malloc(sizeof(int64_t) * n);
    for (int i = 0; i < n; i++) q[i] = y14[i] * 16;
    fq_t f = {q, n, PHI_BIAS};
    triples_from_fixed(&f, out);
    free(x14);
    free(y14);
    free(q);
}
