/* phi_conv — straightforward blocked form (M2.2 correctness vehicle).
 * OpenMP over co-blocks comes after parity is green (order-free sums).
 */
#include <stdlib.h>
#include "conv.h"
#include "bridge.h"
#include "luts.h"

void phi_conv(const trip_t *in, int H, int Wd, int Cin, const cW_t *cw,
              const trip_t *bias_or_null, int m_out, int stride, int pad,
              trip_t *out, int *Ho, int *Wo) {
    int kh = cw->kh, Cout = cw->Cout;
    int ho = (H + 2 * pad - kh) / stride + 1;
    int wo = (Wd + 2 * pad - kh) / stride + 1;
    *Ho = ho;
    *Wo = wo;
    int Hp = H + 2 * pad, Wp = Wd + 2 * pad;
    int64_t base = LUT_FRAC[0];
    /* padded input (zero triples: s=0,e=0,z=0 — contributes via d>m->0,
     * exactly like numpy np.pad constant mode) */
    trip_t *P = (trip_t *)calloc((size_t)Hp * Wp * Cin, sizeof(trip_t));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < Wd; x++)
            for (int c = 0; c < Cin; c++)
                P[((y + pad) * Wp + (x + pad)) * Cin + c] =
                    in[(y * Wd + x) * Cin + c];
    int N = ho * wo;
    int64_t *acc = (int64_t *)malloc(sizeof(int64_t) * N);
    for (int co = 0; co < Cout; co++) {
        for (int i = 0; i < N; i++) acc[i] = 0;
        if (bias_or_null) {
            /* bias in fixed domain at m_out, broadcast (mirrors phi_conv) */
            trip_t b = bias_or_null[co];
            int64_t q;
            if (b.z) q = 0;
            else {
                int64_t d = (int64_t)m_out - (int64_t)b.e;
                q = d < 0 ? (int64_t)b.s * base
                    : (d > FRAC_CAP ? 0 : (int64_t)b.s * LUT_FRAC[d]);
            }
            for (int i = 0; i < N; i++) acc[i] = q;
        }
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kh; kx++) {
                for (int ci = 0; ci < Cin; ci++) {
                    int8_t ws = cw->s[((co * kh + ky) * kh + kx) * Cin + ci];
                    int32_t we = cw->e[((co * kh + ky) * kh + kx) * Cin + ci];
                    uint8_t wz = cw->z[((co * kh + ky) * kh + kx) * Cin + ci];
                    for (int oy = 0; oy < ho; oy++) {
                        int iy = oy * stride + ky;
                        for (int ox = 0; ox < wo; ox++) {
                            int ix = ox * stride + kx;
                            const trip_t *a =
                                &P[(iy * Wp + ix) * Cin + ci];
                            int8_t ps = (int8_t)((int)a->s * (int)ws);
                            int64_t pe = (int64_t)a->e + (int64_t)we - PHI_BIAS;
                            if (pe < 0) pe = 0;
                            if (pe > 65535) pe = 65535;
                            uint8_t pz = a->z | wz;
                            int64_t q;
                            if (pz) q = 0;
                            else {
                                int64_t d = (int64_t)m_out - pe;
                                q = d < 0 ? (int64_t)ps * base
                                    : (d > FRAC_CAP
                                           ? 0
                                           : (int64_t)ps * LUT_FRAC[d]);
                            }
                            acc[oy * wo + ox] += q;
                        }
                    }
                }
            }
        }
        fq_t f = {acc, N, m_out};
        trip_t *ot = (trip_t *)malloc(sizeof(trip_t) * N);
        triples_from_fixed(&f, ot);
        for (int oy = 0; oy < ho; oy++)
            for (int ox = 0; ox < wo; ox++) {
                trip_t *o = &out[(oy * wo + ox) * Cout + co];
                *o = ot[oy * wo + ox];
            }
        free(ot);
    }
    free(P);
    free(acc);
}
