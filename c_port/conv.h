/* Blocked phi-conv (M2.2). Bit-exact mirror of student_emu.phi_conv.
 *
 * Same math: pad input with zero-triples, per (pixel, tap, co) product
 * triples (sign-mul, exp-add clipped, zero-or), to_fixed at m_out,
 * int64 accumulate, triples_from_fixed per co-column. Integer sums are
 * order-free, so blocking/threading never changes values (parity-gated).
 */
#pragma once
#include "phi_types.h"

/* W: OHWI triples (Cout,kh,kh,Cin), packed co-major. NULL bias = none. */
typedef struct {
    const int8_t *s;
    const int32_t *e;
    const uint8_t *z;
    int Cout, kh, Cin;
} cW_t;

void phi_conv(const trip_t *in, int H, int Wd, int Cin, const cW_t *cw,
              const trip_t *bias_or_null, int m_out, int stride, int pad,
              trip_t *out, int *Ho, int *Wo);
