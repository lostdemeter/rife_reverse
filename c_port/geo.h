/* Geometric resampling kernels (M2.3): integer bilinear core shared by
 * warp and interp. Bit-exact mirrors of int_emu (nihui-exact unclamped
 * alpha, edge replicate). All coords 2^-14 px, int64 throughout.
 */
#pragma once
#include "phi_types.h"

/* F int64 (H,W,C) row-major; X,Y int64 2^-14 coords (Ho,Wo). Edge
 * replicate; out (Ho,Wo,C), same fixed unit as F. */
void bilinear_sample(const int64_t *F, int H, int W, int C, const int64_t *X,
                     const int64_t *Y, int Ho, int Wo, int64_t *out);

/* backward warp: sample = p + flow (nihui). feat (H,W,C) tagged
 * (sets out tag = feat tag); flow14 int64 2^-14 px (caller builds via
 * to_fixed@flow_m times div). out pre-allocated (H,W,C). */
void warp_fixed(const fq_t *feat, int H, int W, int C, const int64_t *flow14,
                fq_t *out);

/* dyadic resample by 2^k (k<0 downsamples; RIFE uses -3..+3).
 * Integer shift coords — exact, no division (cf. interp_fixed). */
void interp_dyadic(const fq_t *f, int H, int W, int C, int k, fq_t *out,
                   int *Ho, int *Wo);

/* sigmoid via EXPACT gather + LUT (any pre-act range; mirrors sigmoid_int:
 * triples in -> triples out, U1 output range). */
void sigmoid_int(const trip_t *t, int n, trip_t *out);
