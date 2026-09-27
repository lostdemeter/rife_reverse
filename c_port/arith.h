/* Elementwise integer kernels (M2.3). Bit-exact mirrors of int_emu /
 * exec_int helpers. All scale handling explicit (tags asserted).
 */
#pragma once
#include "phi_types.h"

/* exact triple multiply (sign-mul, exp-add clipped, zero-or) */
void tmul(const trip_t *a, const trip_t *b, int n, trip_t *out);
/* PReLU: x>=0 (sign bit, zero-aware) ? x : x*slope */
void prelu_int(const trip_t *t, const trip_t *slope, int n, trip_t *out);
/* triple compare: a > b in lattice order (mirrors _cmp_gt) */
int cmp_gt_trip(trip_t a, trip_t b);
/* clip to [lo,hi] scalar const triples (mirrors _clip_int) */
void clip_int(const trip_t *t, trip_t lo, trip_t hi, int n, trip_t *out);
/* binary ops on TAGGED fixed (assert same tag; mirrors _bin2 add/sub/rsub
 * at m_out; op2 = exact triple multiply, scale-free) */
void bin2_add(const fq_t *a, const fq_t *b, fq_t *out);
void bin2_sub(const fq_t *a, const fq_t *b, fq_t *out);
void bin2_rsub(const fq_t *a, const fq_t *b, fq_t *out);
void bin2_mul(const trip_t *a, const trip_t *b, int n, trip_t *out);
/* global avg-pool: tagged in -> tagged out, SAME tag (mirrors Pooling) */
void pool_avg(const fq_t *f, int H, int W, int C, fq_t *out);
/* negation (UnaryOp Neg) */
void neg_trip(const trip_t *t, int n, trip_t *out);
