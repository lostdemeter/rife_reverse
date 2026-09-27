/* Bridge kernels: triples <-> tagged fixed (M2.1). Bit-exact vs numpy. */
#pragma once
#include "phi_types.h"

/* triples -> tagged fixed at scale m (mirrors student_emu.to_fixed) */
void to_fixed(const trip_t *t, fq_t *out, int m);
/* tagged fixed -> triples (reads tag; mirrors _from_fixed_inner) */
void triples_from_fixed(const fq_t *f, trip_t *out);
/* explicit rescale (the ONLY scale changer; mirrors to_fixed@new-m of values) */
void fq_rescale(const fq_t *f, int m2, fq_t *out);
/* same-scale add/sub (assert tags match — unit confusion fails loud) */
void fq_add(const fq_t *a, const fq_t *b, fq_t *out);
void fq_sub(const fq_t *a, const fq_t *b, fq_t *out);
