/* Scale-tagged fixed-point types (Phase 2 core design — see PHASE2_SPEC).
 *
 * The load-bearing rule: a fixed-point buffer ALWAYS knows its scale.
 * triples_from_fixed() reads the tag; there is no (q, m) two-argument
 * form, so the Phase-1 decode-at-wrong-scale bug class cannot be written.
 * The only scale-changing op is fq_rescale(), explicit and greppable.
 */
#pragma once
#include <assert.h>
#include <stdint.h>

/* lattice constants (must match student_emu: K=512, BIAS=32768) */
#define PHI_K 512
#define PHI_BIAS 32768
#define PHI_NLVL 65536
#define FIXED_F 18
#define FRAC_CAP 13312

/* triples: untagged lattice values (scales live in the M table) */
typedef struct { int8_t s; int32_t e; uint8_t z; } trip_t;

/* fixed 2^-18 buffer, SCALE-TAGGED */
typedef struct { int64_t *q; int n; int m; } fq_t;

/* audits (non-zero fails tests; mirrors int_emu.AUDIT / WARP_AUDIT) */
typedef struct { long sig_span; long sig_n; long flow_over; } audit_t;
extern audit_t AUDIT;

/* truncating division (C / is NOT trunc for negatives — never use /) */
static inline int64_t tdiv(int64_t a, int64_t b) {
    int64_t q = (a >= 0 ? a : -a) / (b >= 0 ? b : -b);
    return ((a < 0) ^ (b < 0)) ? -q : q;
}

/* integer bit_length (0 for a==0); exact over full int64, no FPU */
static inline int bit_length_int(int64_t a) {
    uint64_t x = a >= 0 ? (uint64_t)a : (uint64_t)(-(a + 1)) + 1u;
    int bl = 0;
    if (x >> 32) { bl += 32; x >>= 32; }
    if (x >> 16) { bl += 16; x >>= 16; }
    if (x >> 8) { bl += 8; x >>= 8; }
    if (x >> 4) { bl += 4; x >>= 4; }
    if (x >> 2) { bl += 2; x >>= 2; }
    if (x >> 1) { bl += 1; }
    return bl + (a != 0);
}
