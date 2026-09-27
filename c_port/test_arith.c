/* M2.3 harness driver: tmul/prelu/bin2/clip/pool/neg. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arith.h"
#include "bridge.h"

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(2); }
static trip_t *rdtrip(FILE *f, int n) {
    int8_t *s = malloc(n ? n : 1);
    int32_t *e = malloc(n ? 4u * n : 1);
    uint8_t *z = malloc(n ? n : 1);
    if (n && (fread(s, 1, n, f) != (size_t)n || fread(e, 4, n, f) != (size_t)n ||
              fread(z, 1, n, f) != (size_t)n))
        die("short read");
    trip_t *t = malloc(sizeof(trip_t) * (n ? n : 1));
    for (int i = 0; i < n; i++) { t[i].s = s[i]; t[i].e = e[i]; t[i].z = z[i]; }
    free(s);
    free(e);
    free(z);
    return t;
}
static void wrtrip(FILE *f, const trip_t *t, int n) {
    for (int i = 0; i < n; i++) fwrite(&t[i].s, 1, 1, f);
    for (int i = 0; i < n; i++) {
        int32_t e = t[i].e;
        fwrite(&e, 4, 1, f);
    }
    for (int i = 0; i < n; i++) fwrite(&t[i].z, 1, 1, f);
}

int main(int argc, char **argv) {
    if (argc != 4) die("usage: test_arith <op> in out");
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) die("open in");
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) die("open out");
    int32_t h[6];
    if (fread(h, 4, 6, fi) != 6) die("header");
    int n = h[0], m = h[1];
    trip_t *a = rdtrip(fi, n);
    trip_t *out = malloc(sizeof(trip_t) * (n ? n : 1));
    if (!strcmp(argv[1], "tmul") || !strcmp(argv[1], "mul")) {
        trip_t *b = rdtrip(fi, n);
        bin2_mul(a, b, n, out);
        wrtrip(fo, out, n);
    } else if (!strcmp(argv[1], "prelu")) {
        trip_t *sl = rdtrip(fi, n);
        prelu_int(a, sl, n, out);
        wrtrip(fo, out, n);
    } else if (!strcmp(argv[1], "add") || !strcmp(argv[1], "sub") ||
               !strcmp(argv[1], "rsub")) {
        trip_t *b = rdtrip(fi, n);
        int64_t *qa = malloc(sizeof(int64_t) * (n ? n : 1));
        int64_t *qb = malloc(sizeof(int64_t) * (n ? n : 1));
        int64_t *qo = malloc(sizeof(int64_t) * (n ? n : 1));
        fq_t fa = {qa, n, 0}, fb = {qb, n, 0}, foq = {qo, n, 0};
        to_fixed(a, &fa, m);
        to_fixed(b, &fb, m);
        if (!strcmp(argv[1], "add")) bin2_add(&fa, &fb, &foq);
        if (!strcmp(argv[1], "sub")) bin2_sub(&fa, &fb, &foq);
        if (!strcmp(argv[1], "rsub")) bin2_rsub(&fa, &fb, &foq);
        triples_from_fixed(&foq, out);
        wrtrip(fo, out, n);
    } else if (!strcmp(argv[1], "clip")) {
        trip_t *lo = rdtrip(fi, 1);
        trip_t *hi = rdtrip(fi, 1);
        clip_int(a, lo[0], hi[0], n, out);
        wrtrip(fo, out, n);
    } else if (!strcmp(argv[1], "pool")) {
        int H = h[2], W = h[3], C = h[4];
        int64_t *q = malloc(sizeof(int64_t) * n);
        fq_t f = {q, n, 0};
        to_fixed(a, &f, m);
        int64_t *qo = malloc(sizeof(int64_t) * C);
        fq_t o = {qo, C, 0};
        pool_avg(&f, H, W, C, &o);
        trip_t *t = malloc(sizeof(trip_t) * C);
        triples_from_fixed(&o, t);
        wrtrip(fo, t, C);
    } else if (!strcmp(argv[1], "neg")) {
        neg_trip(a, n, out);
        wrtrip(fo, out, n);
    } else {
        die("unknown op");
    }
    fclose(fi);
    fclose(fo);
    printf("ok\n");
    return 0;
}
