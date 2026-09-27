/* M2.3 harness driver: warp_full / interp_full / sigmoid. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "geo.h"
#include "bridge.h"

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(2); }
static void *rd(FILE *f, size_t n, size_t sz) {
    size_t t = n * sz;
    void *p = malloc(t ? t : 1);
    if (n && fread(p, sz, n, f) != n) die("short read");
    return p;
}
static trip_t *rdtrip(FILE *f, int n) {
    int8_t *s = rd(f, n, 1);
    int32_t *e = rd(f, n, 4);
    uint8_t *z = rd(f, n, 1);
    trip_t *t = malloc(sizeof(trip_t) * n);
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
    if (argc != 4) die("usage: test_geo <warp_full|interp_full|sigmoid> in out");
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) die("open in");
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) die("open out");
    if (!strcmp(argv[1], "warp_full")) {
        int32_t h[7];
        if (fread(h, 4, 7, fi) != 7) die("header");
        int H = h[0], W = h[1], C = h[2], mi = h[3], mf = h[4], div = h[5];
        int N = H * W * C, Nf = H * W * 2;
        (void)h[6];
        trip_t *img = rdtrip(fi, N);
        trip_t *flo = rdtrip(fi, Nf);
        int64_t *F = malloc(sizeof(int64_t) * N);
        int64_t *Fq = malloc(sizeof(int64_t) * Nf);
        fq_t a = {F, N, 0}, b = {Fq, Nf, 0};
        to_fixed(img, &a, mi);
        to_fixed(flo, &b, mf);
        for (int i = 0; i < Nf; i++) Fq[i] *= div;
        int64_t *O = malloc(sizeof(int64_t) * N);
        fq_t o = {O, N, 0};
        warp_fixed(&a, H, W, C, Fq, &o);
        trip_t *t = malloc(sizeof(trip_t) * N);
        triples_from_fixed(&o, t);
        int32_t hw[3] = {H, W, C};
        fwrite(hw, 4, 3, fo);
        wrtrip(fo, t, N);
    } else if (!strcmp(argv[1], "interp_full")) {
        int32_t h[5];
        if (fread(h, 4, 5, fi) != 5) die("header");
        int H = h[0], W = h[1], C = h[2], m = h[3], k = h[4];
        int N = H * W * C;
        trip_t *img = rdtrip(fi, N);
        int64_t *F = malloc(sizeof(int64_t) * N);
        fq_t a = {F, N, 0};
        to_fixed(img, &a, m);
        int Ho = 0, Wo = 0;
        int cap = (k >= 0) ? (N << (2 * k)) + 64 : N + 64;
        int64_t *O = malloc(sizeof(int64_t) * cap);
        fq_t o = {O, 0, 0};
        interp_dyadic(&a, H, W, C, k, &o, &Ho, &Wo);
        int No = Ho * Wo * C;
        trip_t *t = malloc(sizeof(trip_t) * No);
        triples_from_fixed(&o, t);
        int32_t hw[2] = {Ho, Wo};
        fwrite(hw, 4, 2, fo);
        wrtrip(fo, t, No);
    } else if (!strcmp(argv[1], "sigmoid")) {
        int32_t h[1];
        if (fread(h, 4, 1, fi) != 1) die("header");
        int n = h[0];
        trip_t *t = rdtrip(fi, n);
        trip_t *o = malloc(sizeof(trip_t) * n);
        sigmoid_int(t, n, o);
        wrtrip(fo, o, n);
    } else {
        die("unknown op");
    }
    fclose(fi);
    fclose(fo);
    printf("ok\n");
    return 0;
}
