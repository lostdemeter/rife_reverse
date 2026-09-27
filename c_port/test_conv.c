/* M2.2 harness driver: file-vector phi_conv. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "conv.h"

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(2); }
static void *rd(FILE *f, size_t n, size_t sz) {
    void *p = malloc(n * sz);
    if (fread(p, sz, n, f) != n) die("short read");
    return p;
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: test_conv in out");
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) die("open in");
    int32_t h[9];
    if (fread(h, 4, 9, fi) != 9) die("header");
    int H = h[0], Wd = h[1], Ci = h[2], Co = h[3], kh = h[4], st = h[5],
        pad = h[6], m = h[7], hb = h[8];
    int ni = H * Wd * Ci, nw = Co * kh * kh * Ci;
    int8_t *is = rd(fi, ni, 1);
    int32_t *ie = rd(fi, ni, 4);
    uint8_t *iz = rd(fi, ni, 1);
    int8_t *ws = rd(fi, nw, 1);
    int32_t *we = rd(fi, nw, 4);
    uint8_t *wz = rd(fi, nw, 1);
    trip_t *in = malloc(sizeof(trip_t) * ni);
    for (int i = 0; i < ni; i++) { in[i].s = is[i]; in[i].e = ie[i]; in[i].z = iz[i]; }
    cW_t cw = {ws, we, wz, Co, kh, Ci};
    trip_t *b = NULL;
    trip_t *bb = NULL;
    if (hb) {
        int8_t *bs = rd(fi, Co, 1);
        int32_t *be = rd(fi, Co, 4);
        uint8_t *bz = rd(fi, Co, 1);
        bb = malloc(sizeof(trip_t) * Co);
        for (int i = 0; i < Co; i++) { bb[i].s = bs[i]; bb[i].e = be[i]; bb[i].z = bz[i]; }
        b = bb;
    }
    fclose(fi);
    int Ho = 0, Wo = 0;
    /* size probe: run once into temp to learn Ho/Wo? phi_conv writes
     * directly; allocate worst case (same size as input map, s1) — for
     * stride-2 this over-allocates, which is safe. */
    int cap = (H + 2 * pad) * (Wd + 2 * pad) * Co + 64;
    trip_t *out = malloc(sizeof(trip_t) * cap);
    phi_conv(in, H, Wd, Ci, &cw, b, m, st, pad, out, &Ho, &Wo);
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) die("open out");
    int32_t hw[2] = {Ho, Wo};
    fwrite(hw, 4, 2, fo);
    int no = Ho * Wo * Co;
    for (int i = 0; i < no; i++) fwrite(&out[i].s, 1, 1, fo);
    for (int i = 0; i < no; i++) {
        int32_t e = out[i].e;
        fwrite(&e, 4, 1, fo);
    }
    for (int i = 0; i < no; i++) fwrite(&out[i].z, 1, 1, fo);
    fclose(fo);
    printf("ok %dx%d\n", Ho, Wo);
    return 0;
}
