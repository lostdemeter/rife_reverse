/* M2.1 harness: file-vector tests for bridge kernels. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bridge.h"

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(2); }

int main(int argc, char **argv) {
    if (argc != 4) die("usage: test_bridge <to_fixed|from_fixed|tdiv> in out");
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) die("open in");
    int32_t n = 0, m = 0;
    if (fread(&n, 4, 1, fi) != 1 || fread(&m, 4, 1, fi) != 1 || n <= 0) die("header");
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) die("open out");
    if (!strcmp(argv[1], "to_fixed")) {
        int8_t *s = malloc(n);
        int32_t *e = malloc(4u * n);
        uint8_t *z = malloc(n);
        if (fread(s, 1, n, fi) != (size_t)n) die("s");
        if (fread(e, 4, n, fi) != (size_t)n) die("e");
        if (fread(z, 1, n, fi) != (size_t)n) die("z");
        trip_t *t = malloc(sizeof(trip_t) * n);
        for (int i = 0; i < n; i++) { t[i].s = s[i]; t[i].e = e[i]; t[i].z = z[i]; }
        int64_t *q = malloc(8u * n);
        fq_t out = {q, n, 0};
        to_fixed(t, &out, m);
        if (out.m != m) die("tag");
        fwrite(q, 8, n, fo);
    } else if (!strcmp(argv[1], "from_fixed")) {
        int64_t *q = malloc(8u * n);
        if (fread(q, 8, n, fi) != (size_t)n) die("q");
        fq_t f = {q, n, m};
        trip_t *t = malloc(sizeof(trip_t) * n);
        triples_from_fixed(&f, t);
        for (int i = 0; i < n; i++) {
            fwrite(&t[i].s, 1, 1, fo);
        }
        for (int i = 0; i < n; i++) {
            int32_t e = t[i].e;
            fwrite(&e, 4, 1, fo);
        }
        for (int i = 0; i < n; i++) {
            fwrite(&t[i].z, 1, 1, fo);
        }
    } else if (!strcmp(argv[1], "tdiv")) {
        int64_t *a = malloc(8u * n);
        int64_t *b = malloc(8u * n);
        if (fread(a, 8, n, fi) != (size_t)n) die("a");
        if (fread(b, 8, n, fi) != (size_t)n) die("b");
        for (int i = 0; i < n; i++) {
            int64_t r = tdiv(a[i], b[i]);
            fwrite(&r, 8, 1, fo);
        }
    } else {
        die("unknown op");
    }
    fclose(fi);
    fclose(fo);
    printf("ok\n");
    return 0;
}
