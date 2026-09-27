CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -I.
# float gate: math.h must never appear in rife/ sources
TRAP = ! grep -rn "math\.h\|sin\|cos\|expf\?(\|logf\?(\|powf\?(\|float\|double" --include="*.c" --include="*.h" . | grep -v luts.h | grep -v test_

all: test_bridge test_conv test_geo test_arith

test_bridge: test_bridge.c bridge.c
	$(CC) $(CFLAGS) -o $@ $^

test_conv: test_conv.c conv.c bridge.c
	$(CC) $(CFLAGS) -o $@ $^

test_geo: test_geo.c geo.c bridge.c
	$(CC) $(CFLAGS) -o $@ $^

test_arith: test_arith.c arith.c bridge.c
	$(CC) $(CFLAGS) -o $@ $^

test: test_bridge test_conv test_geo test_arith
	python3 run_bridge_test.py && python3 run_conv_test.py && python3 run_geo_test.py && python3 run_arith_test.py

trap:
	$(TRAP) && echo "no float in C sources"

luts:
	python3 ../../scripts/export_luts.py

clean:
	rm -f test_bridge test_conv test_geo test_arith

.PHONY: all test trap luts clean
