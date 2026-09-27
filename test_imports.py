"""Import sweep: every module must import (catches migration damage).

Regression gate: a dropped colon in torch_emu.py once survived all
suites because nothing imported it. Best-effort for torch/CUDA-absent
systems on backend modules; hard fail otherwise.
Usage: python3 test_imports.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CORE = ["phi_core", "rife_ncnn", "rife_ref", "rife_int", "rife_exec",
        "rife_cal", "test_core", "fetch_rife_v23", "demo_interp"]
BACKEND = ["torch_emu", "exec_torch"]
FAIL = []


def main():
    for m in CORE:
        try:
            __import__(m)
            print(f"{m}: OK")
        except Exception as ex:
            print(f"{m}: FAIL {str(ex)[:120]}")
            FAIL.append(m)
    try:
        import torch  # noqa
        have_torch = True
    except ImportError:
        have_torch = False
    for m in BACKEND:
        try:
            __import__(m)
            print(f"{m}: OK")
        except Exception as ex:
            if have_torch:
                print(f"{m}: FAIL {str(ex)[:120]}")
                FAIL.append(m)
            else:
                print(f"{m}: SKIP (no torch)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
