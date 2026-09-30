# run_get_init.py
#
# Purpose:
#   Launch get_init for all (N, sample, K_input) combinations.
#   Runs sequentially and reports success/failure for each.
#
# Usage:
#   python3 run_get_init.py
#
# Notes:
#   get_init must succeed (valid synchronized equilibrium) before
#   running run_attack_2nd_server.py.

import os
import subprocess

# --------------------------------------------------
# User settings  (keep in sync with run_attack_2nd_server.py)
# --------------------------------------------------

EXECUTABLE = "../../01_code/build/get_init"

N          = 847
# ALPHA removed: mass and gamma are loaded per node from the node file
T          = 500000
K_INPUT_SET = [100] # 100 -> K = 1.00
SAMPLE_SET  = [0]  # sample0

# --------------------------------------------------
# Helper
# --------------------------------------------------

def kr_from_k_input(k_input: int) -> float:
    return 0.01 * float(k_input)


def expected_init_path(N_val: int, sam_num: int, k_input: int) -> str:
    kr = kr_from_k_input(k_input)
    return f"init/{N_val}_sample{sam_num}_init_Kr_{kr:.2f}.txt"


def check_executable() -> None:
    if not os.path.isfile(EXECUTABLE):
        raise FileNotFoundError(
            f"Executable not found: {EXECUTABLE}\n"
            "Compile get_init first:\n"
            "  gcc -O2 -Wall -Wextra -o build/get_init get_init.c mt19937-64.c -lm"
        )
    if not os.access(EXECUTABLE, os.X_OK):
        raise PermissionError(f"Not executable: {EXECUTABLE}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    check_executable()
    os.makedirs("init", exist_ok=True)

    combos = [
        (N, sam, k)
        for sam in SAMPLE_SET
        for k in K_INPUT_SET
    ]

    print("=" * 52)
    print("run_get_init.py")
    print("=" * 52)
    print(f"Executable  : {EXECUTABLE}")
    print(f"N           : {N}")
    print(f"T           : {T}")
    print(f"(mass, gamma loaded per-node from node file)")
    print(f"K_INPUT_SET : {K_INPUT_SET}")
    print(f"SAMPLE_SET  : {SAMPLE_SET}")
    print(f"Total jobs  : {len(combos)}")
    print("=" * 52)

    succeeded = []
    failed    = []

    for idx, (n_val, sam_num, k_input) in enumerate(combos, 1):
        init_path = expected_init_path(n_val, sam_num, k_input)

        # Skip if already exists
        if os.path.isfile(init_path):
            print(f"[{idx}/{len(combos)}] SKIP  N={n_val} sample={sam_num} K_input={k_input}  (already exists: {init_path})")
            succeeded.append((n_val, sam_num, k_input))
            continue

        cmd = [EXECUTABLE, str(n_val), str(sam_num), str(k_input), str(T)]
        print(f"\n[{idx}/{len(combos)}] Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, text=True)

        if result.returncode == 0 and os.path.isfile(init_path):
            print(f"  -> SUCCESS: {init_path}")
            succeeded.append((n_val, sam_num, k_input))
        else:
            print(f"  -> FAILED  (returncode={result.returncode})")
            print(f"     Try increasing T or K_input.")
            failed.append((n_val, sam_num, k_input))

    # Summary
    print("\n" + "=" * 52)
    print(f"Done.  {len(succeeded)} succeeded,  {len(failed)} failed.")
    if failed:
        print("Failed combinations (no init file generated -- normal for weak coupling):")
        for n_val, sam_num, k_input in failed:
            print(f"  N={n_val}  sample={sam_num}  K_input={k_input}")


if __name__ == "__main__":
    main()