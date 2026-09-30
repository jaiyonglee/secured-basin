# run_attack_2nd_server.py
#
# Purpose:
#   Launch one attack_random_server job per (sample, target, K_input) combination.
#   Jobs are independent across combinations and run fully in parallel on SLURM.
#   Within each job, the adaptive boundary search runs sequentially (phase 1 + 2).
#
# Usage:
#   python3 run_attack_2nd_server.py
#
# Notes:
#   - One sbatch job per (N, sample, target, K_input) combination.
#   - No chunking — REPEAT is handled entirely within a single C process.
#   - Phase 1: uniform (theta, w) sampling until E_safe_max and E_danger_min found.
#   - Phase 2: uniform sampling of E_temp in [mu - half, mu + half]
#              where mu = (E_safe_max + E_danger_min) / 2,
#                    half = |E_danger_min - E_safe_max|.
#   - Before running, make sure get_init has succeeded for all combinations.

import os
import time
import itertools
import random
import shutil
from subprocess import run, PIPE

_RANDOM_BASE_SEED = random.randint(0, (1 << 31) - 1)

# --------------------------------------------------
# User settings
# --------------------------------------------------

EXECUTABLE = "../../01_code/build/attack_random_server"

N          = 161
T          = 500000
REPEAT      = 1000
EARLY_EXIT  = False # True=stop immediately on crossing (faster), False=run the full simulation
INIT_ONLY   = True # True=uniform sampling, False=concentrated boundary search

# This setting only applies when INIT_ONLY=True (uniform sampling).
#   True  -> Disable the UEP restriction, sample the full range (-PI to PI), and save to result_raw2
#   False -> Enable the UEP restriction (exclude the far side) and save to result_raw1
# When INIT_ONLY=False (concentrated boundary search), the UEP restriction is always enabled
# regardless of this setting, and results are saved to result_raw3 as before.
UEP_PERTURB = False

K_INPUT_SET = [100]
SAMPLE_SET  = [0]
TARGET_SET  = list(range(N))

# --------------------------------------------------
# SLURM / resource settings
# --------------------------------------------------

MAX_WORKERS = 460
BASE_SEED   = 42
JOB_NAME    = "atk_rnd"
SBATCH_TEMPLATE = "../../01_code/build/run_atk.sh"

# --------------------------------------------------
# Output / logging directories
# --------------------------------------------------

LOG_DIR = "logs_attack_random"

if INIT_ONLY:
    RESULT_DIR = "result_raw2" if UEP_PERTURB else "result_raw1"
else:
    RESULT_DIR = "result_raw3"

# apply_uep passed to the C executable: always 1 (enabled) when INIT_ONLY=False;
# when INIT_ONLY=True, invert UEP_PERTURB (True=full range->0, False=UEP restriction->1)
APPLY_UEP = True if not INIT_ONLY else (not UEP_PERTURB)

# --------------------------------------------------
# Helper: SLURM queries
# --------------------------------------------------

def slurm_total_jobs(username: str = "jaiyong") -> int:
    result = run(
        f"squeue -u {username} | grep '{username}' | wc -l",
        shell=True, stdout=PIPE
    )
    return int(result.stdout.strip())


def slurm_named_jobs(job_name: str = JOB_NAME, username: str = "jaiyong") -> int:
    result = run(
        f"squeue -u {username} | grep '{job_name}' | wc -l",
        shell=True, stdout=PIPE
    )
    return int(result.stdout.strip())


def wait_for_slot(username: str = "jaiyong", poll_interval: int = 5) -> None:
    while slurm_total_jobs(username) >= MAX_WORKERS:
        time.sleep(poll_interval)


def wait_for_all_named(job_name: str = JOB_NAME, username: str = "jaiyong",
                       poll_interval: int = 10) -> None:
    initial = slurm_named_jobs(job_name, username)
    if initial == 0:
        return
    print(f"\nMonitoring {initial} submitted jobs  (polling every {poll_interval}s)...")
    while True:
        remaining = slurm_named_jobs(job_name, username)
        completed = initial - remaining
        pct       = 100.0 * completed / initial
        print(
            f"  Progress: {completed}/{initial} done  ({pct:.1f}%)"
            f"  -- {remaining} still running/queued",
            flush=True,
        )
        if remaining == 0:
            break
        time.sleep(poll_interval)

# --------------------------------------------------
# Helper: misc
# --------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def kr_from_k_input(k_input: int) -> float:
    return 0.01 * float(k_input)


def expected_init_path(N_val: int, sam_num: int, k_input: int) -> str:
    kr = kr_from_k_input(k_input)
    return f"init/{N_val}_sample{sam_num}_init_Kr_{kr:.2f}.txt"


def make_seed(N_val: int, sam_num: int, alpha: float, k_input: int,
              T_val: int, target: int) -> int:
    base      = BASE_SEED if BASE_SEED is not None else _RANDOM_BASE_SEED
    alpha_int = int(round(alpha * 1_000_000))
    seed = (
        base
        + 1_000_000_007 * (N_val + 1)
        + 1_000_003     * (sam_num + 1)
        + 10_007        * (target + 1)
        + 1_009         * (k_input + 1)
        + 101           * (T_val + 1)
        + alpha_int
    )
    return seed & ((1 << 63) - 1)

# --------------------------------------------------
# Preflight checks
# --------------------------------------------------

def check_executable() -> None:
    if not os.path.isfile(EXECUTABLE):
        raise FileNotFoundError(
            f"Executable not found: {EXECUTABLE}\n"
            "Compile attack_random_server first:\n"
            "  gcc -O2 -Wall -Wextra -o build/attack_random_server "
            "attack_random_server.c mt19937-64.c -lm"
        )
    if not os.access(EXECUTABLE, os.X_OK):
        raise PermissionError(f"Not executable: {EXECUTABLE}")


def check_sbatch_template() -> None:
    if not os.path.isfile(SBATCH_TEMPLATE):
        raise FileNotFoundError(
            f"sbatch template not found: {SBATCH_TEMPLATE}"
        )


def check_init_files(combos) -> list:
    ready, missing = [], []
    for n, s, k in sorted({(n, s, k) for n, s, k in combos}):
        p = expected_init_path(n, s, k)
        if os.path.isfile(p):
            ready.append((n, s, k))
        else:
            missing.append((n, s, k, p))
    if missing:
        print("WARNING: missing init files (these combos will be skipped):")
        for n, s, k, p in missing:
            print(f"  N={n}, sample={s}, K_input={k} -> {p}")
        print()
    return ready

# --------------------------------------------------
# Job submission  (one job per combination)
# --------------------------------------------------

def submit_job(N_val: int, sam_num: int, k_input: int,
               T_val: int, target: int, abs_executable: str) -> None:
    seed = make_seed(N_val, sam_num, 0.0, k_input, T_val, target)
    kr   = kr_from_k_input(k_input)

    log_name = f"N{N_val}_sam{sam_num}_tgt{target}_K{k_input}_seed{seed}"
    log_out  = os.path.abspath(os.path.join(LOG_DIR, log_name + ".out"))
    log_err  = os.path.abspath(os.path.join(LOG_DIR, log_name + ".err"))

    c_cmd = (
        f"{abs_executable} {N_val} {sam_num} {k_input} {T_val} "
        f"{target} {REPEAT} {1 if EARLY_EXIT else 0} {seed} {1 if INIT_ONLY else 0} "
        f"{1 if APPLY_UEP else 0}"
    )

    tmp_sh = os.path.abspath(
        f"atk_tmp_{N_val}_{sam_num}_{target}_{k_input}.sh"
    )
    run(f"cp {SBATCH_TEMPLATE} {tmp_sh}", shell=True)
    with open(tmp_sh, "a") as f:
        f.write(f"\n#SBATCH --job-name={JOB_NAME}\n")
        f.write(f"#SBATCH --output={log_out}\n")
        f.write(f"#SBATCH --error={log_err}\n")
        f.write(c_cmd + "\n")

    run(f"sbatch {tmp_sh}", shell=True)
    run(f"rm -f {tmp_sh}", shell=True)

# --------------------------------------------------
# Print settings
# --------------------------------------------------

def print_settings(jobs) -> None:
    print("=" * 52)
    print("run_attack_2nd_server.py  (SLURM-aware, no chunking)")
    print("=" * 52)
    print(f"Executable      : {EXECUTABLE}")
    print(f"N               : {N}")
    print(f"T               : {T}")
    print(f"REPEAT          : {REPEAT}")
    print(f"EARLY_EXIT      : {EARLY_EXIT}  ({'stop on crossing' if EARLY_EXIT else 'full simulation'})")
    print(f"INIT_ONLY       : {INIT_ONLY}  ({'phase1 only' if INIT_ONLY else 'phase1 + phase2'})")
    if INIT_ONLY:
        print(f"UEP_PERTURB     : {UEP_PERTURB}  "
             f"({'full-range perturbations (-PI to PI)' if UEP_PERTURB else 'UEP exclusion interval applied'})")
    print(f"APPLY_UEP (C argument): {APPLY_UEP}")
    print(f"K_INPUT_SET     : {K_INPUT_SET}")
    print(f"SAMPLE_SET      : {SAMPLE_SET}")
    print(f"TARGET_SET      : {TARGET_SET}")
    print(f"MAX_WORKERS     : {MAX_WORKERS}")
    base_display = BASE_SEED if BASE_SEED is not None else f"random ({_RANDOM_BASE_SEED})"
    print(f"BASE_SEED       : {base_display}")
    print(f"JOB_NAME        : {JOB_NAME}")
    print(f"SBATCH_TEMPLATE : {SBATCH_TEMPLATE}")
    print(f"LOG_DIR         : {LOG_DIR}")
    print(f"RESULT_DIR      : {RESULT_DIR}")
    print(f"Total jobs      : {len(jobs)}")
    print("=" * 52)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ensure_dir(LOG_DIR)
    ensure_dir(RESULT_DIR)

    abs_executable = os.path.abspath(EXECUTABLE)

    combos = list(itertools.product([N], SAMPLE_SET, K_INPUT_SET))
    check_executable()
    check_sbatch_template()
    ready_combos = check_init_files(combos)

    if not ready_combos:
        print("No valid init files found. Run run_get_init.py first.")
        return

    ready_set = set(ready_combos)
    jobs = [
        (n, s, k, T, tgt)
        for n, s, k in ready_combos
        for tgt in TARGET_SET
    ]

    skipped = [(n, s, k) for n, s, k in combos if (n, s, k) not in ready_set]
    print(f"Ready combos    : {len(ready_combos)} / {len(combos)}")
    if skipped:
        print(f"Skipped combos  : {skipped}")

    print_settings(jobs)

    # --------------------------------------------------
    # Submit phase: one job per combination
    # --------------------------------------------------
    submitted = 0
    for n_val, sam_num, k_input, t_val, target in jobs:
        wait_for_slot()
        submit_job(n_val, sam_num, k_input, t_val, target, abs_executable)
        submitted += 1
        print(
            f"[{submitted}/{len(jobs)}] submitted  "
            f"sample={sam_num}  target={target}  K_input={k_input}"
        )

    print(f"\nAll {submitted} jobs submitted. Waiting for completion...")
    wait_for_all_named(JOB_NAME)
    print("All jobs finished.")
    print(f"\nResults in: {RESULT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()