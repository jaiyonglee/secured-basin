# 5_run_attack_power_server.py
#
# Purpose:
#   Launch power-perturbation attack jobs (attack_power_server)
#   via SLURM. Each job applies delta_P power to the target node
#   for delta_t seconds, then runs the full simulation.
#
#   mass and gamma are loaded per-node from the node file — no alpha argument.
#
# Usage:
#   python3 5_run_attack_power_server.py

import os
import time
import itertools
from subprocess import run, PIPE

# --------------------------------------------------
# User settings
# --------------------------------------------------

EXECUTABLE = "../../01_code/build/attack_power_server"

N          = 838
T          = 500000
REPEAT     = 1000
EARLY_EXIT  = False     # True=stop immediately on crossing (faster), False=run the full simulation
P_MAX_FACTOR  = 100.0    # P_max = factor * sum(K[target])
P_MIN_FACTOR  = 1e-3     # P_min = factor * sum(K[target])
T_MAX_SEC     = 100.0   # max delta_t in seconds

K_INPUT_SET = [100]
SAMPLE_SET  = [0]
TARGET_SET  = list(range(N))

# --------------------------------------------------
# SLURM / resource settings
# --------------------------------------------------

MAX_WORKERS = 360
BASE_SEED   = 42
JOB_NAME    = "atk_pwr"
SBATCH_TEMPLATE = "../../01_code/build/run_atk.sh"

# --------------------------------------------------
# Output / logging
# --------------------------------------------------
 
LOG_DIR    = "logs_attack_power"
RESULT_DIR = "result_raw4"
 
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
        for _ in range(6):
            time.sleep(5)
            initial = slurm_named_jobs(job_name, username)
            if initial > 0:
                break
    if initial == 0:
        print("No jobs found in queue — may have completed immediately.")
        return
    print(f"\nMonitoring {initial} submitted jobs  (polling every {poll_interval}s)...")
    while True:
        remaining = slurm_named_jobs(job_name, username)
        completed = initial - remaining
        pct = 100.0 * completed / initial
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
 
 
def make_seed(N_val: int, sam_num: int, k_input: int,
              T_val: int, target: int) -> int:
    seed = (
        BASE_SEED
        + 1_000_000_007 * (N_val + 1)
        + 1_000_003     * (sam_num + 1)
        + 10_007        * (target + 1)
        + 1_009         * (k_input + 1)
        + 101           * (T_val + 1)
    )
    return seed & ((1 << 63) - 1)
 
# --------------------------------------------------
# Preflight checks
# --------------------------------------------------
 
def check_executable() -> None:
    if not os.path.isfile(EXECUTABLE):
        raise FileNotFoundError(
            f"Executable not found: {EXECUTABLE}\n"
            "Compile first:\n"
            "  gcc -O2 -Wall -Wextra -o build/attack_power_server "
            "attack_power_server.c mt19937-64.c -lm"
        )
    if not os.access(EXECUTABLE, os.X_OK):
        raise PermissionError(f"Not executable: {EXECUTABLE}")
 
 
def check_sbatch_template() -> None:
    if not os.path.isfile(SBATCH_TEMPLATE):
        raise FileNotFoundError(f"sbatch template not found: {SBATCH_TEMPLATE}")
 
 
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
# Job submission
# --------------------------------------------------
 
def submit_job(N_val: int, sam_num: int, k_input: int,
               T_val: int, target: int, abs_executable: str) -> None:
    seed = make_seed(N_val, sam_num, k_input, T_val, target)
 
    log_name = f"N{N_val}_sam{sam_num}_tgt{target}_K{k_input}_seed{seed}"
    log_out  = os.path.abspath(os.path.join(LOG_DIR, log_name + ".out"))
    log_err  = os.path.abspath(os.path.join(LOG_DIR, log_name + ".err"))
 
    # argv: N sam_num Kr_input T target REPEAT [EARLY_EXIT] [P_MAX_FACTOR] [P_MIN_FACTOR] [T_MAX_SEC] [seed]
    c_cmd = (
        f"{abs_executable} {N_val} {sam_num} {k_input} {T_val} "
        f"{target} {REPEAT} {EARLY_EXIT} {P_MAX_FACTOR} {P_MIN_FACTOR} {T_MAX_SEC} {seed}"
    )
 
    tmp_sh = os.path.abspath(
        f"atk_tmp_pwr_{N_val}_{sam_num}_{target}_{k_input}.sh"
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
    print("=" * 56)
    print("5_run_attack_power_server.py  (power perturbation, SLURM)")
    print("=" * 56)
    print(f"Executable        : {EXECUTABLE}")
    print(f"N                 : {N}")
    print(f"T                 : {T}")
    print(f"REPEAT            : {REPEAT}")
    print(f"EARLY_EXIT        : {EARLY_EXIT}")
    print(f"P_MAX_FACTOR      : {P_MAX_FACTOR}")
    print(f"P_MIN_FACTOR      : {P_MIN_FACTOR}")
    print(f"T_MAX_SEC         : {T_MAX_SEC}")
    print(f"K_INPUT_SET       : {K_INPUT_SET}")
    print(f"SAMPLE_SET        : {SAMPLE_SET}")
    print(f"TARGET_SET        : {TARGET_SET}")
    print(f"MAX_WORKERS       : {MAX_WORKERS}")
    print(f"BASE_SEED         : {BASE_SEED}")
    print(f"JOB_NAME          : {JOB_NAME}")
    print(f"SBATCH_TEMPLATE   : {SBATCH_TEMPLATE}")
    print(f"LOG_DIR           : {LOG_DIR}")
    print(f"RESULT_DIR        : {RESULT_DIR}")
    print(f"(mass/gamma loaded per-node from node file)")
    print(f"Total jobs        : {len(jobs)}")
    print("=" * 56)
 
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
 
    jobs = [
        (n, s, k, T, tgt)
        for n, s, k in ready_combos
        for tgt in TARGET_SET
    ]
 
    print_settings(jobs)
 
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
    print(f"\nResults in: {RESULT_DIR}/")
    print("Done.")
 
 
if __name__ == "__main__":
    main()