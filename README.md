# Secured basin

Research code for analyzing secured basins and energy barriers in the second-order Kuramoto model with inertia and damping. The repository includes equilibrium computation, single-node perturbation experiments in state space and power injection, and scripts for generating manuscript figures and tables.

The secured basin is defined through trajectories that remain within the phase-angle boundary π on every connected pair throughout the observation period. This distinguishes transient constraint violations from eventual recovery of synchronization.

## Repository structure

```text
secured-basin/
├── 01_code/                 # C simulators and random number generator
│   └── build/
│       └── run_atk.sh       # Example SLURM template; adapt to your cluster
├── 02_network/              # Network inputs and simulation launchers
│   ├── 01_motif_3/
│   ├── 02_South_Korea_193/
│   ├── 04_UK_54/
│   ├── 05_UK_313/
│   ├── 06_Swiss_161/
│   ├── 07_Germany_553/
│   ├── 08_Spain_838/
│   └── 09_France_847/
└── 03_paper_figure/          # Preprocessing, figures, and tables
```

The experiments use a three-node motif and seven power-grid models. The three-node motif is included for demonstration, whereas the original third-party power-grid datasets used in the paper are not redistributed through this repository. See **Data availability** below for information on obtaining them.

## Requirements and compilation

- GCC and the C math library (`libm`).
- Python 3.
- For parallel perturbation experiments: a SLURM cluster with `sbatch` and `squeue` available.
- For preprocessing and figures: NumPy, pandas, Matplotlib, and NetworkX. `tqdm` is optional.

```bash
python3 -m pip install numpy pandas matplotlib networkx tqdm

cd 01_code
mkdir -p build
gcc -O2 -Wall -Wextra -o build/get_init get_init.c mt19937-64.c -lm
gcc -O2 -Wall -Wextra -o build/attack_random_server attack_random_server.c mt19937-64.c -lm
gcc -O2 -Wall -Wextra -o build/attack_power_server attack_power_server.c mt19937-64.c -lm
cd ..
```

`mt19937-64.c` provides the random number generator and is linked with the simulators. Recompile the corresponding executable whenever you change a C preprocessor constant (`#define`).

## 01_code: C simulators

| File | Purpose |
| --- | --- |
| `get_init.c` | Integrate the dynamics and save an equilibrium that satisfies synchronization and power-balance criteria. |
| `attack_random_server.c` | Apply initial phase and angular-velocity perturbations to one node, using uniform sampling or a search concentrated near the energy boundary. |
| `attack_power_server.c` | Apply a temporary power-injection perturbation to one node and simulate its response. |
| `mt19937-64.c` | 64-bit Mersenne Twister random number generator. |

The simulators integrate the following equations using the fourth-order Runge–Kutta method:

```text
dθ_i/dt = ω_i
M_i dω_i/dt = P_i - γ_i ω_i + Σ_j K_ij sin(θ_j - θ_i)
```

### Compile-time parameters

| Constant | Current value | Meaning |
| --- | --- | --- |
| `H` | `0.001` | Integration time step in seconds, defined separately in all three simulators. |
| `SYNC_TOL` | `5e-2` | A node is classified as synchronized when `abs(ω_i) < SYNC_TOL`. |
| `RESIDUAL_TOL` | `1e-3` | Maximum power-balance residual allowed when saving an equilibrium in `get_init.c`. |
| `W_MAX` | `20.0` | Angular-velocity perturbation bound: `Δω ∈ [-W_MAX, W_MAX]` in state-space experiments. |
| `PROGRESS_EVERY` | `1000` | Progress logging interval measured in perturbation trials. |
| `EARLY_FRAC` | `0.01` | Fraction of the simulation used to record the maximum absolute angular velocity across all nodes during the initial observation window. |

The equilibrium residual is

```text
max_residual = max_i abs(P_i + Σ_j K_ij sin(θ_j - θ_i))
```

Small angular velocities alone do not ensure power balance. `get_init` saves an equilibrium only if all nodes are synchronized and `max_residual < RESIDUAL_TOL`.

`EARLY_FRAC` controls the diagnostic `max_abs_w_early` used for survivability comparisons in state-space experiments. The measurement includes the initial state and the first `floor(T × EARLY_FRAC)` integration steps. It does not affect the `safe` classification or early termination. For example, `T=500000` and `H=0.001` give a 500-second simulation with a 5-second initial observation window.

### Outcome classification

In the current C implementation, `safe=1` requires **all nodes to be synchronized at termination and no detected phase-boundary crossing**. This additional synchronization requirement should be distinguished from the manuscript's trajectory-based definition of secured outcomes. The perturbation simulators record `max_residual` as a diagnostic but do not use it in the `safe` classification.

When early termination is enabled, `num_synced` and the final-state columns describe the state at the detected crossing. Use a full simulation to assess eventual synchronization or basin stability. See the power-launcher note below for its current handling of `EARLY_EXIT`.

## 02_network: inputs and simulation workflow

## Data availability

The original network datasets used in the paper are **not distributed with this repository**. Users who wish to reproduce the experiments should obtain the corresponding datasets directly from their original sources or according to the data-access procedures described in the cited references.

- **UK 54-node network:** obtain following the data-access information provided in the corresponding reference.
- **South Korean 193-node network:** obtain following the data-access information provided in the corresponding reference.
- **Swiss 161-node, UK 313-node, German 553-node, Spanish 838-node, and French 847-node networks:** these datasets were provided separately to the authors. Their original sources are cited in the paper as requested by the data providers.

After obtaining the datasets, convert them to the input format described in the **Network files** section of this README and place them in the corresponding dataset directories under `02_network/`.

The repository provides the source code and processing workflow, but users are responsible for obtaining any third-party datasets and complying with their respective access, licensing, and redistribution conditions.

### Network files

Each dataset's `network/` directory contains two simulator input files per sample. For S samples, there are `2 × S` input files; conversion scripts and source data may also be present.

| Filename | Columns in each row |
| --- | --- |
| `{N}_sample{sample}_nodes.txt` | `node_id power inertia damping` |
| `{N}_sample{sample}_edges.txt` | `node_id_1 node_id_2 coupling_strength` |

Files contain whitespace-separated numeric values without headers. Node IDs range from `0` to `N-1`. The node file supplies power injection `P_i`, inertia `M_i`, and damping `γ_i`. Edges are read as undirected connections, so each connection should appear once.

The coupling used in the simulation is `edge-file coupling × Kr`, where `Kr = K_input / 100`. For example, `K_input=100` gives a multiplier of 1.00, and `300` gives 3.00.

### Execution order

Edit the settings at the top of each launcher, then run it **from its dataset directory**. The launchers resolve relative paths from the current working directory.

```bash
cd 02_network/01_motif_3
python3 1_run_get_init.py
python3 2_run_attack_basin_server.py
python3 3_run_attack_power_server.py
```

The first script computes equilibria locally and sequentially. The second and third submit independent SLURM jobs for each `(sample, target, K_input)` combination. Repeated trials within a job run in a single C process. State-space and power-injection experiments can be run independently once their required equilibrium files exist.

### 1_run_get_init.py: equilibrium computation

| Variable | Meaning |
| --- | --- |
| `N` | Number of nodes. |
| `T` | Total integration steps; the physical observation time is `T × H` seconds. |
| `K_INPUT_SET` | List of coupling multipliers expressed as integers scaled by 100. |
| `SAMPLE_SET` | List of sample IDs to process. |
| `EXECUTABLE` | Path to the compiled `get_init` executable. |

Equilibria are written to `init/{N}_sample{sample}_init_Kr_{Kr:.2f}.txt` with columns `node_id omega theta`. Phases are normalized relative to node 0. If the equilibrium criteria are not met, no output is saved; check the integration horizon, coupling strength, and network inputs.

Existing equilibrium files are skipped. Archive or remove the relevant file before recomputing after changes to the network or simulation settings. The perturbation launchers skip combinations with missing equilibrium files.

### 2_run_attack_basin_server.py: state-space perturbations

`N`, `T`, `K_INPUT_SET`, and `SAMPLE_SET` have the same meanings as in the equilibrium launcher.

| Variable | Meaning |
| --- | --- |
| `REPEAT` | Trial budget per combination, including initial anchor trials in the concentrated search. |
| `TARGET_SET` | Target node IDs; `list(range(N))` selects every node. |
| `EARLY_EXIT` | `True`: terminate upon detecting a phase-boundary crossing. `False`: integrate all `T` steps. |
| `INIT_ONLY` | `True`: uniform sampling. `False`: initial exploration followed by sampling near the energy boundary. |
| `UEP_PERTURB` | Applies to uniform sampling only. `True` disables the UEP restriction and samples the full phase range. |

The unstable equilibrium point (UEP) is estimated using the local mean field. When enabled, the UEP restriction excludes the phase interval on the far side of the estimated UEP. If the estimate is invalid, the code falls back to unrestricted sampling.

| `INIT_ONLY` | `UEP_PERTURB` | Sampling mode | Output directory |
| --- | --- | --- | --- |
| `True` | `False` | Uniform sampling with the UEP restriction. | `result_raw1/` |
| `True` | `True` | Uniform sampling over the full phase range `[-π, π]`. | `result_raw2/` |
| `False` | Either | Concentrated boundary search with the UEP restriction. | `result_raw3/` |

Run each configuration separately to generate the required output directories. The concentrated search uses anchor trials and initial uniform sampling to obtain both safe and unsafe outcomes before focusing on the energy boundary. It may not enter the concentrated phase if both outcomes are not found within the trial budget.

### 3_run_attack_power_server.py: power-injection perturbations

Starting from equilibrium, this simulator temporarily changes the target node's power injection to `P_target + ΔP`, then restores the original injection. Results are saved to `result_raw4/`. This launcher does not use `INIT_ONLY` or `UEP_PERTURB`.

| Variable | Meaning |
| --- | --- |
| `P_MAX_FACTOR` | Upper magnitude factor: `P_max = P_MAX_FACTOR × Σ_j K_target,j`. |
| `P_MIN_FACTOR` | Lower magnitude factor: `P_min = P_MIN_FACTOR × Σ_j K_target,j`. |
| `T_MAX_SEC` | Maximum perturbation duration `Δt` in seconds. |

The weighted degree `Σ_j K_target,j` includes the coupling multiplier `Kr`. The magnitude `abs(ΔP)` is sampled uniformly in logarithmic space over `[P_min, P_max]`, with a random sign. The duration `Δt` is also sampled uniformly in logarithmic space over `[H, T_MAX_SEC]`. The applied duration is converted to integer integration steps and capped at `T` steps. Set `0 < P_MIN_FACTOR < P_MAX_FACTOR` and `H ≤ T_MAX_SEC ≤ T × H`.

**Current launcher behavior:** the Python launcher passes Boolean `EARLY_EXIT` values as the strings `True` or `False`, while the C program parses the argument using `atoi`. Both strings are therefore interpreted as `0`, resulting in full integration. With the current launcher, set `EARLY_EXIT` to the integer `1` for early termination or `0` for full integration.

### SLURM configuration

| Variable | Meaning |
| --- | --- |
| `MAX_WORKERS` | Submission limit based on the specified user's queued and running SLURM jobs. It does not set the C process's thread count. |
| `BASE_SEED` | Base value for deriving a random seed for each parameter combination; currently `42`. |
| `JOB_NAME` | Job name used for submission and queue monitoring. |
| `SBATCH_TEMPLATE` | Base job-script template; defaults to `../../01_code/build/run_atk.sh`. |

Replace the default username `jaiyong` in the launchers' SLURM query functions with your cluster account name. Jobs sharing the same `JOB_NAME` may be counted together during monitoring.

An example template is provided at [01_code/build/run_atk.sh](01_code/build/run_atk.sh):

```bash
#!/bin/sh

# job name (overridden per-job by the launcher via #SBATCH --job-name)
#SBATCH -J atk_rnd

# output and error files (overridden per-job by the launcher)
#SBATCH -o logs_attack_random/%x.o%j
#SBATCH -e logs_attack_random/%x.e%j

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
```

This is an **example**, not a cluster-specific configuration. It requests one node with one task per node. Adapt the partition, account, memory, time limit, and any other required resource directives to your cluster before submitting jobs.

The Python launchers copy this template and append the job name, output/error paths, and C execution command for each parameter combination. The appended settings provide the appropriate names and log paths for either experiment type. Run the Python launcher to submit experiments; the template itself contains no simulation command.

Keep the template limited to the shebang, comments, and `#SBATCH` directives. Adding executable shell statements before the launcher's appended directives can prevent those directives from being recognized by SLURM.

### Output files

Each `result_raw*/` directory contains files named `{N}_analysis{sample}_n_{target}_Kr_{Kr:.2f}.csv`. Rerunning the same combination in the same output directory overwrites its CSV.

- Common columns include the target node, coupling multiplier, `num_synced`, `safe`, final states `w_*` and `theta_*`, diagnostics, trial index, and seed.
- State-space outputs include `attack_w`, `attack_theta`, perturbation energy `E_temp`, and `max_abs_w_early`.
- Power-injection outputs include `attack_P`, `attack_t`, and states at the end of the perturbation (`w_at_dt_*`, `theta_at_dt_*`). In this simulator, `E_temp` is `ΔP × Δt`.
- `E_safe_max` is the largest observed energy among safe outcomes; `E_danger_min` is the smallest among unsafe outcomes. Distinguish these column names from the manuscript's energy-boundary symbols.

Logs are saved to `logs_attack_random/` and `logs_attack_power/`, respectively.

## 03_paper_figure: manuscript figures and tables

After generating the simulation data, run preprocessing and the required plotting scripts:

```bash
cd 03_paper_figure
python3 fig_preprocess.py
python3 fig_1.py
python3 fig_2.py
python3 fig_3.py
python3 fig_4.py
python3 fig_5.py
python3 fig_6.py
python3 fig_7.py
python3 table_1_8.py
```

`fig_preprocess.py` reads `result_raw1/`, network inputs, and equilibria for the seven power-grid models, then writes `result/0_rank.csv` in each dataset directory. Its current defaults are `SAMPLE_NUM=0` and `KR=1.00`. Match the preprocessing and plotting settings when analyzing other samples or coupling multipliers.

Each `fig_N.py` creates a directory named `fig_N/` and saves panel PDFs there. `table_1_8.py` saves tables and related outputs under `table_1_8/`. The data must match the input paths, target nodes, sample IDs, and coupling multipliers configured in each script.

| Script | Main inputs |
| --- | --- |
| `fig_1.py` | Motif `result_raw2`, network inputs, and equilibrium. |
| `fig_2.py` | Motif `result_raw1`, network inputs, and equilibrium. |
| `fig_3.py` | UK 54 `result_raw1`, `result_raw3`, `result/0_rank.csv`, and related inputs. |
| `fig_4.py` | Configured UK 54 dataset's `result_raw1`, `result_raw3`, `result_raw4`, and related inputs. |
| `fig_5.py`, `fig_6.py` | Power-grid `result/0_rank.csv` files and related network data. |
| `fig_7.py` | Motif `result_raw1` and `result_raw3` by default, plus related inputs. |
| `table_1_8.py` | Power-grid `result/0_rank.csv`, `result_raw4`, network inputs, and equilibria. |

Equilibrium and simulation outputs are not currently included in this repository folder. In addition, `fig_4.py` currently points to `02_network/99_test_UK_54`, which is not included. Before running it, supply that dataset or update `NET_ROOT` to the intended dataset directory.

## License

The original code in this repository is released under the [MIT License](LICENSE).
`01_code/mt19937-64.c` retains the copyright notice and license conditions provided by its original authors in the file header. Third-party datasets are not relicensed by the software license.
