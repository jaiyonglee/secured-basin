# convert_pantaGruEl_flat_to_network.py
#
# Purpose:
#   Convert PanTaGruEl country data into simulation-ready network files:
#     {N}_sample0_nodes.txt      (id  P_i  mass  gamma)
#     {N}_sample0_edges.txt      (u   v    K_ij)
#     {country}_{N}_sample0.gml  (visualization with lat/lon)
#
# Source: PanTaGruEl (Pagnier & Jacquod, PLOS ONE, 2019)
#   https://doi.org/10.1371/journal.pone.0213550
#
# Input files (all in DATA_DIR, no subfolders):
#   {CC}_M_D_P.txt              — mass  gamma  P_i   (one row per node)
#   {CC}_edge_weights.txt       — u  v  K_ij          (0-indexed, per-unit)
#   {CC}_position_long_lat.txt  — lon  lat             (one row per node)
#
# Notes:
#   - K_ij = 0 edges are cross-border ties with no impedance data → skipped.
#   - Data is already in per-unit / normalized form → no unit conversion needed.
#   - sum(P_i) ≈ 0 for all countries → no scaling needed.
#
# Usage:
#   python3 convert_pantaGruEl_flat_to_network.py
#   (Change COUNTRY below to switch: UK | CH | DE | ES | FR)

from pathlib import Path
from collections import defaultdict
import numpy as np

# ===== 사용자 설정 =====
COUNTRY    = "UK"          # UK | CH | DE | ES | FR

# 입력 파일들이 있는 폴더 (서브폴더 없이 {CC}_*.txt 가 직접 있는 곳)
DATA_DIR   = Path(__file__).resolve().parent

# 출력 폴더 (nodes.txt, edges.txt, gml 저장 위치)
OUT_DIR    = Path(__file__).resolve().parent

SAMPLE_NUM = 0

# K=0 처리: None = 제거 (cross-border tie),  숫자 = 해당 값으로 대체
K_ZERO_REPLACE = None

# ---- 질량 보정 설정 ----
# 2차 쿠라모토 모델에서 M=0 노드는 수렴 불가.
# M=0 (load/변전소) 노드에 대해 발전기 노드의 M~D 선형 회귀를 이용해 M을 추정.
#
# MASS_IMPUTE_METHOD:
#   'regression' — 발전기 데이터로 OLS (M = a*D, 원점 통과) 후 load 노드에 적용
#   'min_gen'    — load 노드에 발전기 최솟값 고정
#   'none'       — 보정 없음 (M=0 그대로)
MASS_IMPUTE_METHOD = 'regression'   # 'regression' | 'min_gen' | 'none'

# M_FLOOR: 보정 후에도 이 값 이하면 강제로 M_FLOOR 적용 (0 방지 안전장치)
M_FLOOR = 1e-4

# Per-node 개별 오버라이드 (0-indexed, 보정 후 덮어씀)
P_OVERRIDE     = {}   # e.g. {0: 1.5}
MASS_OVERRIDE  = {}
GAMMA_OVERRIDE = {}
# ======================


def load_mdp(path):
    """mass  gamma  P_i 를 노드 순서대로 읽어 반환."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            rows.append((float(v[0]), float(v[1]), float(v[2])))
    return rows


def load_edges(path, k_zero_replace=None):
    """
    u  v  K_ij 를 읽어 (u, v, K) 리스트 반환.
    K=0 엣지: k_zero_replace=None 이면 제거, 숫자면 해당 값으로 대체.
    병렬 엣지는 K 합산.
    """
    edge_K = defaultdict(float)
    skipped = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            u, w, K = int(v[0]), int(v[1]), float(v[2])
            if u > w:
                u, w = w, u
            if K < 1e-12:
                if k_zero_replace is not None:
                    K = float(k_zero_replace)
                else:
                    skipped += 1
                    continue
            edge_K[(u, w)] += K

    if skipped:
        print(f"  Skipped {skipped} K=0 edges (cross-border ties, no impedance data)")
    return [(u, w, K) for (u, w), K in sorted(edge_K.items())]


def load_positions(path):
    """lon  lat 를 노드 순서대로 읽어 반환."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            rows.append((float(v[0]), float(v[1])))
    return rows


def impute_mass(mdp, method='regression', m_floor=1e-4):
    """
    M=0 노드(load/변전소)의 질량을 보정한다.

    발전기 노드: M > 0  → 그대로 사용
    부하 노드:   M = 0  → method에 따라 추정

    method='regression':
        발전기 데이터로 OLS (원점 통과): M = a * D
        a = sum(M_gen * D_gen) / sum(D_gen^2)
        부하 노드 M_imputed = a * D_load
        (M과 D의 상관관계가 R²>0.94 로 매우 강함)

    method='min_gen':
        부하 노드 M_imputed = min(M_gen)

    method='none':
        보정 없음 (M=0 그대로)

    Returns: list of (mass_new, gamma, pi)
    """
    M = np.array([r[0] for r in mdp])
    D = np.array([r[1] for r in mdp])
    P = np.array([r[2] for r in mdp])

    gen_mask  = M > 1e-9
    load_mask = ~gen_mask

    n_gen  = gen_mask.sum()
    n_load = load_mask.sum()

    print(f"  Mass imputation  : method='{method}', M_FLOOR={m_floor}")
    print(f"  Generator nodes  : {n_gen}  (M>0, used for regression)")
    print(f"  Load nodes       : {n_load}  (M=0, to be imputed)")

    if method == 'none' or n_load == 0:
        print(f"  No imputation applied.")
        return list(mdp)

    M_new = M.copy()

    if method == 'regression':
        # OLS through origin: a = Σ(M*D) / Σ(D²) on generator nodes
        D_gen = D[gen_mask]
        M_gen = M[gen_mask]
        a = np.dot(M_gen, D_gen) / np.dot(D_gen, D_gen)

        M_pred_gen = a * D_gen
        ss_res = np.sum((M_gen - M_pred_gen)**2)
        ss_tot = np.sum((M_gen - M_gen.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        print(f"  OLS result       : M = {a:.4f} × D  (R²={r2:.4f} on gen nodes)")

        M_new[load_mask] = a * D[load_mask]

    elif method == 'min_gen':
        m_min = float(M[gen_mask].min())
        print(f"  Min gen mass     : {m_min:.6f}")
        M_new[load_mask] = m_min

    # Apply floor
    below_floor = M_new < m_floor
    if below_floor.any():
        print(f"  M_FLOOR applied  : {below_floor.sum()} nodes raised to {m_floor}")
        M_new[below_floor] = m_floor

    # Stats after imputation
    print(f"  Final M range    : [{M_new.min():.6f}, {M_new.max():.6f}]  "
          f"mean={M_new.mean():.6f}")
    print(f"  Gen  M range     : [{M_new[gen_mask].min():.6f}, {M_new[gen_mask].max():.6f}]")
    print(f"  Load M range     : [{M_new[load_mask].min():.6f}, {M_new[load_mask].max():.6f}]")

    return [(M_new[i], mdp[i][1], mdp[i][2]) for i in range(len(mdp))]


def write_nodes(path, mdp):
    zero = 0
    with open(path, 'w') as f:
        for i, (mass, gamma, pi) in enumerate(mdp):
            pi    = P_OVERRIDE.get(i, pi)
            mass  = MASS_OVERRIDE.get(i, mass)
            gamma = GAMMA_OVERRIDE.get(i, gamma)
            if abs(pi) < 1e-9:
                zero += 1
            f.write(f"{i} {pi:.15f} {mass:.6f} {gamma:.6f}\n")
    print(f"  Written: {path}  ({len(mdp)} nodes, "
          f"{len(mdp)-zero} P≠0, {zero} P=0)")


def write_edges(path, edges):
    with open(path, 'w') as f:
        for u, v, K in edges:
            f.write(f"{u} {v} {K:.15f}\n")
    print(f"  Written: {path}  ({len(edges)} edges)")


def write_gml(path, mdp, edges, positions, cc):
    with open(path, 'w') as f:
        f.write("graph [\n")
        f.write("  directed 0\n")
        for i, ((mass, gamma, pi), (lon, lat)) in enumerate(zip(mdp, positions)):
            pi    = P_OVERRIDE.get(i, pi)
            mass  = MASS_OVERRIDE.get(i, mass)
            gamma = GAMMA_OVERRIDE.get(i, gamma)
            f.write("  node [\n")
            f.write(f"    id {i}\n")
            f.write(f"    label \"{cc}_{i}\"\n")
            f.write(f"    power {pi:.15f}\n")
            f.write(f"    mass {mass:.6f}\n")
            f.write(f"    gamma {gamma:.6f}\n")
            f.write(f"    pos {lon:.6f}\n")
            f.write(f"    pos {lat:.6f}\n")
            f.write("  ]\n")
        for u, v, K in edges:
            f.write("  edge [\n")
            f.write(f"    source {u}\n")
            f.write(f"    target {v}\n")
            f.write(f"    weight {K:.15f}\n")
            f.write("  ]\n")
        f.write("]\n")
    print(f"  Written: {path}  (GML with lat/lon)")


def summary(mdp, edges, cc):
    Pi   = [P_OVERRIDE.get(i, r[2])    for i, r in enumerate(mdp)]
    M    = [MASS_OVERRIDE.get(i, r[0]) for i, r in enumerate(mdp)]
    K    = [k for _, _, k in edges]
    print(f"\n  Country   : {cc}")
    print(f"  Nodes     : {len(mdp)}")
    print(f"  Edges     : {len(edges)}")
    print(f"  P range   : [{min(Pi):.4f}, {max(Pi):.4f}]")
    print(f"  sum(P)    : {sum(Pi):.6f}  (should be ~0)")
    if K:
        print(f"  K range   : [{min(K):.4f}, {max(K):.4f}]")
    nonzero_m = [m for m in M if m > 1e-9]
    if nonzero_m:
        print(f"  mass (>0) : [{min(nonzero_m):.6f}, {max(nonzero_m):.6f}]  "
              f"({len(nonzero_m)} nodes)")


def main():
    cc = COUNTRY.upper()

    mdp_path  = DATA_DIR / f"{cc}_M_D_P.txt"
    edge_path = DATA_DIR / f"{cc}_edge_weights.txt"
    pos_path  = DATA_DIR / f"{cc}_position_long_lat.txt"

    for p in [mdp_path, edge_path, pos_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"File not found: {p}\n"
                f"DATA_DIR = {DATA_DIR}\n"
                f"Expected: {{CC}}_M_D_P.txt, {{CC}}_edge_weights.txt, "
                f"{{CC}}_position_long_lat.txt"
            )

    print("=" * 52)
    print(f"convert_pantaGruEl_flat_to_network.py  ({cc})")
    print("=" * 52)
    print(f"Data dir : {DATA_DIR}")
    print(f"Out dir  : {OUT_DIR}")
    print()

    mdp       = load_mdp(mdp_path)
    edges     = load_edges(edge_path, k_zero_replace=K_ZERO_REPLACE)
    positions = load_positions(pos_path)

    assert len(mdp) == len(positions), \
        f"Node count mismatch: M_D_P={len(mdp)}, position={len(positions)}"

    # ---- 질량 보정 ----
    print()
    mdp = impute_mass(mdp, method=MASS_IMPUTE_METHOD, m_floor=M_FLOOR)

    summary(mdp, edges, cc)
    print()

    N      = len(mdp)
    prefix = f"{N}_sample{SAMPLE_NUM}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_nodes(OUT_DIR / f"{prefix}_nodes.txt", mdp)
    write_edges(OUT_DIR / f"{prefix}_edges.txt", edges)
    write_gml(  OUT_DIR / f"{cc.lower()}_{prefix}.gml", mdp, edges, positions, cc)

    print()
    print("Notes:")
    print("  Data already in per-unit form — no conversion needed.")
    print("  K=0 edges (cross-border ties) are skipped by default.")
    print(f"  Mass imputation: MASS_IMPUTE_METHOD='{MASS_IMPUTE_METHOD}', M_FLOOR={M_FLOOR}")
    print("  Override per-node values via P/MASS/GAMMA_OVERRIDE dicts.")
    print("\nDone.")


if __name__ == "__main__":
    main()