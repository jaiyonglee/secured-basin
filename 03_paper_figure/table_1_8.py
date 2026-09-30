# table_1_8.py
# 출력: 2026_Safe_Zone/03_paper_figure/table_1_8/
#   table1.csv ~ table7.csv — 국가별 노드단위 threshold violation rate (epsilon_i)
#   table8.csv               — 국가별 요약 통계
#
# 방법론 요약 (논문 식 threshold_violation 참고):
#   - 각 노드 i에 대해 E_i^S(secured 임계, 초록), E_i^U(unsecured 임계, 빨강) 두 경계로
#     Δt vs ΔP 평면에 두 곡선을 그림 (fig_4.py의 analytic curve와 동일 공식):
#         ΔP(Δt) = sqrt(2E/M_i) * D_i / (1 - exp(-D_i/M_i * Δt))
#   - raw4의 각 섭동 표본 (Δt, ΔP, outcome)에 대해:
#       |ΔP| < ΔP_S(Δt)  (초록선 안쪽, "secured 기대")
#           → 실제 secured면 N_in, 아니면(=unsecured, desync 포함) N_out
#       |ΔP| > ΔP_U(Δt)  (빨간선 바깥쪽, "unsecured 기대")
#           → 실제 unsecured(desync 포함)면 N_in, 아니면(=secured) N_out
#       그 사이(mixed-outcome interval)는 집계에서 제외
#   - epsilon_i = N_out / (N_in + N_out)
#   - secured = safe 판정, unsecured = danger 판정 + desync 전부 포함 (이진 분류)
#   - Δt 범위는 T/4 이내로 제한 (fig_4.py의 quarter_T와 동일)
#
# 설계 메모: E_i^S, E_i^U는 raw1를 직접 읽지 않고, 사전계산된
#   result/0_rank.csv의 energy_safety_mean(m_ES), energy_safety_dev(sigma_ES)로부터
#   E_S = m/sigma, E_U = m*sigma 로 역산해서 사용한다 (raw1 재로딩 비용 절감).
#   이 방식은 우선 시도해보는 것이며, 오차가 크면 추후 raw1 직접 로딩 방식으로
#   교체할 수 있음을 전제로 한다.

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 경로 ────────────────────────────────────────────────────────
_base     = Path(__file__).resolve().parent
NET_ROOT  = _base.parent / "02_network"
OUT_DIR   = _base / "table_1_8"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUT_DIR / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── 데이터셋 정의 (table1~table7 순서 그대로) ─────────────────────
DATASETS_D = [
    ("04_UK_54",           "UK 54",       54),
    ("06_Swiss_161",       "Swiss 161",   161),
    ("02_South_Korea_193", "Korea 193",   193),
    ("05_UK_313",          "UK 313",      313),
    ("07_Germany_553",     "Germany 553", 553),
    ("08_Spain_838",       "Spain 838",   838),
    ("09_France_847",      "France 847",  847),
]

# 데이터 문제로 제외할 노드
EXCLUDE_NODES = {
    "07_Germany_553": {549},
}

SAMPLE_NUM = 0
KR = 1.00

# ── 스타일 (다른 fig_*.py와 통일) ─────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        11,
    'axes.linewidth':   1.2,
    'axes.labelsize':   12,
    'legend.fontsize':  9,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'mathtext.fontset': 'stix',
})
COLORS = ['#4C9BE8', '#E8954C', '#53AB38', '#D94946',
          '#9B59B6', '#16A085', '#8B5A2B']


# ================================================================
# 네트워크 로더 (fig_4.py의 load_network 패턴, 인접리스트 버전)
# ================================================================
def load_network(net_root, node_num):
    net_dir  = net_root / "network"
    init_dir = net_root / "init"

    P = np.zeros(node_num); M = np.ones(node_num); D = np.full(node_num, 0.1)
    with open(net_dir / f"{node_num}_sample{SAMPLE_NUM}_nodes.txt") as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            i = int(parts[0])
            if len(parts) > 1: P[i] = float(parts[1])
            if len(parts) > 2: M[i] = float(parts[2])
            if len(parts) > 3: D[i] = float(parts[3])

    adj = {i: [] for i in range(node_num)}  # node -> [(neighbor, K_ij), ...]
    with open(net_dir / f"{node_num}_sample{SAMPLE_NUM}_edges.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2]) * KR
            if w > 0:
                adj[u].append((v, w))
                adj[v].append((u, w))

    theta_eq = np.zeros(node_num)
    with open(init_dir / f"{node_num}_sample{SAMPLE_NUM}_init_Kr_{KR:.2f}.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                theta_eq[int(parts[0])] = float(parts[2])

    return P, M, D, adj, theta_eq


def quarter_period(node_num, attack_node, M_i, D_i, adj, theta_eq):
    """T/4 계산: ω_d = sqrt(k_i/M_i - (D_i/2M_i)^2). 계산 불가하면 None."""
    k_i = sum(w * np.cos(theta_eq[attack_node] - theta_eq[j])
              for j, w in adj[attack_node])
    omega_n = np.sqrt(k_i / M_i) if k_i > 0 else None
    alpha_  = D_i / (2.0 * M_i)
    if omega_n and omega_n > alpha_:
        omega_d = np.sqrt(omega_n**2 - alpha_**2)
        return np.pi / (2.0 * omega_d)
    return None


def dP_threshold(E_val, M_i, D_i, t):
    """ΔP(Δt) = sqrt(2E/M) * D / (1 - exp(-D/M * Δt))"""
    exp_term = np.exp(-D_i / M_i * t)
    denom = np.where(1.0 - exp_term < 1e-12, 1e-12, 1.0 - exp_term)
    return np.sqrt(2.0 * E_val / M_i) * D_i / denom


# ================================================================
# 노드 하나 처리: N_in, N_out 계산 (캐시 활용)
# ================================================================
def process_node(folder, node_num, node, P, M, D, adj, theta_eq, E_safety_i, E_danger_i):
    cache_fp = CACHE_DIR / f"{folder}_node{node}.npz"
    if cache_fp.exists():
        d = np.load(cache_fp)
        return int(d['N_in']), int(d['N_out'])

    net_root = NET_ROOT / folder
    raw4_fp = net_root / "result_raw4" / f"{node_num}_analysis0_n_{node}_Kr_{KR:.2f}.csv"
    if not raw4_fp.exists():
        return None  # 파일 없음

    if not (np.isfinite(E_safety_i) and np.isfinite(E_danger_i) and E_safety_i > 0):
        return None  # 임계값 계산 불가

    M_i = M[node]; D_i = D[node]

    t_max = quarter_period(node_num, node, M_i, D_i, adj, theta_eq)

    try:
        df4 = pd.read_csv(raw4_fp)
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if df4.empty or 'attack_t' not in df4.columns or 'attack_P' not in df4.columns:
        return None

    if t_max is None:
        t_max = df4['attack_t'].max()

    df4 = df4[df4['attack_t'] <= t_max]
    if df4.empty:
        N_in, N_out = 0, 0
    else:
        has_safe   = 'safe' in df4.columns
        mask_sync  = df4['num_synced'] >= node_num
        secured    = (mask_sync & (df4['safe'] == 1)) if has_safe else mask_sync
        # unsecured = secured가 아닌 모든 경우 (sync&danger + desync 전부 포함)
        unsecured  = ~secured

        t_arr   = df4['attack_t'].values
        absP    = df4['attack_P'].abs().values
        dP_S    = dP_threshold(E_safety_i, M_i, D_i, t_arr)
        dP_U    = dP_threshold(E_danger_i, M_i, D_i, t_arr)

        mask_inner = absP < dP_S   # 초록선 안쪽: secured 기대
        mask_outer = absP > dP_U   # 빨간선 바깥쪽: unsecured 기대

        N_in  = int((mask_inner & secured.values).sum())
        N_in += int((mask_outer & unsecured.values).sum())
        N_out = int((mask_inner & unsecured.values).sum())
        N_out += int((mask_outer & secured.values).sum())

    np.savez(cache_fp, N_in=N_in, N_out=N_out)
    return N_in, N_out


# ================================================================
# 네트워크 하나 처리 → table_k.csv 생성
# ================================================================
def process_dataset(folder, label, node_num, table_idx):
    print(f"\n=== [table{table_idx}] {label} ({folder}, N={node_num}) ===")
    net_root = NET_ROOT / folder

    rank_fp = net_root / "result" / "0_rank.csv"
    if not rank_fp.exists():
        print(f"  WARNING: {rank_fp} not found, skip dataset")
        return None
    rank_df = pd.read_csv(rank_fp).set_index('node')

    print("  Loading network...")
    P, M, D, adj, theta_eq = load_network(net_root, node_num)

    exclude = EXCLUDE_NODES.get(folder, set())

    rows = []
    n_missing = 0
    for node in range(node_num):
        if node in exclude:
            continue
        if node % max(1, node_num // 10) == 0:
            print(f"    node {node}/{node_num} ({node/node_num:.0%})")

        if node not in rank_df.index:
            n_missing += 1
            continue
        m_es  = rank_df.loc[node, 'energy_safety_mean']
        sig_es = rank_df.loc[node, 'energy_safety_dev']
        if not (np.isfinite(m_es) and np.isfinite(sig_es) and sig_es > 0):
            n_missing += 1
            continue
        E_safety_i = m_es / sig_es   # 초록 임계 (secured)
        E_danger_i = m_es * sig_es   # 빨강 임계 (unsecured)

        result = process_node(folder, node_num, node, P, M, D, adj, theta_eq,
                              E_safety_i, E_danger_i)
        if result is None:
            n_missing += 1
            continue
        N_in, N_out = result
        N_total = N_in + N_out
        epsilon_i = N_out / N_total if N_total > 0 else np.nan
        rows.append({
            'node': node, 'N_in': N_in, 'N_out': N_out,
            'N_total': N_total, 'epsilon_i': epsilon_i,
        })

    if n_missing:
        print(f"  WARNING: {n_missing} nodes skipped (missing data/files)")

    if not rows:
        print(f"  WARNING: no valid nodes for {label}, skip")
        return None

    out_df = pd.DataFrame(rows).sort_values('node').reset_index(drop=True)
    out_fp = OUT_DIR / f'table{table_idx}.csv'
    out_df.to_csv(out_fp, index=False)
    print(f"  Saved: {out_fp}  ({len(out_df)} nodes)")
    return label, out_df


# ================================================================
# table8 — 국가별 요약 통계
# ================================================================
def build_table8(per_dataset_results):
    rows = []
    for label, df in per_dataset_results:
        valid = df['epsilon_i'].dropna()
        n_zero = int((valid == 0).sum())
        total  = len(valid)
        rows.append({
            'dataset': label,
            'mean_epsilon': valid.mean() if total else np.nan,
            'std_epsilon':  valid.std()  if total else np.nan,
            'min_epsilon':  valid.min()  if total else np.nan,
            'max_epsilon':  valid.max()  if total else np.nan,
            'p95_epsilon':  valid.quantile(0.95) if total else np.nan,
            'n_zero_error_nodes': n_zero,
            'ratio_zero_error_nodes': n_zero / total if total else np.nan,
            'n_valid_nodes': total,
        })
    out_df = pd.DataFrame(rows)
    out_fp = OUT_DIR / 'table8.csv'
    out_df.to_csv(out_fp, index=False)
    print(f"\nSaved: {out_fp}")
    return out_df


# ================================================================
# fig_epsilon_percentile — 데이터셋별 epsilon_i 오름차순 백분위 산점도
# x축: 백분위(0~100, 데이터셋 내에서 epsilon 오름차순 정렬 순위)
# y축: epsilon_i
# ================================================================
def plot_epsilon_percentile(per_dataset_results):
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    all_nonzero = []
    for (label, df), color in zip(per_dataset_results, COLORS):
        valid = df['epsilon_i'].dropna().sort_values().values
        n = len(valid)
        if n == 0:
            continue
        # 백분위: (순위+0.5)/n * 100  (0~100 사이에 고르게 위치)
        percentile = (np.arange(n) + 0.5) / n * 100

        ax.scatter(percentile, valid, c=color, s=10, alpha=0.8,
                  edgecolors='white', linewidths=0.2, label=label,
                  zorder=3, clip_on=False)

        nz = valid[valid > 0]
        if len(nz):
            all_nonzero.append(nz.min())

    ax.set_xlabel('Percentile (ascending $\\epsilon_i$)')
    ax.set_ylabel(r'Threshold violation rate $\epsilon_i$')
    ax.set_xlim(-2, 102)
    ax.set_yscale('log')
    if all_nonzero:
        ax.set_ylim(min(all_nonzero) / 1.3, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', framealpha=0.88, ncol=1, fontsize=8)

    fig.tight_layout()
    out_fp = OUT_DIR / 'fig_epsilon_percentile.pdf'
    fig.savefig(out_fp, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_fp}')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    per_dataset_results = []
    for idx, (folder, label, node_num) in enumerate(DATASETS_D, start=1):
        result = process_dataset(folder, label, node_num, idx)
        if result is not None:
            per_dataset_results.append(result)

    if per_dataset_results:
        build_table8(per_dataset_results)
        plot_epsilon_percentile(per_dataset_results)
    else:
        print("\nNo datasets processed successfully.")

    print("\nAll done. Output in:", OUT_DIR)