# fig_6.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_6/
#   fig6a.pdf  — Basin stability vs (topology | dynamics) metrics, correlation heatmap
#   fig6b.pdf  — Basin safety vs ...
#   fig6c.pdf  — Energy safety mean (m_ES, log) vs ...
#   fig6d.pdf  — Energy safety deviation (sigma_ES, log) vs ...

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base    = Path(__file__).resolve().parent
NET_ROOT = _base.parent / "02_network"
OUT_DIR  = _base / "fig_6"
OUT_DIR.mkdir(exist_ok=True)

# ── 데이터셋 정의 (label, folder, enabled) ───────────────────────
DATASETS = [
    ("UK 54",      "04_UK_54",           True),
    ("Swiss 161",  "06_Swiss_161",       True),
    ("Korea 193",  "02_South_Korea_193", True),
    ("UK 313",     "05_UK_313",          True),
    ("Germany 553","07_Germany_553",     True),
    ("Spain 838",  "08_Spain_838",       True),
    ("France 847", "09_France_847",      True),
]

# ── 파라미터 ────────────────────────────────────────────────────
LOG_ENERGY_CORR = True   # energy_safety_mean / energy_safety_dev 상관계수 계산시 log10 변환 여부
SAMPLE_NUM = 0
KR = 1.00

# energy_safety_mean / energy_safety_dev 정의 방식
#   'geometric' : m_ES = sqrt(E_safety*E_danger), sigma_ES = sqrt(E_danger/E_safety)  (기존, 기하평균/곱셈형 편차)
#   'arithmetic': mean  = (E_safety+E_danger)/2,  dev     = (E_danger-E_safety)/2    (산술평균/차이평균)
# 0_rank.csv에는 기하 버전(m_ES, sigma_ES)만 저장되어 있으므로,
# E_safety = m_ES/sigma_ES, E_danger = m_ES*sigma_ES 로 역산한 뒤 원하는 방식으로 재계산한다.
ENERGY_SUMMARY_MODE = 'geometric'   # 'geometric' 또는 'arithmetic'

# ── 스타일 ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        11,
    'axes.linewidth':   1.2,
    'axes.labelsize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'mathtext.fontset': 'stix',
})

# ── 지표 정의 (이니셜 표기) ───────────────────────────────────────
# 거리(distance) 기반 지표(BC, CC)는 결합강도 K_ij의 역수(K_ij^{-1})를
# 거리로 사용 — 강한 결합 = 짧은 거리로 정의해야 weighted degree,
# eigenvector centrality 등 다른 지표와 방향성이 일관됨.
TOPOLOGY_METRICS = [
    ('weighted_degree',       '$k_w$'),
    ('betweenness',           r'BC ($K^{-1}$)'),
    ('clustering',            '$C$'),
    ('fiedler_abs',           r'$|v_2|$'),
]

DYNAMICS_METRICS = [
    ('P', '$P_i$'),
    ('M', '$M_i$'),
    ('D', '$D_i$'),
]

TARGETS = [
    ('basin_stability',     False,           'fig6a'),
    ('basin_safety',        False,           'fig6b'),
    ('energy_safety_mean',  LOG_ENERGY_CORR, 'fig6c'),
    ('energy_safety_dev',   LOG_ENERGY_CORR, 'fig6d'),
]


# ================================================================
# 네트워크 로더 + 지표 계산
# ================================================================
def _node_count(net_dir):
    candidates = list(net_dir.glob(f"*_sample{SAMPLE_NUM}_nodes.txt"))
    if not candidates:
        raise FileNotFoundError(f"No nodes file in {net_dir}")
    return candidates[0].name.split('_')[0]


def load_network_metrics(folder):
    net_dir = NET_ROOT / folder / "network"
    prefix  = _node_count(net_dir)
    nodes_fp = net_dir / f"{prefix}_sample{SAMPLE_NUM}_nodes.txt"
    edges_fp = net_dir / f"{prefix}_sample{SAMPLE_NUM}_edges.txt"

    P = {}; M = {}; D = {}
    with open(nodes_fp) as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            i = int(parts[0])
            P[i] = float(parts[1]) if len(parts) > 1 else 0.0
            M[i] = float(parts[2]) if len(parts) > 2 else 1.0
            D[i] = float(parts[3]) if len(parts) > 3 else 0.1

    N = len(P)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    with open(edges_fp) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2]) * KR
            if w > 0:
                G.add_edge(u, v, weight=w, inv_weight=1.0/w)
            else:
                G.add_edge(u, v, weight=w, inv_weight=np.inf)

    # ── 토폴로지 지표 ────────────────────────────────────────────
    weighted_degree = dict(G.degree(weight='weight'))
    # 거리 = 1/K_ij : 강한 결합일수록 가까움 (다른 지표와 방향성 일관)
    betweenness     = nx.betweenness_centrality(G, weight='inv_weight')
    clustering      = nx.clustering(G)

    # Fiedler vector (weighted Laplacian)
    L = nx.laplacian_matrix(G, weight='weight').toarray().astype(float)
    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    fiedler_vec = eigvecs[:, order[1]]
    fiedler_abs = {i: abs(fiedler_vec[i]) for i in range(N)}

    df = pd.DataFrame({
        'node':            range(N),
        'weighted_degree': [weighted_degree.get(i, np.nan) for i in range(N)],
        'betweenness':     [betweenness.get(i, np.nan) for i in range(N)],
        'clustering':      [clustering.get(i, np.nan) for i in range(N)],
        'fiedler_abs':     [fiedler_abs.get(i, np.nan) for i in range(N)],
        'P':               [P.get(i, np.nan) for i in range(N)],
        'M':               [M.get(i, np.nan) for i in range(N)],
        'D':               [D.get(i, np.nan) for i in range(N)],
    })
    return df


# ================================================================
# 데이터 로드 + 상관계수 계산
# ================================================================
def build_corr_matrices():
    all_metrics = [m for m, _ in TOPOLOGY_METRICS] + [m for m, _ in DYNAMICS_METRICS]
    results = {tcol: [] for tcol, _, _ in TARGETS}
    labels  = []

    for label, folder, enabled in DATASETS:
        if not enabled:
            continue
        print(f"  Processing {label} ...")
        rank_fp = NET_ROOT / folder / "result" / "0_rank.csv"
        if not rank_fp.exists():
            print(f"    WARNING: {rank_fp} not found, skip")
            continue
        rank_df = pd.read_csv(rank_fp)

        net_df = load_network_metrics(folder)
        merged = pd.merge(rank_df, net_df, on='node', how='inner')

        # 산술 버전 요청 시: 저장된 기하평균(m_ES)/편차(sigma_ES)로부터
        # E_safety, E_danger를 역산한 뒤 산술평균/차이평균으로 재계산
        if ENERGY_SUMMARY_MODE == 'arithmetic':
            m_es  = merged['energy_safety_mean'].values.astype(float)
            sig_es = merged['energy_safety_dev'].values.astype(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                E_safety = m_es / sig_es
                E_danger = m_es * sig_es
            merged['energy_safety_mean'] = (E_safety + E_danger) / 2.0
            merged['energy_safety_dev']  = (E_danger - E_safety) / 2.0

        labels.append(label)

        for tcol, use_log, _ in TARGETS:
            row = []
            y = merged[tcol].values.astype(float)
            if use_log:
                y_calc = np.where(y > 0, np.log10(np.where(y > 0, y, 1)), np.nan)
            else:
                y_calc = y

            for mcol in all_metrics:
                x = merged[mcol].values.astype(float)
                mask = np.isfinite(x) & np.isfinite(y_calc)
                if mask.sum() > 2 and np.nanstd(x[mask]) > 0 and np.nanstd(y_calc[mask]) > 0:
                    r = np.corrcoef(x[mask], y_calc[mask])[0, 1]
                else:
                    r = np.nan
                row.append(r)
            results[tcol].append(row)

    out = {}
    for tcol, _, _ in TARGETS:
        out[tcol] = np.array(results[tcol])
    return out, labels


# ================================================================
# 히트맵 플랏 (토폴로지 + 다이나믹스 하나로 합치고, 경계 강조)
# ================================================================
def plot_heatmap(arr, labels, out_name):
    n_topo = len(TOPOLOGY_METRICS)
    n_dyn  = len(DYNAMICS_METRICS)
    n_rows = len(labels)
    col_labels = [lbl for _, lbl in TOPOLOGY_METRICS] + [lbl for _, lbl in DYNAMICS_METRICS]

    fig_w = 1.6 + 0.62 * (n_topo + n_dyn)
    fig_h = 0.6 + 0.50 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap('RdBu_r')
    ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                continue
            txt_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=10, color=txt_color)

    # 셀 구분 그리드 (얇은 회색선)
    ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{out_name}.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_name}.pdf')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print(f"Energy summary mode: {ENERGY_SUMMARY_MODE}")
    print("Building correlation matrices...")
    results, labels = build_corr_matrices()
    print(f"  Datasets included: {labels}")

    for tcol, _, out_name in TARGETS:
        print(f"\n--- {out_name} ({tcol}) ---")
        plot_heatmap(results[tcol], labels, out_name)

    print("\n--- x-axis abbreviations ---")
    print("[Topology]")
    print("  k_w   : Weighted degree (sum of K_ij)")
    print("  BC    : Betweenness centrality (distance = 1/K_ij)")
    print("  C     : Clustering coefficient")
    print("  |v_2| : Fiedler vector component (2nd-smallest eigval of weighted Laplacian)")
    print("[Dynamics]")
    print("  P_i   : Power injection/consumption")
    print("  M_i   : Inertia (mass)")
    print("  D_i   : Damping coefficient")

    print("\nAll done. Output in:", OUT_DIR)