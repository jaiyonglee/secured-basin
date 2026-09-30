# fig_3.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_3/
#   fig3a.pdf  — UK54 network map (plot8 로직)
#   fig3b.pdf  — node17 transient path (max link angle difference)
#   fig3c.pdf  — basin stability vs energy safety (m_ES, σ_ES)
#   fig3d.pdf  — node17 basin stability view (UK54)
#   fig3e.pdf  — node17 basin safety + E bounds + contours
#   fig3f.pdf  — node17 transient

import re
import csv
import math
import os
import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base      = Path(__file__).resolve().parent
NET_ROOT   = _base.parent / "02_network" / "04_UK_54"
NET_DIR    = NET_ROOT / "network"
INIT_DIR   = NET_ROOT / "init"
CSV_PATH   = NET_ROOT / "result_raw1" / "54_analysis0_n_17_Kr_1.00.csv"
OUT_DIR    = _base / "fig_3"
OUT_DIR.mkdir(exist_ok=True)

# ── 파라미터 ────────────────────────────────────────────────────
NODE_NUM     = 54
SAMPLE_NUM   = 0
KR           = 1.00
ATTACK_NODE  = 17
T_SIM        = 50.0
N_CONTOURS   = 9
SHOW_NODE_LABELS = False  # fig3c 노드 번호 표시 여부

# 섭동 정의: 각도 2개 + 에너지 → Δω 역산
PERTURB_THETA1 =  0.43 * np.pi
PERTURB_THETA2 =  0.40 * np.pi
PERTURB_E      =  500.0
LABELS = ['A', 'B']

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

PI    = np.pi
W_MAX = 20.0

COLOR_SAFE        = '#4C9BE8'
COLOR_SAFE_DARK   = '#B22222'
COLOR_DANGER      = '#E8954C'
COLOR_DESYNC      = '#555555'
COLOR_EDANGER_LINE = '#006400'
COLOR_LIMIT       = '#F5CBA7'

# ── 온도계(컬러바) 스타일 파라미터 (fig_2.py와 동일) ────────────
THERMO_LIGHTEN_FACTOR  = 0.6    # 채워지는 영역의 명도 조절 (0=원색 그대로, 1=흰색에 가깝게)
THERMO_SATURATE_FACTOR = 0.5    # 채워지는 영역의 채도 배율 (0=완전 무채색, 1=원래 채도 그대로)
THERMO_BOUNDARY_COLOR  = None   # 경계선 색. None이면 각 구간의 본문 라인색을 그대로 사용
THERMO_BOUNDARY_LW     = 2.4    # 경계선 굵기 (본문 등에너지선과 맞추려면 2.4 권장)
THERMO_TICK_FONTSIZE   = 9      # 눈금값 폰트 크기
THERMO_TICK_COLOR      = 'black'  # 눈금값/눈금 색


def soften_color(hex_color, lighten_factor=0.6, saturate_factor=0.5):
    """명도(L)를 흰색 쪽으로, 채도(S)를 낮추는 두 조절을 함께 적용 — 온도계 배경용
    lighten_factor: 0=원래 명도, 1=흰색에 가깝게
    saturate_factor: 0=완전 무채색, 1=원래 채도 그대로"""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1.0 - l) * lighten_factor
    s = s * saturate_factor
    return colorsys.hls_to_rgb(h, l, s)


# ================================================================
# 네트워크 로더
# ================================================================
def load_network_txt(node_num, sample_num, kr):
    P = {}; mass = {}; gamma = {}
    with open(NET_DIR / f"{node_num}_sample{sample_num}_nodes.txt") as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            i = int(parts[0])
            P[i]     = float(parts[1]) if len(parts) > 1 else 0.0
            mass[i]  = float(parts[2]) if len(parts) > 2 else 1.0
            gamma[i] = float(parts[3]) if len(parts) > 3 else 0.1
    edges = []
    with open(NET_DIR / f"{node_num}_sample{sample_num}_edges.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            edges.append((int(parts[0]), int(parts[1]), float(parts[2]) * kr))
    theta_eq = np.zeros(node_num)
    with open(INIT_DIR / f"{node_num}_sample{sample_num}_init_Kr_{kr:.2f}.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                theta_eq[int(parts[0])] = float(parts[2])
    K = np.zeros((node_num, node_num))
    for u, v, w in edges:
        K[u,v] = K[v,u] = w
    return P, mass, gamma, edges, K, theta_eq


# ================================================================
# GML loader (from 4_plot.py)
# ================================================================
def parse_gml(path):
    with open(path) as f:
        content = f.read()
    nodes = {}
    for block in re.findall(r'node \[(.*?)\]', content, re.DOTALL):
        nid   = int(re.search(r'id (\d+)', block).group(1))
        pos   = re.findall(r'pos ([-\d.]+)', block)
        lon, lat = float(pos[0]), float(pos[1])
        power = float(re.search(r'power ([-\d.eE+\-]+)', block).group(1))
        mass  = float(re.search(r'mass ([-\d.eE+\-]+)',  block).group(1))
        gamma = float(re.search(r'gamma ([-\d.eE+\-]+)', block).group(1))
        nodes[nid] = dict(lon=lon, lat=lat, power=power, mass=mass, gamma=gamma)
    edges = []
    for block in re.findall(r'edge \[(.*?)\]', content, re.DOTALL):
        src = int(re.search(r'source (\d+)', block).group(1))
        tgt = int(re.search(r'target (\d+)', block).group(1))
        w   = float(re.search(r'weight ([-\d.eE+\-]+)', block).group(1))
        edges.append((src, tgt, w))
    return nodes, edges


def load_energy_bounds():
    bounds = {}
    result_dir = NET_ROOT / "result_raw3"
    kr_str = f"{KR:.2f}"
    for n in range(NODE_NUM):
        fp = result_dir / f"{NODE_NUM}_analysis{SAMPLE_NUM}_n_{n}_Kr_{kr_str}.csv"
        if not fp.exists():
            bounds[n] = {'E_safe_max': np.nan, 'E_danger_min': np.nan}
            continue
        last_row = None
        with open(fp, newline='') as f:
            for row in csv.DictReader(f):
                last_row = row
        if last_row is None:
            bounds[n] = {'E_safe_max': np.nan, 'E_danger_min': np.nan}
            continue
        try:
            esm = float(last_row.get('E_safe_max',  'nan') or 'nan')
            edm = float(last_row.get('E_danger_min','nan') or 'nan')
        except (ValueError, KeyError):
            esm = edm = np.nan
        bounds[n] = {'E_safe_max': esm, 'E_danger_min': edm}
    return bounds


# ================================================================
# fig3a — Network map (plot7 스타일, 노드색=basin_safety)
# ================================================================
def fig3a():
    RANK_CSV = NET_ROOT / "result" / "0_rank.csv"

    # GML 로드
    gml_candidates = [
        NET_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}.gml",
        NET_DIR / "uk_reduced.gml",
    ]
    gml_path = next((p for p in gml_candidates if p.exists()), None)

    if gml_path:
        nodes_gml, edges_gml = parse_gml(gml_path)
        use_geo = True
    else:
        P_d, M_d, D_d, edges_list, K, _ = load_network_txt(NODE_NUM, SAMPLE_NUM, KR)
        nodes_gml = {i: dict(lon=0, lat=0, power=P_d[i], mass=M_d[i], gamma=D_d[i])
                     for i in range(NODE_NUM)}
        edges_gml = [(u, v, w) for u, v, w in edges_list]
        use_geo = False

    # basin_safety from 0_rank.csv
    rank_df = pd.read_csv(RANK_CSV)
    bs_vals = np.full(NODE_NUM, np.nan)
    if 'basin_safety' in rank_df.columns:
        for _, row in rank_df.iterrows():
            n = int(row['node'])
            if n < NODE_NUM:
                bs_vals[n] = float(row['basin_safety'])

    bs_valid = bs_vals[np.isfinite(bs_vals)]
    bs_lo = bs_valid.min() if len(bs_valid) else 0.0
    bs_hi = bs_valid.max() if len(bs_valid) else 1.0
    if bs_lo == bs_hi: bs_lo = max(bs_hi - 0.1, 0.0)
    norm_bs = mcolors.Normalize(vmin=bs_lo, vmax=bs_hi)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_bs = LinearSegmentedColormap.from_list(
        'safety', ['#E8954C', '#4C9BE8'], N=256)  # low=주황, high=파랑

    xs = {nid: d['lon'] for nid, d in nodes_gml.items()}
    ys = {nid: d['lat'] for nid, d in nodes_gml.items()}

    x_vals = list(xs.values()); y_vals = list(ys.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    margin_x = (x_max - x_min) * 0.12
    margin_y = (y_max - y_min) * 0.12

    # rim 굵기 = |P|
    powers = np.array([nodes_gml[i]['power'] for i in range(NODE_NUM)])
    p_abs  = np.where(np.abs(powers) < 1e-6, 1e-6, np.abs(powers))
    p_lo   = max(np.percentile(p_abs, 5),  1e-6)
    p_hi   = max(np.percentile(p_abs, 95), p_lo * 10)
    log_p_lo = np.log10(p_lo); log_p_hi = np.log10(p_hi)

    def _rim_lw(p):
        t = np.clip((np.log10(max(abs(p), 1e-6)) - log_p_lo) /
                    max(log_p_hi - log_p_lo, 1e-6), 0, 1)
        return 0.4 + 4.0 * t

    NODE_SIZE = 220  # 모두 동일

    fig = plt.figure(figsize=(5.0, 4.5), facecolor='white')

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92],
                          projection=ccrs.PlateCarree())
        ax.set_extent([x_min-margin_x, x_max+margin_x,
                       y_min-margin_y, y_max+margin_y], crs=ccrs.PlateCarree())
        for feat in [
            cfeature.NaturalEarthFeature('physical','ocean','10m',
                facecolor='#c8dff0', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','land','10m',
                facecolor='#eae6df', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','coastline','10m',
                facecolor='none', edgecolor='#555', linewidth=0.8),
        ]:
            ax.add_feature(feat, zorder=0)
    except Exception:
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92])
        ax.set_facecolor('#eae6df')
        ax.set_xlim(x_min-margin_x, x_max+margin_x)
        ax.set_ylim(y_min-margin_y, y_max+margin_y)
        ax.set_xticks([]); ax.set_yticks([])

    # 1. 링크 먼저
    for src_n, tgt_n, w in edges_gml:
        ax.plot([xs[src_n], xs[tgt_n]], [ys[src_n], ys[tgt_n]],
                color='black', lw=1.2, alpha=0.5, zorder=2,
                solid_capstyle='round')

    # 2. 노드+숫자 한 쌍씩: basin_safety 높은 순 → 낮은 순 (낮은게 위에 그려짐)
    draw_order = sorted(range(NODE_NUM),
                        key=lambda n: bs_vals[n] if np.isfinite(bs_vals[n]) else float('inf'),
                        reverse=True)

    for z_base, nid in enumerate(draw_order):
        x, y = xs[nid], ys[nid]
        bs   = bs_vals[nid]
        node_color = ('#aaaaaa' if not np.isfinite(bs)
                      else cmap_bs(norm_bs(np.clip(bs, bs_lo, bs_hi))))
        z = 10 + z_base * 2
        ax.scatter(x, y, s=NODE_SIZE, c=[node_color], zorder=z,
                   edgecolors='black', linewidths=1.2, marker='o')
        ax.text(x, y, str(nid), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=z+1,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

    legend_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#4C9BE8',
               markeredgecolor='black', markeredgewidth=1.0,
               markersize=9, label='High basin safety'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#E8954C',
               markeredgecolor='black', markeredgewidth=1.0,
               markersize=9, label='Low basin safety'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=8.5, framealpha=0.88, facecolor='white', edgecolor='#aaa')

    # ── 인셋: node17 basin safety view (sync safe/danger + desync), legend 아래 ──
    ax_in = ax.inset_axes([0.65, 0.60, 0.315, 0.27])
    df17 = pd.read_csv(CSV_PATH)
    has_safe17    = 'safe' in df17.columns
    mask_sync17   = df17['num_synced'] >= NODE_NUM
    mask_desync17 = df17['num_synced'] < NODE_NUM
    if has_safe17:
        mask_safe17   = mask_sync17 & (df17['safe'] == 1)
        mask_danger17 = mask_sync17 & (df17['safe'] != 1)
    else:
        mask_safe17   = mask_sync17
        mask_danger17 = pd.Series([False]*len(df17))

    ax_in.scatter(df17.loc[mask_desync17, 'attack_theta'],
                  df17.loc[mask_desync17, 'attack_w'],
                  c=COLOR_DESYNC, s=3, linewidths=0, rasterized=True, zorder=2)
    ax_in.scatter(df17.loc[mask_danger17, 'attack_theta'],
                  df17.loc[mask_danger17, 'attack_w'],
                  c=COLOR_DANGER, s=3, linewidths=0, rasterized=True, zorder=3)
    ax_in.scatter(df17.loc[mask_safe17, 'attack_theta'],
                  df17.loc[mask_safe17, 'attack_w'],
                  c=COLOR_SAFE, s=3, linewidths=0, rasterized=True, zorder=4)

    ax_in.set_xlim(-PI, PI)
    ax_in.set_ylim(-W_MAX, W_MAX)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_facecolor('white')
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#666')

    fig.savefig(OUT_DIR / 'fig3a.pdf', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3a.pdf')


# ================================================================
# fig3h — Network map (fig3a 양식, 노드색=basin_stability)
# ================================================================
def fig3h():
    RANK_CSV = NET_ROOT / "result" / "0_rank.csv"

    gml_candidates = [
        NET_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}.gml",
        NET_DIR / "uk_reduced.gml",
    ]
    gml_path = next((p for p in gml_candidates if p.exists()), None)

    if gml_path:
        nodes_gml, edges_gml = parse_gml(gml_path)
    else:
        P_d, M_d, D_d, edges_list, K, _ = load_network_txt(NODE_NUM, SAMPLE_NUM, KR)
        nodes_gml = {i: dict(lon=0, lat=0, power=P_d[i], mass=M_d[i], gamma=D_d[i])
                     for i in range(NODE_NUM)}
        edges_gml = [(u, v, w) for u, v, w in edges_list]

    # basin_stability from 0_rank.csv
    rank_df = pd.read_csv(RANK_CSV)
    bs_vals = np.full(NODE_NUM, np.nan)
    if 'basin_stability' in rank_df.columns:
        for _, row in rank_df.iterrows():
            n = int(row['node'])
            if n < NODE_NUM:
                bs_vals[n] = float(row['basin_stability'])

    bs_valid = bs_vals[np.isfinite(bs_vals)]
    bs_lo = bs_valid.min() if len(bs_valid) else 0.0
    bs_hi = bs_valid.max() if len(bs_valid) else 1.0
    if bs_lo == bs_hi: bs_lo = max(bs_hi - 0.1, 0.0)
    norm_bs = mcolors.Normalize(vmin=bs_lo, vmax=bs_hi)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_bs = LinearSegmentedColormap.from_list(
        'stability', ['#555555', '#71bd58'], N=256)  # low=회색, high=녹색

    xs = {nid: d['lon'] for nid, d in nodes_gml.items()}
    ys = {nid: d['lat'] for nid, d in nodes_gml.items()}
    x_vals = list(xs.values()); y_vals = list(ys.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    margin_x = (x_max - x_min) * 0.12
    margin_y = (y_max - y_min) * 0.12

    # rim 굵기 = |P|
    powers = np.array([nodes_gml[i]['power'] for i in range(NODE_NUM)])
    p_abs  = np.where(np.abs(powers) < 1e-6, 1e-6, np.abs(powers))
    p_lo   = max(np.percentile(p_abs, 5),  1e-6)
    p_hi   = max(np.percentile(p_abs, 95), p_lo * 10)
    log_p_lo = np.log10(p_lo); log_p_hi = np.log10(p_hi)

    def _rim_lw(p):
        t = np.clip((np.log10(max(abs(p), 1e-6)) - log_p_lo) /
                    max(log_p_hi - log_p_lo, 1e-6), 0, 1)
        return 0.4 + 4.0 * t

    NODE_SIZE = 220

    fig = plt.figure(figsize=(5.0, 4.5), facecolor='white')

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92],
                          projection=ccrs.PlateCarree())
        ax.set_extent([x_min-margin_x, x_max+margin_x,
                       y_min-margin_y, y_max+margin_y], crs=ccrs.PlateCarree())
        for feat in [
            cfeature.NaturalEarthFeature('physical','ocean','10m',
                facecolor='#c8dff0', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','land','10m',
                facecolor='#eae6df', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','coastline','10m',
                facecolor='none', edgecolor='#555', linewidth=0.8),
        ]:
            ax.add_feature(feat, zorder=0)
    except Exception:
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92])
        ax.set_facecolor('#eae6df')
        ax.set_xlim(x_min-margin_x, x_max+margin_x)
        ax.set_ylim(y_min-margin_y, y_max+margin_y)
        ax.set_xticks([]); ax.set_yticks([])

    for src_n, tgt_n, w in edges_gml:
        ax.plot([xs[src_n], xs[tgt_n]], [ys[src_n], ys[tgt_n]],
                color='black', lw=1.2, alpha=0.5, zorder=2,
                solid_capstyle='round')

    draw_order = sorted(range(NODE_NUM),
                        key=lambda n: bs_vals[n] if np.isfinite(bs_vals[n]) else float('inf'),
                        reverse=True)

    for z_base, nid in enumerate(draw_order):
        x, y = xs[nid], ys[nid]
        bs   = bs_vals[nid]
        node_color = ('#aaaaaa' if not np.isfinite(bs)
                      else cmap_bs(norm_bs(np.clip(bs, bs_lo, bs_hi))))
        z = 10 + z_base * 2
        ax.scatter(x, y, s=NODE_SIZE, c=[node_color], zorder=z,
                   edgecolors='black', linewidths=1.2, marker='o')
        ax.text(x, y, str(nid), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=z+1,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

    legend_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#71bd58',
               markeredgecolor='black', markeredgewidth=1.0,
               markersize=9, label='High basin stability'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#555555',
               markeredgecolor='black', markeredgewidth=1.0,
               markersize=9, label='Low basin stability'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=8.5, framealpha=0.88, facecolor='white', edgecolor='#aaa')

    # ── 인셋: fig3d (node17 basin stability view), legend 바로 아래 ──────
    ax_in = ax.inset_axes([0.65, 0.60, 0.315, 0.27])
    df17 = pd.read_csv(CSV_PATH)
    mask_sync_in   = df17['num_synced'] >= NODE_NUM
    mask_desync_in = df17['num_synced'] < NODE_NUM

    ax_in.scatter(df17.loc[mask_desync_in, 'attack_theta'],
                  df17.loc[mask_desync_in, 'attack_w'],
                  c=COLOR_DESYNC, s=3, linewidths=0, rasterized=True, zorder=2)
    ax_in.scatter(df17.loc[mask_sync_in, 'attack_theta'],
                  df17.loc[mask_sync_in, 'attack_w'],
                  c='#71bd58', s=3, linewidths=0, rasterized=True, zorder=3)

    ax_in.set_xlim(-PI, PI)
    ax_in.set_ylim(-W_MAX, W_MAX)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_facecolor('white')
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#666')

    fig.savefig(OUT_DIR / 'fig3h.pdf', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3h.pdf')



# ================================================================
# 에너지 계산 및 iso-energy curve (슬라이드 수식)
# ================================================================
def potential_diff(target, theta_trial, theta_eq, P, K):
    """ΔEp(θ_i, θ_i^eq)"""
    dth = (theta_trial - theta_eq[target] + PI) % (2*PI) - PI
    V_trial = V_eq = 0.0
    for nb in range(NODE_NUM):
        if K[target, nb] > 0:
            V_trial -= K[target, nb] * np.cos(theta_trial      - theta_eq[nb])
            V_eq    -= K[target, nb] * np.cos(theta_eq[target]  - theta_eq[nb])
    V_trial -= P[target] * dth
    return V_trial - V_eq


def kinetic_diff(M_i, omega, omega_eq=0.0):
    """ΔEk = M/2 * (ω² - ω_eq²)"""
    return 0.5 * M_i * (omega**2 - omega_eq**2)


def omega_from_energy(E_perturb, dth, target, theta_eq, P, M, K):
    """Δω = sqrt(2/M * (ΔE - ΔEp))"""
    theta_trial = theta_eq[target] + dth
    dEp = potential_diff(target, theta_trial, theta_eq, P, K)
    dEk = E_perturb - dEp
    if dEk < 0:
        return None
    return np.sqrt(2.0 / M[target] * dEk)


def wrap_to_pi(x):
    return (x + PI) % (2*PI) - PI


def compute_uep_dtheta(target, theta_eq, P, K):
    """Local mean-field 방법으로 UEP의 delta-theta를 계산 (attack_random_server.c와 동일 로직).
    반환: (uep_dtheta, valid). valid=False면 계산 불가(폴백, 제한 없음)."""
    sum_cos = 0.0; sum_sin = 0.0
    for nb in range(NODE_NUM):
        if nb == target or K[target, nb] <= 0:
            continue
        sum_cos += K[target, nb] * np.cos(theta_eq[nb])
        sum_sin += K[target, nb] * np.sin(theta_eq[nb])

    R = np.hypot(sum_cos, sum_sin)
    if R < 1e-9:
        return 0.0, False
    psi = np.arctan2(sum_sin, sum_cos)
    ratio = P[target] / R
    if ratio > 1.0 or ratio < -1.0:
        return 0.0, False

    base_asin = np.arcsin(ratio)
    sol1 = psi + base_asin
    sol2 = psi + (PI - base_asin)
    cand1 = wrap_to_pi(sol1)
    cand2 = wrap_to_pi(sol2)
    d1 = abs(wrap_to_pi(cand1 - theta_eq[target]))
    d2 = abs(wrap_to_pi(cand2 - theta_eq[target]))
    uep_abs = cand2 if d1 < d2 else cand1
    uep_dtheta = wrap_to_pi(uep_abs - theta_eq[target])

    if abs(uep_dtheta) < 1e-9:
        return 0.0, False
    return uep_dtheta, True


def excluded_theta_range(uep_dtheta, uep_valid):
    """UEP 반대편(먼 쪽) 금지구간 (lo, hi)를 반환. valid=False면 None."""
    if not uep_valid:
        return None
    if uep_dtheta < 0.0:
        return (-PI, uep_dtheta)   # 이 구간은 배제
    else:
        return (uep_dtheta, PI)    # 이 구간은 배제


def iso_energy_curve(E_val, target, theta_eq, P, M, K, n_interp=720, excl_range=None):
    if np.isnan(E_val) or E_val <= 0:
        return [], []
    m_i     = M[target]
    th_fine = np.linspace(-PI, PI, n_interp, endpoint=False)
    ws_pos  = np.full(n_interp, np.nan)
    ws_neg  = np.full(n_interp, np.nan)
    for i, th in enumerate(th_fine):
        if excl_range is not None and excl_range[0] < th < excl_range[1]:
            continue  # 금지구간은 에너지 계산에서 제외
        V = potential_diff(target, theta_eq[target] + th, theta_eq, P, K)
        k = E_val - V
        if k < 0:
            continue
        w = np.sqrt(2.0 * k / m_i)
        if w <= W_MAX:
            ws_pos[i] = w
            ws_neg[i] = -w

    def to_segs(ws):
        segs = []; seg_th = []; seg_w = []
        for i in range(len(th_fine)):
            if not np.isnan(ws[i]):
                seg_th.append(th_fine[i]); seg_w.append(ws[i])
            else:
                if len(seg_th) >= 2:
                    segs.append((np.array(seg_th), np.array(seg_w)))
                seg_th = []; seg_w = []
        if len(seg_th) >= 2:
            segs.append((np.array(seg_th), np.array(seg_w)))
        return segs

    return to_segs(ws_pos), to_segs(ws_neg)


def draw_iso_curve(ax, E_val, target, theta_eq, P, M, K, color, lw=2.4, zorder=5, excl_range=None):
    segs_pos, segs_neg = iso_energy_curve(E_val, target, theta_eq, P, M, K, excl_range=excl_range)
    for th_arr, w_arr in segs_pos + segs_neg:
        ax.plot(th_arr, w_arr, color=color, lw=lw, zorder=zorder, linestyle='-')

# ================================================================
# RK4 시뮬레이터
# ================================================================
def swing_rhs(state, N, P_arr, M_arr, D_arr, K):
    theta = state[:N]; omega = state[N:]
    dtheta = omega.copy()
    domega = np.zeros(N)
    for i in range(N):
        coupling = np.sum(K[i] * np.sin(theta - theta[i]))
        domega[i] = (P_arr[i] - D_arr[i]*omega[i] + coupling) / M_arr[i]
    return np.concatenate([dtheta, domega])

def rk4_step(state, dt, N, P_arr, M_arr, D_arr, K):
    k1 = swing_rhs(state,           N, P_arr, M_arr, D_arr, K)
    k2 = swing_rhs(state + dt/2*k1, N, P_arr, M_arr, D_arr, K)
    k3 = swing_rhs(state + dt/2*k2, N, P_arr, M_arr, D_arr, K)
    k4 = swing_rhs(state + dt*k3,   N, P_arr, M_arr, D_arr, K)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def simulate(P_arr, M_arr, D_arr, K, theta_eq, dth, dom,
             perturb_node, T_end=100.0, dt=1e-3, theta_c=PI):
    N = len(P_arr)
    n_steps    = int(T_end / dt)
    save_every = max(1, n_steps // 3000)

    theta0 = theta_eq.copy(); omega0 = np.zeros(N)
    theta0[perturb_node] += dth
    omega0[perturb_node] += dom
    state = np.concatenate([theta0, omega0])

    t_list = [0.0]; max_link_list = [0.0]
    crossed = False
    t = 0.0; th_prev = theta0.copy(); th_unwrap = theta0.copy()

    for step in range(n_steps):
        state = rk4_step(state, dt, N, P_arr, M_arr, D_arr, K)
        t += dt
        th = state[:N]

        d_step = th - th_prev
        d_step = (d_step + PI) % (2*PI) - PI
        th_unwrap += d_step; th_prev = th.copy()

        max_diff = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if K[i,j] > 0:
                    diff = abs(th_unwrap[i] - th_unwrap[j])
                    max_diff = max(max_diff, diff)
        if max_diff >= theta_c and not crossed:
            crossed = True

        if step % save_every == 0:
            t_list.append(t)
            max_link_list.append(max_diff)

    synced = np.all(np.abs(state[N:]) < 0.5)
    return np.array(t_list), np.array(max_link_list), crossed, synced


# ================================================================
# fig3b — Transient path
# ================================================================




# ================================================================
# fig3g — Network map, 노드색=m_ES, 외곽선 굵기=σ_ES
# ================================================================
def fig3g():
    import glob, re as _re
    from matplotlib.colors import LinearSegmentedColormap

    # GML 로드
    gml_candidates = [
        NET_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}.gml",
        NET_DIR / "uk_reduced.gml",
    ]
    gml_path = next((p for p in gml_candidates if p.exists()), None)
    if gml_path:
        nodes_gml, edges_gml = parse_gml(gml_path)
    else:
        P_d, M_d, D_d, edges_list, _, _ = load_network_txt(NODE_NUM, SAMPLE_NUM, KR)
        nodes_gml = {i: dict(lon=0, lat=0, power=P_d[i], mass=M_d[i], gamma=D_d[i])
                     for i in range(NODE_NUM)}
        edges_gml = [(u, v, w) for u, v, w in edges_list]

    # E 값 로드 (raw1 폴더)
    raw1_dir = NET_ROOT / "result_raw1"
    kr_str   = f"{KR:.2f}"
    e_safety = np.full(NODE_NUM, np.nan)
    e_danger = np.full(NODE_NUM, np.nan)
    for fp in glob.glob(str(raw1_dir / f"*_n_*_Kr_{kr_str}.csv")):
        m = _re.search(r'_n_(\d+)_', os.path.basename(fp))
        if not m: continue
        n = int(m.group(1))
        if n >= NODE_NUM: continue
        try:
            tmp = pd.read_csv(fp, usecols=["E_safe_max", "E_danger_min"])
            last = tmp.iloc[-1]
            e_safety[n] = float(last['E_danger_min'])  # E^safety
            e_danger[n] = float(last['E_safe_max'])    # E^danger
        except Exception:
            continue

    with np.errstate(invalid='ignore', divide='ignore'):
        m_ES   = np.sqrt(e_safety * e_danger)
        sig_ES = np.sqrt(e_danger / e_safety)

    # 색: m_ES 기반 주황↔파랑 — fig3c와 동일하게 log10 정규화
    valid = np.isfinite(m_ES) & (m_ES > 0)
    log_m_all = np.log10(m_ES[valid])
    m_log_lo = log_m_all.min() if valid.any() else 0.0
    m_log_hi = log_m_all.max() if valid.any() else 1.0
    if m_log_lo == m_log_hi: m_log_lo = m_log_hi - 0.1

    def _node_color(m):
        if not (np.isfinite(m) and m > 0):
            return '#aaaaaa'
        t = np.clip((np.log10(m) - m_log_lo) / max(m_log_hi - m_log_lo, 1e-9), 0, 1)
        blue_c  = np.array([0x4C, 0x9B, 0xE8]) / 255.0
        orange_c = np.array([0xE8, 0x95, 0x4C]) / 255.0
        c = t * blue_c + (1 - t) * orange_c
        return tuple(np.clip(c, 0, 1))

    # 외곽선 굵기: σ_ES를 log 정규화 (선형이면 17번 outlier에 나머지가 다 눌림)
    sig_valid = sig_ES[np.isfinite(sig_ES) & (sig_ES > 0)]
    log_sig = np.log10(sig_valid) if len(sig_valid) else np.array([0.0, 1.0])
    log_sig_lo = log_sig.min()
    log_sig_hi = log_sig.max()
    if log_sig_lo == log_sig_hi: log_sig_lo = log_sig_hi - 0.1
    LW_BASE = 1.2
    def _rim_lw(sig):
        if not (np.isfinite(sig) and sig > 0):
            return LW_BASE
        t = np.clip((np.log10(sig) - log_sig_lo) / max(log_sig_hi - log_sig_lo, 1e-9), 0, 1)
        return LW_BASE * 0.5 + LW_BASE * 1.5 * t  # 0.6 ~ 2.4

    xs = {nid: d['lon'] for nid, d in nodes_gml.items()}
    ys = {nid: d['lat'] for nid, d in nodes_gml.items()}
    x_vals = list(xs.values()); y_vals = list(ys.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    margin_x = (x_max - x_min) * 0.12
    margin_y = (y_max - y_min) * 0.12

    NODE_SIZE = 220
    fig = plt.figure(figsize=(5.0, 4.5), facecolor='white')
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92],
                          projection=ccrs.PlateCarree())
        ax.set_extent([x_min-margin_x, x_max+margin_x,
                       y_min-margin_y, y_max+margin_y], crs=ccrs.PlateCarree())
        for feat in [
            cfeature.NaturalEarthFeature('physical','ocean','10m',
                facecolor='#c8dff0', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','land','10m',
                facecolor='#eae6df', edgecolor='none'),
            cfeature.NaturalEarthFeature('physical','coastline','10m',
                facecolor='none', edgecolor='#555', linewidth=0.8),
        ]:
            ax.add_feature(feat, zorder=0)
    except Exception:
        ax = fig.add_axes([0.04, 0.04, 0.88, 0.92])
        ax.set_facecolor('#eae6df')
        ax.set_xlim(x_min-margin_x, x_max+margin_x)
        ax.set_ylim(y_min-margin_y, y_max+margin_y)
        ax.set_xticks([]); ax.set_yticks([])

    # 링크
    for src_n, tgt_n, w in edges_gml:
        ax.plot([xs[src_n], xs[tgt_n]], [ys[src_n], ys[tgt_n]],
                color='black', lw=1.2, alpha=0.5, zorder=2, solid_capstyle='round')

    # 노드: m_ES 낮은 순 → 높은 순 (낮은게 위에)
    draw_order = sorted(range(NODE_NUM),
                        key=lambda n: m_ES[n] if np.isfinite(m_ES[n]) else float('inf'),
                        reverse=True)
    for z_base, nid in enumerate(draw_order):
        x, y = xs[nid], ys[nid]
        m   = m_ES[nid]
        sig = sig_ES[nid]
        node_color = _node_color(m_ES[nid])
        lw = _rim_lw(sig) if np.isfinite(sig) else LW_BASE
        z  = 10 + z_base * 2
        ax.scatter(x, y, s=NODE_SIZE, c=[node_color], zorder=z,
                   edgecolors='black', linewidths=lw, marker='o')
        ax.text(x, y, str(nid), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=z+1,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

    legend_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#4C9BE8', markeredgecolor='black',
               markeredgewidth=1.0, markersize=9, label=r'High $m^{\mathrm{ES}}$'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#E8954C', markeredgecolor='black',
               markeredgewidth=1.0, markersize=9, label=r'Low $m^{\mathrm{ES}}$'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#888', markeredgecolor='black',
               markeredgewidth=2.4, markersize=9, label=r'Large $\sigma^{\mathrm{ES}}$'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#888', markeredgecolor='black',
               markeredgewidth=0.6, markersize=9, label=r'Small $\sigma^{\mathrm{ES}}$'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=8.0, framealpha=0.88, facecolor='white', edgecolor='#aaa')

    fig.savefig(OUT_DIR / 'fig3g.pdf', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3g.pdf')


def fig3b():
    RANK_CSV = NET_ROOT / "result" / "0_rank.csv"
    rank_df  = pd.read_csv(RANK_CSV)
 
    bs_stab = rank_df['basin_stability'].values
    bs_safe = rank_df['basin_safety'].values
 
    mask = np.isfinite(bs_stab) & np.isfinite(bs_safe)
    x, y = bs_stab[mask], bs_safe[mask]
 
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-9)
 
    blue   = np.array([0x4C, 0x9B, 0xE8]) / 255.0
    orange = np.array([0xE8, 0x95, 0x4C]) / 255.0
 
    # safety(y)만으로 파랑↔주황 결정, stability(x)는 색에 반영 안 함
    colors = np.outer(y_norm, blue) + np.outer(1 - y_norm, orange)
    colors = np.clip(colors, 0, 1)
 
    r = np.corrcoef(x, y)[0, 1]
    r2 = r**2
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.scatter(x, y, c=colors, s=35, alpha=0.85,
               edgecolors='white', linewidths=0.5, zorder=3)
 
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='#555555', lw=1.5,
            ls='--', zorder=2, label=f'$r={r:.3f}$, $R^2={r2:.3f}$')
 
    ax.set_xlabel('basin stability', fontsize=12)
    ax.set_ylabel('basin safety',    fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3b.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: fig3b.pdf  |  r={r:.4f}  R2={r2:.4f}')


# ================================================================
# fig3c — basin stability vs energy safety (m_ES, σ_ES)
# ================================================================
def fig3c_energy():
    import glob
    RANK_CSV = NET_ROOT / "result" / "0_rank.csv"
    rank_df  = pd.read_csv(RANK_CSV)

    # raw1 폴더에서 노드별 CSV를 읽어 마지막 행의 E_safe_max, E_danger_min 수집
    raw1_dir = NET_ROOT / "result_raw1"
    kr_str   = f"{KR:.2f}"
    e_safety = np.full(NODE_NUM, np.nan)
    e_danger = np.full(NODE_NUM, np.nan)

    for fp in glob.glob(str(raw1_dir / f"*_n_*_Kr_{kr_str}.csv")):
        import re as _re
        m = _re.search(r'_n_(\d+)_', os.path.basename(fp))
        if not m:
            continue
        n = int(m.group(1))
        if n >= NODE_NUM:
            continue
        try:
            tmp = pd.read_csv(fp, usecols=["E_safe_max", "E_danger_min"])
            if 'E_safe_max' not in tmp.columns or 'E_danger_min' not in tmp.columns:
                continue
            last = tmp.iloc[-1]
            e_safety[n] = float(last['E_danger_min'])  # E^safety = min of danger outcomes
            e_danger[n] = float(last['E_safe_max'])    # E^danger = max of safe outcomes
        except Exception:
            continue

    print(f"  Loaded E values for {np.sum(np.isfinite(e_safety))} nodes")

    bs = rank_df['basin_stability'].values.astype(float) \
         if 'basin_stability' in rank_df.columns else np.full(NODE_NUM, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        m_ES = np.sqrt(e_safety * e_danger)

    mask = (np.isfinite(bs) & np.isfinite(m_ES)
            & (e_safety > 0) & (e_danger > 0))
    bs_v, m_v, es_v, ed_v = bs[mask], m_ES[mask], e_safety[mask], e_danger[mask]

    print(f"  Valid nodes: {mask.sum()}")

    # 회귀: basin_stability vs log10(m_ES)
    log_m = np.log10(m_v)
    r  = np.corrcoef(bs_v, log_m)[0, 1]
    r2 = r**2
    z  = np.polyfit(bs_v, log_m, 1)
    p  = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    # 에러바: E^safety ~ E^danger, 양끝 cap 처리
    cap_w = 0.004
    nodes_v = rank_df['node'].values[mask]
    cap_w = 0.004

    # 1. 일반 노드 에러바 먼저
    for i in range(len(bs_v)):
        if int(nodes_v[i]) == ATTACK_NODE:
            continue
        x = bs_v[i]
        ax.plot([x, x], [es_v[i], ed_v[i]], color='#aaaaaa', lw=0.8, zorder=1)
        ax.plot([x-cap_w, x+cap_w], [es_v[i], es_v[i]], color='#aaaaaa', lw=0.8, zorder=1)
        ax.plot([x-cap_w, x+cap_w], [ed_v[i], ed_v[i]], color='#aaaaaa', lw=0.8, zorder=1)

    # 2. 점 색: m_ES 낮으면 주황, 높으면 파랑
    log_m_v = np.log10(m_v)
    m_norm = (log_m_v - log_m_v.min()) / max(log_m_v.max() - log_m_v.min(), 1e-9)
    blue   = np.array([0x4C, 0x9B, 0xE8]) / 255.0
    orange = np.array([0xE8, 0x95, 0x4C]) / 255.0
    colors = np.outer(m_norm, blue) + np.outer(1 - m_norm, orange)
    colors = np.clip(colors, 0, 1)

    # 3. 일반 노드 점
    non_target = np.array([int(n) != ATTACK_NODE for n in nodes_v])
    ax.scatter(bs_v[non_target], m_v[non_target], c=colors[non_target],
               s=35, alpha=0.85, edgecolors='white', linewidths=0.5, zorder=2)

    # 4. 대상 노드 에러바
    tgt = np.where(np.array([int(n) for n in nodes_v]) == ATTACK_NODE)[0]
    if len(tgt):
        i = tgt[0]
        x = bs_v[i]
        ax.plot([x, x], [es_v[i], ed_v[i]], color='#333333', lw=0.8, zorder=3)
        ax.plot([x-cap_w, x+cap_w], [es_v[i], es_v[i]], color='#333333', lw=0.8, zorder=3)
        ax.plot([x-cap_w, x+cap_w], [ed_v[i], ed_v[i]], color='#333333', lw=0.8, zorder=3)

    # 5. 대상 노드 점 (맨 위)
    if len(tgt):
        i = tgt[0]
        ax.scatter([bs_v[i]], [m_v[i]], c=[colors[i]], s=35, alpha=0.85,
                   edgecolors='white', linewidths=0.5, zorder=4)

    if SHOW_NODE_LABELS:
        for i in range(len(bs_v)):
            ax.text(bs_v[i], m_v[i], str(int(nodes_v[i])),
                    fontsize=6, ha='left', va='bottom', color='#444', zorder=5)

    # 회귀선 (log 공간에서 선형 → 원래 공간에서 power law)
    x_line = np.linspace(bs_v.min(), bs_v.max(), 100)
    ax.plot(x_line, 10**p(x_line), color='#555555', lw=1.5,
            ls='--', zorder=4, label=f'$r={r:.3f}$, $R^2={r2:.3f}$')

    ax.set_yscale('log')
    ax.set_xlabel('basin stability', fontsize=12)
    ax.set_ylabel('energy safety (log)', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3c.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    # 대상 노드 m, sigma 출력
    tgt_print = np.where(np.array([int(n) for n in nodes_v]) == ATTACK_NODE)[0]
    if len(tgt_print):
        i_t = tgt_print[0]
        m_t   = m_v[i_t]
        sig_t = np.sqrt(ed_v[i_t] / es_v[i_t])
        print(f'  Node {ATTACK_NODE}: m_ES={m_t:.4f}  sigma_ES={sig_t:.4f}')
        print(f'    E_safety={es_v[i_t]:.4f}  E_danger={ed_v[i_t]:.4f}')
    print(f'Saved: fig3c.pdf  |  r={r:.4f}  R2={r2:.4f}')


# ================================================================
# fig3d — basin stability view (fig1c 스타일, UK54 node17)
# ================================================================
def fig3d_stability(df):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.set_facecolor('white')

    mask_sync   = df['num_synced'] >= NODE_NUM
    mask_desync = df['num_synced'] < NODE_NUM

    ax.scatter(df.loc[mask_desync, 'attack_theta'],
               df.loc[mask_desync, 'attack_w'],
               c=COLOR_DESYNC, s=6, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(df.loc[mask_sync, 'attack_theta'],
               df.loc[mask_sync, 'attack_w'],
               c='#71bd58', s=6, linewidths=0, rasterized=True, zorder=3)

    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_{17}$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_{17}$')
    legend_handles = [
        mpatches.Patch(color='#71bd58',    label='sync'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3d.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3d.pdf')

# ================================================================
# 온도계 인셋: 아래=초록(secured), 가운데=회색(gray zone), 위=빨강(unsecured)
# - 산점도 왼쪽 안쪽, 세로 90% 높이, 가운데 정렬
# - 축 범위는 3×10^n 단위 (해당 노드의 E_safety~E_danger 기준 반-decade 여백)
# - 눈금: 10의 배수(major)만 라벨 표시(그림 중앙), 2~9배수(minor)는 짧은 틱만
# - 눈금/눈금값 모두 검은색, 배경색은 완전 불투명(투명도 없음)
# - 항상 다른 요소보다 위에 그려짐 (zorder=100)
# ================================================================
def draw_thermometer(ax_th, thermo_lo, thermo_hi, E_safety, E_danger):
    ax_th.set_yscale('log')
    ax_th.set_ylim(thermo_lo, thermo_hi)
    ax_th.set_xlim(0, 1)
    ax_th.set_xticks([])

    soft_green = soften_color(COLOR_EDANGER_LINE, THERMO_LIGHTEN_FACTOR, THERMO_SATURATE_FACTOR)
    soft_gray  = soften_color(COLOR_DESYNC,       THERMO_LIGHTEN_FACTOR, THERMO_SATURATE_FACTOR)
    soft_red   = soften_color(COLOR_SAFE_DARK,    THERMO_LIGHTEN_FACTOR, THERMO_SATURATE_FACTOR)

    if np.isfinite(E_safety) and np.isfinite(E_danger) and E_danger > E_safety:
        ax_th.axhspan(thermo_lo, E_safety, color=soft_green, zorder=1)
        ax_th.axhspan(E_danger, thermo_hi, color=soft_red, zorder=1)
        # 경계값 위치: 본문의 등에너지선과 동일한 색·굵기로 강조
        boundary_color_safety = THERMO_BOUNDARY_COLOR if THERMO_BOUNDARY_COLOR is not None else COLOR_EDANGER_LINE
        boundary_color_danger = THERMO_BOUNDARY_COLOR if THERMO_BOUNDARY_COLOR is not None else COLOR_SAFE_DARK
        ax_th.axhline(E_safety, color=boundary_color_safety, lw=THERMO_BOUNDARY_LW, zorder=8)
        ax_th.axhline(E_danger, color=boundary_color_danger, lw=THERMO_BOUNDARY_LW, zorder=8)
    else:
        ax_th.axhspan(thermo_lo, thermo_hi, color=soft_gray, zorder=1)

    ax_th.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=[1.0], numticks=20))
    ax_th.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                                     numticks=20))
    ax_th.set_yticklabels([])
    ax_th.tick_params(axis='y', which='major', direction='in', length=4, width=1.2,
                      colors=THERMO_TICK_COLOR)
    ax_th.tick_params(axis='y', which='minor', direction='in', length=2, width=0.6,
                      colors=THERMO_TICK_COLOR)

    for spine in ax_th.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    tick_locs = ax_th.yaxis.get_majorticklocs()
    tick_locs = [t for t in tick_locs if thermo_lo <= t <= thermo_hi]
    for t in tick_locs:
        nn = int(round(np.log10(t)))
        ax_th.text(0.5, t, rf'$10^{{{nn}}}$', ha='center', va='center',
                  fontsize=THERMO_TICK_FONTSIZE, color=THERMO_TICK_COLOR, zorder=10,
                  transform=ax_th.get_yaxis_transform())


def thermo_range_from_values(E_safety, E_danger):
    """단일 노드의 E_safety, E_danger로부터 3×10^n 온도계 범위 계산"""
    vals = [v for v in (E_safety, E_danger) if np.isfinite(v) and v > 0]
    if not vals:
        return (3e-2, 3e1)
    lo_val, hi_val = min(vals), max(vals)
    decade_lo = int(np.floor(np.log10(lo_val)))
    decade_hi = int(np.ceil(np.log10(hi_val)))
    if decade_lo == decade_hi:
        decade_hi = decade_lo + 1
    return (3 * 10**(decade_lo - 1), 3 * 10**decade_hi)


# ================================================================
# fig3e — basin safety + E bounds + contours
# ================================================================
def fig3e(df, P, M, D, K, theta_eq, perturbations):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.set_facecolor('white')

    uep_dtheta, uep_valid = compute_uep_dtheta(ATTACK_NODE, theta_eq, P, K)
    excl_range = excluded_theta_range(uep_dtheta, uep_valid)

    has_safe    = 'safe' in df.columns
    mask_sync   = df['num_synced'] >= NODE_NUM
    mask_desync = df['num_synced'] < NODE_NUM
    mask_safe   = mask_sync & (df['safe'] == 1) if has_safe else mask_sync
    mask_danger = mask_sync & (df['safe'] != 1) if has_safe else pd.Series([False]*len(df))

    ax.scatter(df.loc[mask_desync, 'attack_theta'],
               df.loc[mask_desync, 'attack_w'],
               c=COLOR_DESYNC, s=20, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(df.loc[mask_danger, 'attack_theta'],
               df.loc[mask_danger, 'attack_w'],
               c=COLOR_DANGER, s=20, linewidths=0, rasterized=True, zorder=3)
    ax.scatter(df.loc[mask_safe, 'attack_theta'],
               df.loc[mask_safe, 'attack_w'],
               c=COLOR_SAFE, s=20, linewidths=0, rasterized=True, zorder=4)

    last = df.iloc[-1]
    E_safe_max   = float(last['E_safe_max'])   if 'E_safe_max'   in df.columns else np.nan
    E_danger_min = float(last['E_danger_min']) if 'E_danger_min' in df.columns else np.nan
    print(f"  E_safe_max   = {E_safe_max:.4f}")
    print(f"  E_danger_min = {E_danger_min:.4f}")

    corners = [(-PI, -W_MAX), (-PI, W_MAX), (PI, -W_MAX), (PI, W_MAX)]
    E_corners = [potential_diff(ATTACK_NODE,
                                theta_eq[ATTACK_NODE] + cx, theta_eq, P, K)
                 + kinetic_diff(M[ATTACK_NODE], cy)
                 for cx, cy in corners]
    E_max_corner = max(E_corners)
    for E_c in np.linspace(E_max_corner / N_CONTOURS, E_max_corner, N_CONTOURS):
        segs_p, segs_n = iso_energy_curve(E_c, ATTACK_NODE, theta_eq, P, M, K, excl_range=excl_range)
        for th_arr, w_arr in segs_p + segs_n:
            ax.plot(th_arr, w_arr, color='black', lw=0.7,
                    zorder=15, linestyle='-', alpha=0.5)

    draw_iso_curve(ax, E_safe_max, ATTACK_NODE, theta_eq, P, M, K,
                   color=COLOR_SAFE_DARK, lw=2.4, zorder=6, excl_range=excl_range)
    draw_iso_curve(ax, E_danger_min, ATTACK_NODE, theta_eq, P, M, K,
                   color=COLOR_EDANGER_LINE, lw=2.4, zorder=6, excl_range=excl_range)

    for (dth, dom), lbl in zip(perturbations, LABELS):
        ax.scatter(dth, dom, s=800, c='#D94946', zorder=17,
                   edgecolors='black', linewidths=1.5, marker='*')
        ax.text(dth, dom, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=18)

    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_{17}$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_{17}$')

    if excl_range is not None:
        ax.axvspan(excl_range[0], excl_range[1], facecolor='black', alpha=0.5,
                  zorder=25, edgecolor='none', linewidth=0)

    E_sm_str = f'{E_safe_max:.3f}'
    E_dm_str = f'{E_danger_min:.3f}'
    legend_handles = [
        mpatches.Patch(color=COLOR_SAFE,   label='sync (secured)'),
        mpatches.Patch(color=COLOR_DANGER, label='sync (unsecured)'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync (unsecured)'),
        Line2D([0],[0], color=COLOR_EDANGER_LINE, lw=2.4,
               label='$E_{17}^{\\mathrm{S}}=' + E_dm_str + '$'),
        Line2D([0],[0], color=COLOR_SAFE_DARK, lw=2.4,
               label='$E_{17}^{\\mathrm{U}}=' + E_sm_str + '$'),
    ]
    leg = ax.legend(handles=legend_handles, loc='lower right',
                    framealpha=0.9, fontsize=8.5)
    leg.set_zorder(20)

    # 온도계 인셋: 산점도 왼쪽 안쪽, 세로 90% 높이(중앙 정렬), 항상 맨 위에
    thermo_lo, thermo_hi = thermo_range_from_values(E_danger_min, E_safe_max)
    ax_th = ax.inset_axes([0.05, 0.05, 0.09, 0.90], zorder=100)
    ax_th.set_zorder(100)
    ax_th.patch.set_zorder(100)
    draw_thermometer(ax_th, thermo_lo, thermo_hi, E_danger_min, E_safe_max)

    # 사각 테두리(spine)를 온도계 인셋(zorder=100)보다도 위로 올려서 맨 위에 그려지게
    for spine in ax.spines.values():
        spine.set_zorder(150)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3e.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3e.pdf')


# ================================================================
# fig3f — transient path (max angle difference)
# ================================================================
def fig3f(P, M, D, K, theta_eq, perturbations):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    colors_d = [COLOR_SAFE, COLOR_DANGER]
    results  = []

    for (dth, dom), lbl, col in zip(perturbations, LABELS, colors_d):
        print(f"  Running {lbl}: dtheta={np.degrees(dth):.1f}deg  domega={dom:.3f} rad/s")
        t_arr, max_link, crossed, synced = simulate(
            np.array([P[i] for i in range(NODE_NUM)]),
            np.array([M[i] for i in range(NODE_NUM)]),
            np.array([D[i] for i in range(NODE_NUM)]),
            K, theta_eq, dth, dom, ATTACK_NODE, T_end=T_SIM)
        status = 'sync (secured)'   if (synced and not crossed) else \
                 'sync (unsecured)' if crossed else 'desync'
        results.append((lbl, t_arr, max_link, crossed, status, col))
        print(f"    -> {status}  crossed={crossed}  synced={synced}")

    ylim_top = max(max(r[2].max() for r in results) * 1.1, PI * 1.3)
    ylim_top = max(ylim_top, PI + 1.0)

    ax.fill_between([0, T_SIM], PI, ylim_top,
                    color=COLOR_LIMIT, alpha=0.40, zorder=0)

    for lbl, t_arr, max_link, crossed, status, col in results:
        ax.plot(t_arr, max_link, color=col, lw=2.2, label=f'{lbl}: {status}')

    ax.axhline(PI, color='black', lw=1.5, ls='--', zorder=4,
               label=r'angle limit $\pi$')

    ax.set_xlim(0, T_SIM)
    ax.set_ylim(0, ylim_top)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    pi_max   = int(np.ceil(ylim_top / PI))
    pi_ticks = [k*PI for k in range(pi_max+1) if k*PI <= ylim_top*1.01]
    pi_labels = []
    for k in range(pi_max+1):
        if k*PI > ylim_top*1.01: break
        if k == 0:   pi_labels.append('0')
        elif k == 1: pi_labels.append(r'$\pi$')
        else:        pi_labels.append(f'${k}\\pi$')
    ax.set_yticks(pi_ticks)
    ax.set_yticklabels(pi_labels)

    ax.set_xlabel('time (sec)')
    ax.set_ylabel('max angle difference')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3f.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig3f.pdf')
# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("--- fig3a (network map) ---")
    fig3a()

    print("\n--- fig3h (network map, basin stability) ---")
    fig3h()

    print("\nLoading network for simulation...")
    P_d, M_d, D_d, edges_list, K, theta_eq = load_network_txt(
        NODE_NUM, SAMPLE_NUM, KR)

    print("\nLoading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # 섭동주파수 역산: Δω = sqrt(2/M * (E_perturb - ΔEp))
    perturbations = []
    for dth, lbl in zip([PERTURB_THETA1, PERTURB_THETA2], LABELS):
        dom = omega_from_energy(PERTURB_E, dth, ATTACK_NODE, theta_eq, P_d, M_d, K)
        if dom is None:
            raise ValueError(f"PERTURB_E={PERTURB_E} too small for theta={np.degrees(dth):.1f}deg")
        perturbations.append((dth, dom))
        print(f"  {lbl}: Δθ={np.degrees(dth):.1f}°  Δω={dom:.4f} rad/s")

    print("\n--- fig3c (basin stability vs energy safety) ---")
    fig3c_energy()

    print("\n--- fig3d (basin stability view) ---")
    fig3d_stability(df)

    print("\n--- fig3e (basin safety + E bounds) ---")
    fig3e(df, P_d, M_d, D_d, K, theta_eq, perturbations)

    print("\n--- fig3f (transient) ---")
    fig3f(P_d, M_d, D_d, K, theta_eq, perturbations)

    print("\n--- fig3g (network map, energy safety) ---")
    fig3g()

    print("\n--- fig3b (basin_stability vs basin_safety) ---")
    fig3b()

    print("\nAll done. Output in:", OUT_DIR)