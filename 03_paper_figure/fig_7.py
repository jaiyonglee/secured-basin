# fig_7.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_7/
#   fig7a.pdf  — 균등탐색 (raw1) basin safety + E bounds + contours
#   fig7b.pdf  — 집중탐색 (raw3) basin safety + E bounds + contours
#   fig7c.pdf  — E_safe_max / E_danger_min 추정값의 반복에 따른 수렴 오차 (단일 네트워크)
#   fig7d.pdf  — 7개 실제 전력망 비교: E_safety(좌)/E_danger(우) 수렴오차,
#                legend는 노드개수 오름차순

import numpy as np
import os
import glob
import re as _re
import colorsys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base    = Path(__file__).resolve().parent
NET_ROOT = _base.parent / "02_network" / "01_motif_3"
# NET_ROOT = _base.parent / "02_network" / "04_UK_54"
NET_DIR  = NET_ROOT / "network"
INIT_DIR = NET_ROOT / "init"
CSV_RAW1 = NET_ROOT / "result_raw1" / "3_analysis0_n_0_Kr_10.00.csv"
CSV_RAW3 = NET_ROOT / "result_raw3" / "3_analysis0_n_0_Kr_10.00.csv"
# CSV_RAW1 = NET_ROOT / "result_raw1" / "54_analysis0_n_17_Kr_1.00.csv"
# CSV_RAW3 = NET_ROOT / "result_raw3" / "54_analysis0_n_17_Kr_1.00.csv"
OUT_DIR  = _base / "fig_7"
OUT_DIR.mkdir(exist_ok=True)

# ── 파라미터 ────────────────────────────────────────────────────
NODE_NUM    = 3
# NODE_NUM    = 54
SAMPLE_NUM  = 0
KR          = 10.00
# KR          = 1.00
ATTACK_NODE = 0
# ATTACK_NODE = 17
N_CONTOURS  = 12
MAX_ITER    = 1000   # 7c x축 표시 범위 (참고값은 각 노드 전체 데이터의 최종행 사용)

# ── collect_node_files 필터 옵션 (motif_3처럼 analysis/Kr이 여러 개인 경우) ─────
ALL_ANALYSIS = False   # True면 모든 analysis 샘플(0,1,2,...) 사용, False면 analysis0만
ALL_KR       = False   # True면 모든 Kr값 사용, False면 DEFAULT_KR 하나만
DEFAULT_KR   = 1.00    # ALL_KR=False일 때 사용할 기본 Kr 값 (motif_3 외 나머지 네트워크)
DEFAULT_KR_MAP = {
    "01_motif_3": 10.00,   # motif_3는 Kr=10.00이 기본값
}   # 여기 없는 네트워크는 DEFAULT_KR 사용

# ── fig7d 전용 파라미터 ──────────────────────────────────────────
NODE_SKIP = 1        # fig7d: 네트워크당 이 값만큼 건너뛰며 노드를 처리 (1=전체, 5=5개당 1개)
STEP_SKIP = 1        # fig7d: 반복(iteration) 축을 이 간격으로 서브샘플링 (1=전체, 50=50스텝당 1개)
PLOT_ITER = 1000  # fig7d 그림의 x축 표시 범위(오른쪽 끝). 계산 자체는 MAX_ITER까지 그대로 함.
                      # MAX_ITER보다 작게 주면(예: MAX_ITER=1000, PLOT_ITER=200)
                      # 그림만 앞부분을 확대해서 보여준다. LaTeX 표는 영향 없음.
PLOT_ITER_MIN = 0  # fig7d 그림의 x축 표시 범위(왼쪽 끝). None이면 자동(데이터 시작점).
                      # 값을 주면(예: 50) 그 이전 구간은 그림에서 잘라서 보여준다.
TABLE_THRESHOLDS = [0.05, 0.03, 0.02, 0.01]   # LaTeX 표: NMAE가 이 값들 미만이 되는 첫 iteration을 구함 (10%, 1%)
CACHE_DIR = _base / "fig_7" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── fig7d 전용 파라미터 ────────────────────────────────────────
NODE_SUBSAMPLE_STRIDE = 1   # N개 노드당 1개씩만 처리 (1=전체, 5=5개당 1개씩 건너뛰며 대강 미리보기)
CACHE_DIR = Path(__file__).resolve().parent / "fig_7" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# fig7d — 8개 네트워크 비교: 네트워크별 서브플랏(4x2)로 수렴오차 표시
#         (motif_3 + 첨부 이미지 폴더 목록 중 Denmark, copy2 제외한 7개 실제 전력망)
#         정렬은 노드개수 오름차순, 색은 fig7c와 동일(초록=E_safety, 빨강=E_danger)
# ================================================================
DATASETS_D = [
    ("01_motif_3",         "Chain network 3", 3),
    ("04_UK_54",           "UK 54",       54),
    ("06_Swiss_161",       "Swiss 161",   161),
    ("02_South_Korea_193", "Korea 193",   193),
    ("05_UK_313",          "UK 313",      313),
    ("07_Germany_553",     "Germany 553", 553),
    ("08_Spain_838",       "Spain 838",   838),
    ("09_France_847",      "France 847",  847),
]

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
def load_network():
    P = np.zeros(NODE_NUM); M = np.ones(NODE_NUM); D = np.full(NODE_NUM, 0.1)
    with open(NET_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}_nodes.txt") as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            i = int(parts[0])
            if len(parts) > 1: P[i] = float(parts[1])
            if len(parts) > 2: M[i] = float(parts[2])
            if len(parts) > 3: D[i] = float(parts[3])
    K = np.zeros((NODE_NUM, NODE_NUM))
    with open(NET_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}_edges.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2]) * KR
            K[u,v] = K[v,u] = w
    theta_eq = np.zeros(NODE_NUM)
    with open(INIT_DIR / f"{NODE_NUM}_sample{SAMPLE_NUM}_init_Kr_{KR:.2f}.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                theta_eq[int(parts[0])] = float(parts[2])
    return P, M, D, K, theta_eq


# ================================================================
# 에너지 계산 및 iso-energy curve (fig_3.py 동일 로직)
# ================================================================
def potential_diff(target, theta_trial, theta_eq, P, K):
    dth = (theta_trial - theta_eq[target] + PI) % (2*PI) - PI
    V_trial = V_eq = 0.0
    for nb in range(NODE_NUM):
        if K[target, nb] > 0:
            V_trial -= K[target, nb] * np.cos(theta_trial     - theta_eq[nb])
            V_eq    -= K[target, nb] * np.cos(theta_eq[target] - theta_eq[nb])
    V_trial -= P[target] * dth
    return V_trial - V_eq


def kinetic_diff(M_i, omega, omega_eq=0.0):
    return 0.5 * M_i * (omega**2 - omega_eq**2)


def wrap_to_pi(x):
    return (x + PI) % (2*PI) - PI


def compute_uep_dtheta(target, theta_eq, P, K, node_count=None):
    """Local mean-field 방법으로 UEP의 delta-theta를 계산 (fig_2.py / attack_random_server.c와 동일 로직).
    반환: (uep_dtheta, valid). valid=False면 계산 불가(배제구간 없음)."""
    if node_count is None:
        node_count = K.shape[0]
    sum_cos = 0.0; sum_sin = 0.0
    for nb in range(node_count):
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
        return (-PI, uep_dtheta)
    else:
        return (uep_dtheta, PI)


def iso_energy_curve(E_val, target, theta_eq, P, M, K, n_interp=720, excl_range=None):
    if np.isnan(E_val) or E_val <= 0:
        return [], []
    m_i     = M[target]
    th_fine = np.linspace(-PI, PI, n_interp, endpoint=False)
    ws_pos  = np.full(n_interp, np.nan)
    ws_neg  = np.full(n_interp, np.nan)
    for i, th in enumerate(th_fine):
        if excl_range is not None and excl_range[0] < th < excl_range[1]:
            continue  # 배제구간은 에너지 계산에서 제외
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
# fig7a / fig7b — basin safety + E bounds + contours (fig3 로직)
# ================================================================
def basin_safety_plot(df, P, M, D, K, theta_eq, out_name):
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

    # E bounds (최종행)
    last = df.iloc[-1]
    E_safe_max   = float(last['E_safe_max'])   if 'E_safe_max'   in df.columns else np.nan
    E_danger_min = float(last['E_danger_min']) if 'E_danger_min' in df.columns else np.nan

    # 에너지 등고선 (4모서리 최대 기준 N_CONTOURS등분)
    corners = [(-PI, -W_MAX), (-PI, W_MAX), (PI, -W_MAX), (PI, W_MAX)]
    E_corners = [potential_diff(ATTACK_NODE,
                                theta_eq[ATTACK_NODE] + cx, theta_eq, P, K)
                 + kinetic_diff(M[ATTACK_NODE], cy)
                 for cx, cy in corners]
    E_max_corner = max(E_corners)
    for E_c in np.linspace(E_max_corner / N_CONTOURS, E_max_corner, N_CONTOURS):
        segs_p, segs_n = iso_energy_curve(E_c, ATTACK_NODE, theta_eq, P, M, K,
                                          excl_range=excl_range)
        for th_arr, w_arr in segs_p + segs_n:
            ax.plot(th_arr, w_arr, color='black', lw=0.7,
                    zorder=3, linestyle='-', alpha=0.5)

    draw_iso_curve(ax, E_safe_max, ATTACK_NODE, theta_eq, P, M, K,
                   color=COLOR_SAFE_DARK, lw=2.4, zorder=6, excl_range=excl_range)
    draw_iso_curve(ax, E_danger_min, ATTACK_NODE, theta_eq, P, M, K,
                   color=COLOR_EDANGER_LINE, lw=2.4, zorder=6, excl_range=excl_range)

    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_' + f'{ATTACK_NODE}$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_' + f'{ATTACK_NODE}$')

    # UEP 배제구간 쉐이딩(zorder=25) + 사각 테두리(spine)를 온도계 인셋(100)보다도
    # 위로 올려서 맨 위에 그려지게 (fig_2.py와 동일 zorder 체계)
    if excl_range is not None:
        ax.axvspan(excl_range[0], excl_range[1], facecolor='black', alpha=0.5,
                  zorder=25, edgecolor='none', linewidth=0)
    for spine in ax.spines.values():
        spine.set_zorder(150)

    legend_handles = [
        mpatches.Patch(color=COLOR_SAFE,   label='sync (secured)'),
        mpatches.Patch(color=COLOR_DANGER, label='sync (unsecured)'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync (unsecured)'),
        Line2D([0],[0], color=COLOR_EDANGER_LINE, lw=2.4,
               label=r'$E_' + f'{ATTACK_NODE}' + '^{\mathrm{S}}=' + f'{E_danger_min:.3f}$'),
        Line2D([0],[0], color=COLOR_SAFE_DARK, lw=2.4,
               label=r'$E_' + f'{ATTACK_NODE}' + '^{\mathrm{U}}=' + f'{E_safe_max:.3f}$'),
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

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{out_name}.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_name}.pdf  |  E_safe_max={E_safe_max:.4f}  E_danger_min={E_danger_min:.4f}')


# ================================================================
# fig7c — convergence error (전체 노드 집계, absolute mean ± std)
# ================================================================

def _running_max_ignore_nan(vals):
    """NaN을 무시하며 누적 최댓값을 계산 (지금까지 본 유효값 중 최댓값을 유지)."""
    out = np.full(len(vals), np.nan)
    cur = np.nan
    for i, v in enumerate(vals):
        if not np.isnan(v):
            cur = v if np.isnan(cur) else max(cur, v)
        out[i] = cur
    return out


def _running_min_ignore_nan(vals):
    """NaN을 무시하며 누적 최솟값을 계산."""
    out = np.full(len(vals), np.nan)
    cur = np.nan
    for i, v in enumerate(vals):
        if not np.isnan(v):
            cur = v if np.isnan(cur) else min(cur, v)
        out[i] = cur
    return out


def load_energy_series(file_path, node_count=None, excl_range=None):
    """섭동 순서대로 E_safe_max/E_danger_min에 해당하는 누적 시계열을 계산.

    - anchor 단계(phase=='anchor')는 제외한다. 'phase' 컬럼이 있으면 그걸로
      판정하고(구버전 5-anchor, 신버전 3-anchor 스킴 모두 자동 대응), 컬럼이
      없는 아주 오래된 CSV는 앞 4행을 건너뛰는 옛 방식으로 폴백한다.
    - excl_range가 주어지면 그 구간에 attack_theta가 속하는 행(샘플)은 버린 뒤
      남은 행들을 순서대로 다시 인덱싱한다(= 유효 섭동 개수 기준 수렴 곡선).
      두 탐색 방식(균등/집중) 모두 동일 기준으로 필터링해야 성능 비교가
      공정해지므로, 호출하는 쪽에서 두 방식 모두 이 필터를 적용한다.
    - E_safe_max/E_danger_min은 CSV의 기존 누적 컬럼을 그대로 믿지 않고,
      필터링된 행들의 E_temp로부터 직접 누적 최대/최소를 재계산한다.
      (구버전 데이터는 배제구간 샘플이 섞여 처리된 뒤의 누적값이라, 단순히
      행만 걸러내면 이전 배제구간 샘플의 영향이 여전히 남아있을 수 있다.
      E_temp를 남은 행들만으로 처음부터 다시 누적해야 정확하다.)"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if 'phase' in df.columns:
        df = df.loc[df['phase'] != 'anchor'].reset_index(drop=True)
    else:
        df = df.iloc[4:].reset_index(drop=True)  # 구버전 폴백 (고정 5-anchor 가정)

    if excl_range is not None and 'attack_theta' in df.columns:
        th = pd.to_numeric(df['attack_theta'], errors='coerce').values
        mask_excl = (th > excl_range[0]) & (th < excl_range[1])
        df = df.loc[~mask_excl].reset_index(drop=True)

    if 'E_temp' in df.columns and 'num_synced' in df.columns:
        e_temp = pd.to_numeric(df['E_temp'], errors='coerce').values.astype(float)
        num_synced = pd.to_numeric(df['num_synced'], errors='coerce').values
        n_ref = node_count if node_count is not None else np.nanmax(num_synced)
        mask_sync = num_synced >= n_ref
        if 'safe' in df.columns:
            mask_safe_ok = mask_sync & (df['safe'].values == 1)
        else:
            mask_safe_ok = mask_sync

        safe_masked   = np.where(mask_safe_ok,  e_temp, np.nan)
        danger_masked = np.where(~mask_safe_ok, e_temp, np.nan)
        safe   = _running_max_ignore_nan(safe_masked)
        danger = _running_min_ignore_nan(danger_masked)
    else:
        # E_temp/num_synced가 없는 아주 오래된 CSV는 기존 누적 컬럼으로 폴백
        safe   = pd.to_numeric(df["E_safe_max"],   errors="coerce").values.astype(float)
        danger = pd.to_numeric(df["E_danger_min"], errors="coerce").values.astype(float)

    return safe, danger


def load_rank_uep(net_root):
    """result/0_rank.csv를 node 인덱스로 읽어 반환 (uep_dtheta/uep_valid,
    energy_safety_mean/dev 조회용). 없거나 읽기 실패하면 None(폴백 동작)."""
    fp = net_root / "result" / "0_rank.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if 'node' not in df.columns:
        return None
    return df.set_index('node')


def node_uep_and_ref(rank_df, node):
    """0_rank.csv(node 인덱스)에서 해당 노드의 excl_range와 (e_danger, e_safety)
    참고값을 구한다.
      e_danger = max(safe-outcome E_temp) 역할 (CSV 'E_safe_max' 계열, 논문 E^danger)
      e_safety = min(danger-outcome E_temp) 역할 (CSV 'E_danger_min' 계열, 논문 E^safety)
    (fig_preprocess.py의 energy_safety_mean=m_ES, energy_safety_dev=sig_ES로부터 역산:
     e_safety = m_ES/sig_ES, e_danger = m_ES*sig_ES — fig_5.py와 동일 공식)
    정보가 없으면 (None, (nan, nan))."""
    if rank_df is None or node not in rank_df.index:
        return None, (np.nan, np.nan)
    row = rank_df.loc[node]
    uep_dtheta = row['uep_dtheta'] if 'uep_dtheta' in row else np.nan
    uep_valid  = bool(row['uep_valid']) if 'uep_valid' in row else False
    excl_range = (excluded_theta_range(float(uep_dtheta), True)
                  if uep_valid and np.isfinite(uep_dtheta) else None)
    m   = row['energy_safety_mean'] if 'energy_safety_mean' in row else np.nan
    sig = row['energy_safety_dev']  if 'energy_safety_dev'  in row else np.nan
    if np.isfinite(m) and np.isfinite(sig) and sig > 0:
        e_safety = float(m) / float(sig)
        e_danger = float(m) * float(sig)
    else:
        e_safety = e_danger = np.nan
    return excl_range, (e_danger, e_safety)


def pad(arrays, length):
    out = np.full((len(arrays), length), np.nan)
    for i, a in enumerate(arrays):
        n = min(len(a), length)
        out[i, :n] = a[:n]
    return out


def compute_stats(arr):
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.full(arr.shape[1], np.nan)
    std  = np.full(arr.shape[1], np.nan)
    valid_cols = n > 0
    if valid_cols.any():
        mean[valid_cols] = np.nanmean(np.abs(arr[:, valid_cols]), axis=0)
        std[valid_cols]  = np.nanstd(np.abs(arr[:, valid_cols]), axis=0)
    return mean, std


def set_log_ylim_from_last(ax, series_list, margin_factor=0.5, x=None, x_limit=None):
    """로그축 하한을, 그림에 실제로 표시되는 범위(x_limit 이내) 안에서 각
    곡선의 마지막(수렴) 유효값 중 최솟값을 기준으로 자동 설정한다
    (margin_factor를 곱해 살짝 여유를 둠). x/x_limit을 주지 않으면 시리즈
    전체(끝까지)를 기준으로 한다 — 이러면 PLOT_ITER로 그림을 확대했을 때도
    보이지 않는 뒷부분의 값 때문에 y축이 잘못 잡힐 수 있다."""
    last_vals = []
    for arr in series_list:
        if x is not None and x_limit is not None:
            mask = x <= x_limit
            arr = arr[mask]
        v = arr[~np.isnan(arr)]
        v = v[v > 0]
        if len(v):
            last_vals.append(v[-1])
    if last_vals:
        ax.set_ylim(bottom=min(last_vals) * margin_factor)


def collect_node_files(net_root=None):
    """result_raw1, result_raw3 폴더에서 csv 매칭.
    ALL_ANALYSIS=False면 analysis0만, ALL_KR=False면 이 네트워크의 기본 Kr
    (DEFAULT_KR_MAP에 있으면 그 값, 없으면 DEFAULT_KR) 값만 사용.
    True로 두면 해당 축을 전부 포함하고, 조합별로 독립된 항목으로 취급한다
    (다른 네트워크처럼 analysis/Kr이 하나뿐이면 옵션과 무관하게 기존과 동일하게 동작)."""
    if net_root is None:
        net_root = NET_ROOT
    target_kr = DEFAULT_KR_MAP.get(net_root.name, DEFAULT_KR)
    files2 = sorted(glob.glob(str(net_root / "result_raw1" / "*.csv")))
    files3 = sorted(glob.glob(str(net_root / "result_raw3" / "*.csv")))

    def file_key(fp):
        name = os.path.basename(fp)
        m = _re.search(r'analysis(\d+)_n_(\d+)_Kr_(\d+\.\d+)', name)
        if not m:
            return None
        sample = int(m.group(1))
        node   = int(m.group(2))
        kr     = float(m.group(3))
        if not ALL_ANALYSIS and sample != 0:
            return None
        if not ALL_KR and abs(kr - target_kr) > 1e-9:
            return None
        key = (node,)
        if ALL_ANALYSIS:
            key = (sample,) + key
        if ALL_KR:
            key = key + (kr,)
        return key

    map2 = {}
    for f in files2:
        k = file_key(f)
        if k is not None:
            map2[k] = f
    map3 = {}
    for f in files3:
        k = file_key(f)
        if k is not None:
            map3[k] = f

    common = sorted(set(map2) & set(map3))
    # file_key가 만드는 key는 항상 (sample?, node, kr?) 순서이므로, node 위치는
    # ALL_ANALYSIS일 때만 sample 다음(인덱스 1)이고 그 외엔 인덱스 0.
    node_idx = 1 if ALL_ANALYSIS else 0
    return [(k[node_idx], map2[k], map3[k]) for k in common]


def fig7c():
    rank_df = load_rank_uep(NET_ROOT)
    pairs = collect_node_files()
    print(f"  Found {len(pairs)} matched node files (raw1 & raw3)")
    if rank_df is None:
        print("  WARNING: 0_rank.csv 없음 — UEP 배제/참고값 없이 기존 방식으로 폴백")

    err_safe2_list, err_danger2_list = [], []
    err_safe3_list, err_danger3_list = [], []
    skipped = 0

    for node, fp2, fp3 in pairs:
        excl_range, (e_danger, e_safety) = node_uep_and_ref(rank_df, node)

        try:
            safe2, danger2 = load_energy_series(fp2, node_count=NODE_NUM, excl_range=excl_range)
            safe3, danger3 = load_energy_series(fp3, node_count=NODE_NUM, excl_range=excl_range)
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            skipped += 1
            continue

        # MAX_ITER 이내로 먼저 잘라낸 뒤, 그 안에서만 참고값/오차를 계산한다
        # (자르기 전 전체 시계열의 마지막 값을 참고하면 화면 밖 데이터가
        #  기준값에 섞여 들어가는 문제가 생긴다)
        if MAX_ITER:
            safe2, danger2 = safe2[:MAX_ITER], danger2[:MAX_ITER]
            safe3, danger3 = safe3[:MAX_ITER], danger3[:MAX_ITER]

        # NaN(미추정)은 그대로, 0(의도된 값)은 유지

        def final_val(arr):
            v = arr[~np.isnan(arr)]
            return v[-1] if len(v) else np.nan

        # 참고값은 항상 "실제로 이 잘린(MAX_ITER) 구간에서 safe2/safe3/danger2/
        # danger3가 도달하는 값"보다 더 극단적인 쪽으로 잡아야 한다. 0_rank.csv의
        # 외부 참고값만 단독으로 쓰면, 그 값이 이 구간의 내부 관측치보다 덜
        # 극단적일 경우 running max(=safe)/min(=danger)가 도중에 참고값을
        # 넘어서 버려서 오차가 줄다가 다시 느는 비단조 현상이 생긴다.
        internal_ref_safe   = max(final_val(safe2), final_val(safe3))
        internal_ref_danger = min(final_val(danger2), final_val(danger3))

        if np.isfinite(e_danger) and abs(e_danger) > 1e-12:
            ref_safe = (max(e_danger, internal_ref_safe)
                       if np.isfinite(internal_ref_safe) else e_danger)
        else:
            ref_safe = internal_ref_safe

        if np.isfinite(e_safety) and abs(e_safety) > 1e-12:
            ref_danger = (min(e_safety, internal_ref_danger)
                         if np.isfinite(internal_ref_danger) else e_safety)
        else:
            ref_danger = internal_ref_danger

        if not (np.isfinite(ref_safe) and abs(ref_safe) > 1e-12 and
                np.isfinite(ref_danger) and abs(ref_danger) > 1e-12):
            skipped += 1
            continue

        err_safe2_list.append((safe2 - ref_safe) / abs(ref_safe))
        err_safe3_list.append((safe3 - ref_safe) / abs(ref_safe))
        err_danger2_list.append((danger2 - ref_danger) / abs(ref_danger))
        err_danger3_list.append((danger3 - ref_danger) / abs(ref_danger))

    print(f"  Valid nodes: {len(err_safe2_list)}  (skipped {skipped})")
    if not err_safe2_list:
        print("  No valid data, skip fig7c")
        return

    # 이미 MAX_ITER 이내로 잘랐으므로 실제 최대 길이를 그대로 사용
    max_len = max(len(a) for a in err_safe2_list)

    arr_safe2   = pad(err_safe2_list,   max_len)
    arr_safe3   = pad(err_safe3_list,   max_len)
    arr_danger2 = pad(err_danger2_list, max_len)
    arr_danger3 = pad(err_danger3_list, max_len)

    mean_s2, std_s2 = compute_stats(arr_safe2)
    mean_s3, std_s3 = compute_stats(arr_safe3)
    mean_d2, std_d2 = compute_stats(arr_danger2)
    mean_d3, std_d3 = compute_stats(arr_danger3)

    x = np.arange(5, max_len + 5)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    def plot_band(mean, std, color, ls, lw, label):
        valid = ~np.isnan(mean)
        ax.plot(x[valid], mean[valid], color=color, ls=ls, lw=lw,
                label=label, zorder=4)

    # E_danger_min
    plot_band(mean_d2, std_d2, COLOR_EDANGER_LINE, '--', 1.2,
              r'uniform-random, $E_{i}^{\mathrm{S}}$')
    plot_band(mean_d3, std_d3, COLOR_EDANGER_LINE, '-', 1.6,
              r'boundary-focused, $E_{i}^{\mathrm{S}}$')

    # E_safe_max
    plot_band(mean_s2, std_s2, COLOR_SAFE_DARK, '--', 1.2,
              r'uniform-random, $E_{i}^{\mathrm{U}}$')
    plot_band(mean_s3, std_s3, COLOR_SAFE_DARK, '-', 1.6,
              r'boundary-focused, $E_{i}^{\mathrm{U}}$')

    ax.set_yscale('log')
    set_log_ylim_from_last(ax, [mean_d2, mean_d3, mean_s2, mean_s3])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Normalized mean absolute error')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig7c.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig7c.pdf')




def compute_network_error_curves(net_root, label="", node_skip=1, step_skip=1, node_count=None):
    """한 네트워크의 모든 노드에 대해 균등/집중 탐색의 정규화 평균절대오차 곡선 계산.
    반환: (x, mean_safety2, mean_safety3, mean_danger2, mean_danger3) 또는 None
    safety = CSV의 E_danger_min 컬럼 기반 (= 논문의 E^safety)
    danger = CSV의 E_safe_max   컬럼 기반 (= 논문의 E^danger)

    node_skip > 1이면 노드를 그 값만큼 건너뛰며 처리 (대강 결과부터 빠르게 확인용).
    step_skip > 1이면 반복(iteration) 축을 그 간격으로 서브샘플링
    (예: 50이면 50스텝당 1개만 사용, x축 커버 범위는 유지하면서 계산량만 줄임).
    결과는 캐시에 저장되며, node_skip/step_skip 값이 파일명에 포함되어 구분된다."""
    target_kr = DEFAULT_KR_MAP.get(net_root.name, DEFAULT_KR)
    opt_suffix = f"_allA{int(ALL_ANALYSIS)}_allK{int(ALL_KR)}"
    if not ALL_KR:
        opt_suffix += f"_kr{target_kr:.2f}"
    # UEP 배제구간 필터링 도입으로 계산 로직이 바뀌어서, 이전 캐시(옛 로직)와
    # 섞이지 않도록 suffix에 _uepexcl2를 추가 (0_rank.csv를 재생성했거나 이
    # E_temp 재계산 로직이 바뀌었다면 이 캐시도 지우고 다시 계산해야 함 —
    # CACHE_DIR 통째로 비우면 됨)
    cache_fp = CACHE_DIR / f"{net_root.name}_nskip{node_skip}_tskip{step_skip}_maxit{MAX_ITER}{opt_suffix}_uepexcl2.npz"
    if cache_fp.exists():
        print(f"    (cache hit: {cache_fp.name})")
        d = np.load(cache_fp)
        return d['x'], d['mean_safety2'], d['mean_safety3'], d['mean_danger2'], d['mean_danger3']

    rank_df = load_rank_uep(net_root)
    if rank_df is None:
        print(f"    WARNING: {net_root.name}에 0_rank.csv 없음 — UEP 배제/참고값 없이 폴백")

    pairs = collect_node_files(net_root)
    pairs = pairs[::node_skip]
    n_pairs = len(pairs)
    err_safety2_list, err_safety3_list = [], []
    err_danger2_list, err_danger3_list = [], []

    for i, (node, fp2, fp3) in enumerate(pairs):
        if n_pairs >= 10 and (i % max(1, n_pairs // 10) == 0):
            print(f"    [{label}] node {i+1}/{n_pairs} ({(i+1)/n_pairs:.0%})")

        excl_range, (e_danger, e_safety) = node_uep_and_ref(rank_df, node)

        try:
            safe2, danger2 = load_energy_series(fp2, node_count=node_count, excl_range=excl_range)
            safe3, danger3 = load_energy_series(fp3, node_count=node_count, excl_range=excl_range)
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            continue

        # MAX_ITER 이내로 먼저 잘라낸 뒤, 그 안에서만 참고값/오차를 계산한다
        # (자르기 전 전체 시계열의 마지막 값을 참고하면 화면 밖 데이터가
        #  기준값에 섞여 들어가는 문제가 생긴다)
        if MAX_ITER:
            safe2, danger2 = safe2[:MAX_ITER], danger2[:MAX_ITER]
            safe3, danger3 = safe3[:MAX_ITER], danger3[:MAX_ITER]

        def final_val(arr):
            v = arr[~np.isnan(arr)]
            return v[-1] if len(v) else np.nan

        # 참고값은 항상 "실제로 이 잘린(MAX_ITER) 구간에서 safe2/safe3/danger2/
        # danger3가 도달하는 값"보다 더 극단적인 쪽으로 잡아야 한다. 0_rank.csv의
        # 외부 참고값만 단독으로 쓰면, 그 값이 이 구간의 내부 관측치보다 덜
        # 극단적일 경우 running max(=safe)/min(=danger)가 도중에 참고값을
        # 넘어서 버려서 오차가 줄다가 다시 느는 비단조 현상이 생긴다.
        internal_ref_danger = max(final_val(safe2), final_val(safe3))
        internal_ref_safety = min(final_val(danger2), final_val(danger3))

        if np.isfinite(e_danger) and abs(e_danger) > 1e-12:
            ref_danger = (max(e_danger, internal_ref_danger)
                         if np.isfinite(internal_ref_danger) else e_danger)
        else:
            ref_danger = internal_ref_danger

        if np.isfinite(e_safety) and abs(e_safety) > 1e-12:
            ref_safety = (min(e_safety, internal_ref_safety)
                         if np.isfinite(internal_ref_safety) else e_safety)
        else:
            ref_safety = internal_ref_safety

        if not (np.isfinite(ref_danger) and abs(ref_danger) > 1e-12 and
                np.isfinite(ref_safety) and abs(ref_safety) > 1e-12):
            continue

        err_d2 = (safe2   - ref_danger) / abs(ref_danger)
        err_d3 = (safe3   - ref_danger) / abs(ref_danger)
        err_s2 = (danger2 - ref_safety) / abs(ref_safety)
        err_s3 = (danger3 - ref_safety) / abs(ref_safety)

        if step_skip > 1:
            err_d2 = err_d2[::step_skip]
            err_d3 = err_d3[::step_skip]
            err_s2 = err_s2[::step_skip]
            err_s3 = err_s3[::step_skip]

        err_danger2_list.append(err_d2)
        err_danger3_list.append(err_d3)
        err_safety2_list.append(err_s2)
        err_safety3_list.append(err_s3)

    if not err_safety2_list:
        return None

    # 이미 MAX_ITER 이내로 잘랐으므로(및 step_skip 적용 후) 실제 길이를 그대로 사용
    max_len = max(len(a) for a in err_safety2_list)
    arr_safety2 = pad(err_safety2_list, max_len)
    arr_safety3 = pad(err_safety3_list, max_len)
    arr_danger2 = pad(err_danger2_list, max_len)
    arr_danger3 = pad(err_danger3_list, max_len)

    mean_safety2, _ = compute_stats(arr_safety2)
    mean_safety3, _ = compute_stats(arr_safety3)
    mean_danger2, _ = compute_stats(arr_danger2)
    mean_danger3, _ = compute_stats(arr_danger3)

    x = 5 + step_skip * np.arange(max_len)

    np.savez(cache_fp, x=x, mean_safety2=mean_safety2, mean_safety3=mean_safety3,
             mean_danger2=mean_danger2, mean_danger3=mean_danger3)
    print(f"    (cached: {cache_fp.name})")

    return x, mean_safety2, mean_safety3, mean_danger2, mean_danger3


def fig7d():
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 10.0))
    axes_flat = axes.flatten()

    for idx, ((folder, label, node_count), ax) in enumerate(zip(DATASETS_D, axes_flat)):
        net_root = _base.parent / "02_network" / folder
        print(f"  Processing {label} ({folder}, N={node_count}) ...")
        result = compute_network_error_curves(net_root, label=label,
                                              node_skip=NODE_SKIP, step_skip=STEP_SKIP,
                                              node_count=node_count)
        if result is None:
            print(f"    WARNING: no valid data for {label}, skip")
            ax.axis('off')
            continue
        x, mean_safety2, mean_safety3, mean_danger2, mean_danger3 = result

        # E_safety: 초록, E_danger: 빨강 (fig7c와 동일한 원래 색)
        v2 = ~np.isnan(mean_safety2); v3 = ~np.isnan(mean_safety3)
        ax.plot(x[v2], mean_safety2[v2], color=COLOR_EDANGER_LINE, ls='--', lw=1.2, zorder=3)
        ax.plot(x[v3], mean_safety3[v3], color=COLOR_EDANGER_LINE, ls='-',  lw=1.6, zorder=4)

        v2 = ~np.isnan(mean_danger2); v3 = ~np.isnan(mean_danger3)
        ax.plot(x[v2], mean_danger2[v2], color=COLOR_SAFE_DARK, ls='--', lw=1.2, zorder=3)
        ax.plot(x[v3], mean_danger3[v3], color=COLOR_SAFE_DARK, ls='-',  lw=1.6, zorder=4)

        ax.set_yscale('log')
        set_log_ylim_from_last(ax, [mean_safety2, mean_safety3, mean_danger2, mean_danger3],
                               x=x, x_limit=PLOT_ITER)
        ax.set_xlim(left=PLOT_ITER_MIN, right=PLOT_ITER)
        ax.set_title(f'{label}', fontsize=20, pad=10)
        if idx >= 4:  # 아래쪽 행에만 x축 라벨
            ax.set_xlabel('Iteration', fontsize=20)
        # else:  # 위쪽 행은 x축 눈금값 숨김 (아래쪽 행과 중복)
        #     ax.tick_params(labelbottom=False)
        if idx % 4 == 0:  # 맨 왼쪽 열에만 y축 라벨
            ax.set_ylabel('Norm. MAE (log)', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.grid(True, which='major', alpha=0.25)
        ax.grid(True, which='minor', alpha=0.1)

    # 데이터셋 개수가 8보다 적으면 남는 축은 숨김
    for ax in axes_flat[len(DATASETS_D):]:
        ax.axis('off')

    # 그림 전체 공용 legend (모든 패널이 같은 색·스타일 의미를 공유)
    style_handles = [
        Line2D([0], [0], color=COLOR_EDANGER_LINE, lw=1.2, ls='--',
              label=r'uniform-random, $E_i^{\mathrm{S}}$'),
        Line2D([0], [0], color=COLOR_EDANGER_LINE, lw=1.6, ls='-',
              label=r'boundary-focused, $E_i^{\mathrm{S}}$'),
        Line2D([0], [0], color=COLOR_SAFE_DARK, lw=1.2, ls='--',
              label=r'uniform-random, $E_i^{\mathrm{U}}$'),
        Line2D([0], [0], color=COLOR_SAFE_DARK, lw=1.6, ls='-',
              label=r'boundary-focused, $E_i^{\mathrm{U}}$'),
    ]
    fig.legend(handles=style_handles, loc='lower center', ncol=4,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.02), fontsize=20)

    fig.subplots_adjust(hspace=0.875, wspace=0.30)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = ''
    if NODE_SKIP != 1: suffix += f'_nskip{NODE_SKIP}'
    if STEP_SKIP != 1: suffix += f'_tskip{STEP_SKIP}'
    out_name = f'fig7d{suffix}.pdf'
    fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_name}')


# ================================================================
# LaTeX table — 데이터셋별 수렴 속도 요약
# 행: 국가(네트워크), 열: (Uniform-random / Boundary-focused) x
#     (E^safety / E^danger) x (10% / 1% 미만 도달 첫 iteration)
# ================================================================


def first_iter_below_threshold(x, series, threshold):
    """series(NaN 포함 가능)가 threshold 미만이 되는 첫 iteration(x값)을 반환.
    한 번도 도달하지 못하면 None."""
    valid = ~np.isnan(series)
    below = valid & (series < threshold)
    if not below.any():
        return None
    idx = np.argmax(below)  # 첫 True의 인덱스
    return int(x[idx])


def format_cell(val, bold=False):
    s = str(val) if val is not None else '---'
    return r'\textbf{' + s + '}' if bold else s


def _threshold_key(t):
    """threshold(0.10 등)를 딕셔너리 키/라벨에 쓸 문자열로 변환 (예: 0.10 -> '10', 0.01 -> '1')."""
    pct = t * 100
    return str(int(pct)) if pct == int(pct) else str(pct)


def combined_iter(safety_iter, danger_iter):
    """E^safety, E^danger 중 더 늦게(iteration이 더 큰 쪽) 도달하는 값을 반환.
    둘 중 하나라도 끝내 도달 못 했으면(None) 전체도 도달 못 한 것으로 본다."""
    if safety_iter is None or danger_iter is None:
        return None
    return max(safety_iter, danger_iter)


def combined_series(a, b):
    """두 시계열의 원소별로 더 나쁜(큰) 값을 취한다 (worse-of-two).
    둘 다 NaN인 지점만 NaN, 한쪽만 유효하면 그 값을 그대로 쓴다."""
    return np.fmax(a, b)


def first_iter_where_better(x, series_a, series_b):
    """series_b가 series_a보다 '영구적으로' 더 좋아지는(작아지는) 지점을
    반환한다. 중간에 일시적으로 역전됐다가 다시 원상복구되는 노이즈에
    흔들리지 않도록, 맨 마지막 iteration부터 거꾸로 훑어서 "아직 series_b가
    더 안 좋았던(또는 같았던)" 마지막 지점을 찾고, 그 바로 다음 지점을
    crossing으로 삼는다.
    마지막 유효 지점에서조차 series_b가 series_a보다 더 안 좋으면(=영구적
    역전이 끝내 없었다는 뜻) None을 반환한다."""
    valid = ~np.isnan(series_a) & ~np.isnan(series_b)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return None

    better = valid & (series_b < series_a)

    last_idx = valid_idx[-1]
    if not better[last_idx]:
        return None  # 끝까지 series_b가 더 안 좋음(또는 동률) -> 영구적 역전 없음

    worse_positions = np.where(valid & ~better)[0]
    if len(worse_positions) == 0:
        idx = valid_idx[0]   # 유효구간 내내 계속 better -> 첫 유효 지점이 crossing
    else:
        last_worse = worse_positions[-1]
        after = valid_idx[valid_idx > last_worse]
        if len(after) == 0:
            return None
        idx = after[0]
    return int(x[idx])


def print_latex_table():
    """fig7d와 동일한 계산(compute_network_error_curves)을 재사용해서
    LaTeX 표를 print로 출력한다. 열 구성은 TABLE_THRESHOLDS 개수/값에
    맞춰 자동으로 결정된다: (Uniform-random / Boundary-focused) 각각
    TABLE_THRESHOLDS 개수만큼의 열 (E^safety, E^danger 중 더 늦게
    도달하는 쪽 기준으로 하나로 합쳐서 표시) + 맨 끝에 crossing 열 하나
    (boundary-focused가 uniform-random보다 처음으로 더 좋아지는 iteration,
    이것도 E^safety/E^danger 중 더 늦은 쪽 기준)."""
    thr_keys = [_threshold_key(t) for t in TABLE_THRESHOLDS]
    n_thr = len(TABLE_THRESHOLDS)
    n_cols = 2 * n_thr   # uniform + focused, 각각 threshold 개수만큼

    rows = []
    for folder, label, node_count in DATASETS_D:
        net_root = _base.parent / "02_network" / folder
        print(f"  [table] Processing {label} ({folder}, N={node_count}) ...")
        result = compute_network_error_curves(net_root, label=label,
                                              node_skip=NODE_SKIP, step_skip=STEP_SKIP,
                                              node_count=node_count)
        if result is None:
            print(f"    WARNING: no valid data for {label}, skip (표에서 제외)")
            continue
        x, mean_safety2, mean_safety3, mean_danger2, mean_danger3 = result

        row = {'label': label}
        for prefix, safety_series, danger_series in [
            ('u', mean_safety2, mean_danger2),   # Uniform-random
            ('f', mean_safety3, mean_danger3),   # Boundary-focused
        ]:
            for t, tk in zip(TABLE_THRESHOLDS, thr_keys):
                s_iter = first_iter_below_threshold(x, safety_series, t)
                d_iter = first_iter_below_threshold(x, danger_series, t)
                row[f'{prefix}_{tk}'] = combined_iter(s_iter, d_iter)

        combined_uniform = combined_series(mean_safety2, mean_danger2)
        combined_focused = combined_series(mean_safety3, mean_danger3)
        # 최종 시점에 focused쪽 오차가 더 크면(=uniform이 결국 앞서면) 그 영구적
        # crossing iter를, uniform쪽 오차가 더 크면(=focused가 결국 앞서는
        # 정상적인 경우) '--'를 반환하도록 인자 순서를 반대로 넣는다.
        crossing_iter = first_iter_where_better(x, combined_focused, combined_uniform)
        row['crossing'] = crossing_iter
        if crossing_iter is not None:
            # crossing이 일어난 그 iteration에서 focused(집중탐색)의 오차(%)를 함께 기록
            idx = np.where(x == crossing_iter)[0]
            row['crossing_pct'] = float(combined_focused[idx[0]]) * 100.0 if len(idx) else None
        else:
            row['crossing_pct'] = None
        row['node_count'] = node_count   # 맨 아래 가중평균 계산용
        rows.append(row)

    def cell_pair(row, tk):
        """threshold tk에서 uniform/focused 중 더 빨리(작은 iter) 도달한 쪽을
        볼드 처리. 한쪽만 도달했으면(다른 쪽 '--') 도달한 쪽을 볼드, 같으면
        볼드 없음."""
        u_val = row[f'u_{tk}']
        f_val = row[f'f_{tk}']
        u_bold = f_bold = False
        if u_val is not None and f_val is not None:
            if u_val < f_val:
                u_bold = True
            elif f_val < u_val:
                f_bold = True
        elif u_val is not None:
            u_bold = True
        elif f_val is not None:
            f_bold = True
        return format_cell(u_val, u_bold), format_cell(f_val, f_bold)

    def format_crossing_cell(iter_val, pct_val):
        """crossing 열 전용 포맷: "iter (x.x%)" — crossing이 없으면 '---'.
        pct_val이 없으면(예: 가중평균 행) 퍼센트 없이 iter만 표시."""
        if iter_val is None:
            return '---'
        if pct_val is None:
            return f"{iter_val:g}"
        return f"{iter_val:g} ({pct_val:.1f}\\%)"

    def weighted_avg(values, weights, fill_missing):
        """values 중 None은 fill_missing으로 대체한 뒤, weights로 가중평균."""
        vals = np.array([fill_missing if v is None else v for v in values], dtype=float)
        w = np.array(weights, dtype=float)
        return float(np.sum(vals * w) / np.sum(w))

    col_spec = 'l ' + 'c' * n_thr + ' ' + 'c' * n_thr + ' c'
    header_labels = [tk + r'\%' for tk in thr_keys] * 2
    thr_list_str = ' / '.join(f'{t*100:g}\\%' for t in TABLE_THRESHOLDS)

    print("\n" + "=" * 70)
    print("LaTeX table (fig7 convergence summary)")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\hline")
    print(r" & \multicolumn{" + str(n_thr) + r"}{c}{Uniform-random sampling} & "
         r"\multicolumn{" + str(n_thr) + r"}{c}{Boundary-focused search} & \\")
    print(r"\cmidrule(lr){2-" + str(1 + n_thr) + "} "
         r"\cmidrule(lr){" + str(2 + n_thr) + "-" + str(1 + n_cols) + "}")
    print("Network & " + " & ".join(header_labels) + " & Crossing" + r" \\")
    print(r"\hline")
    for row in rows:
        pairs = [cell_pair(row, tk) for tk in thr_keys]
        u_cells = [p[0] for p in pairs]
        f_cells = [p[1] for p in pairs]
        cross_cell = format_crossing_cell(row['crossing'], row['crossing_pct'])
        cells = u_cells + f_cells + [cross_cell]
        print(f"{row['label']} & " + " & ".join(cells) + r" \\")

    # ── 맨 아래: 노드 수 가중평균 행 ('---'는 MAX_ITER로 간주해서 계산) ──
    if rows:
        print(r"\hline")
        weights = [row['node_count'] for row in rows]
        u_avgs, f_avgs = [], []
        for tk in thr_keys:
            u_avg = weighted_avg([row[f'u_{tk}'] for row in rows], weights, MAX_ITER)
            f_avg = weighted_avg([row[f'f_{tk}'] for row in rows], weights, MAX_ITER)
            u_avgs.append(u_avg)
            f_avgs.append(f_avg)
        avg_cells = []
        for u_avg, f_avg in zip(u_avgs, f_avgs):
            avg_cells.append(format_cell(f"{u_avg:.1f}", u_avg < f_avg))
        for u_avg, f_avg in zip(u_avgs, f_avgs):
            avg_cells.append(format_cell(f"{f_avg:.1f}", f_avg < u_avg))
        cross_avg = weighted_avg([row['crossing'] for row in rows], weights, MAX_ITER)
        # crossing_pct는 '--'(도달 안 함)에 대응할 자연스러운 대체값이 없으므로,
        # 실제로 crossing이 일어난 데이터셋들만 노드수 가중평균한다.
        pct_pairs = [(row['crossing_pct'], row['node_count']) for row in rows
                    if row['crossing_pct'] is not None]
        if pct_pairs:
            pct_vals, pct_weights = zip(*pct_pairs)
            cross_pct_avg = float(np.average(pct_vals, weights=pct_weights))
        else:
            cross_pct_avg = None
        avg_cells.append(format_crossing_cell(round(cross_avg, 1), cross_pct_avg))
        print(r"Weighted average & " + " & ".join(avg_cells) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{")
    print(r"Number of iterations required for the normalized mean absolute "
         r"errors (Norm. MAE) of both energy thresholds, $E^{\mathrm{S}}$ "
         r"and $E^{\mathrm{U}}$, to fall below " + thr_list_str + r" for "
         r"each search method, uniform-random and boundary-focused.")
    print(r"Bold values indicate the search method requiring fewer "
         r"iterations to reach the corresponding error level.")
    print(r"``---'' indicates that both energy thresholds did not reach "
         r"the specified error level within " + str(MAX_ITER) + r" "
         r"iterations.")
    print(r"The Crossing column reports the last iteration at which "
         r"uniform-random sampling permanently becomes more accurate "
         r"than boundary-focused search (i.e., the two search methods' "
         r"errors, each combined by taking whichever of $E^{\mathrm{S}}$/"
         r"$E^{\mathrm{U}}$ is larger, swap ranking and never swap back "
         r"within " + str(MAX_ITER) + r" iterations), together with "
         r"boundary-focused search's error at that iteration in "
         r"parentheses; it is reported only when boundary-focused search "
         r"ends up less accurate than uniform-random sampling by the "
         r"final iteration, and ``---'' otherwise.")
    print(r"The bottom row reports the node-count-weighted average across "
         r"all networks, treating ``---'' entries as " + str(MAX_ITER) +
         r" iterations; the Crossing column's percentage in this row is "
         r"averaged only over the networks where a crossing actually "
         r"occurred.")
    print(r"}")
    print(r"\label{tab:fig7_convergence}")
    print(r"\end{table}")
    print("=" * 70)


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("Loading network...")
    P, M, D, K, theta_eq = load_network()

    print("\nLoading CSVs...")
    df2 = pd.read_csv(CSV_RAW1)
    df3 = pd.read_csv(CSV_RAW3)
    print(f"  raw1: {len(df2)} rows")
    print(f"  raw3: {len(df3)} rows")

    print("\n--- fig7a (random / raw1) ---")
    basin_safety_plot(df2, P, M, D, K, theta_eq, 'fig7a')

    print("\n--- fig7b (focused / raw3) ---")
    basin_safety_plot(df3, P, M, D, K, theta_eq, 'fig7b')

    print("\n--- fig7c (convergence error) ---")
    fig7c()

    print("\n--- fig7d (convergence error across 8 networks) ---")
    fig7d()

    print("\n--- LaTeX table (convergence summary) ---")
    print_latex_table()

    print("\nAll done. Output in:", OUT_DIR)