# fig_2.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_2/
#   fig2a.pdf  — E_safe_max 설명 (sync safe 점 + E_safe_max 라인)
#   fig2b.pdf  — E_danger_min 설명 (sync danger + desync 점 + E_danger_min 라인)

import colorsys
import numpy as np
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
_base      = Path(__file__).resolve().parent
NET_ROOT   = _base.parent / "02_network" / "01_motif_3"
CSV_PATH   = NET_ROOT / "result_raw1" / "3_analysis0_n_0_Kr_10.00.csv"
NODE_FILE  = NET_ROOT / "network" / "3_sample0_nodes.txt"
EDGE_FILE  = NET_ROOT / "network" / "3_sample0_edges.txt"
INIT_FILE  = NET_ROOT / "init"    / "3_sample0_init_Kr_10.00.txt"
OUT_DIR    = _base / "fig_2"
OUT_DIR.mkdir(exist_ok=True)

# ── 파라미터 ────────────────────────────────────────────────────
N_CONTOURS = 12   # 에너지 등고선 개수

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
KR    = 10.0
N     = 3

COLOR_SAFE        = '#4C9BE8'
COLOR_SAFE_DARK   = '#B22222'   # E_safe_max 라인: 진한 파랑
COLOR_DANGER      = '#E8954C'
COLOR_DESYNC      = '#555555'
COLOR_EDANGER_LINE = '#006400'

# ── 온도계(컬러바) 스타일 파라미터 ─────────────────────────────
THERMO_LIGHTEN_FACTOR    = 0.6   # 채워지는 영역의 명도 조절 (0=원색 그대로, 1=흰색에 가깝게)
THERMO_SATURATE_FACTOR   = 0.5   # 채워지는 영역의 채도 배율 (0=완전 무채색, 1=원래 채도 그대로)
THERMO_BOUNDARY_COLOR    = None   # 경계선 색. None이면 fill_color(본문 라인색)를 그대로 사용
THERMO_BOUNDARY_LW       = 2.4    # 경계선 굵기 (본문 등에너지선과 맞추려면 2.4 권장)
THERMO_TICK_FONTSIZE     = 9      # 눈금값 폰트 크기
THERMO_TICK_COLOR        = 'black'  # 눈금값/눈금 색


# ================================================================
# 네트워크 로더
# ================================================================
def load_network():
    P = np.zeros(N); M = np.ones(N); D = np.full(N, 0.1)
    with open(NODE_FILE) as f:
        for line in f:
            p = line.split()
            if not p: continue
            i = int(p[0])
            if len(p) > 1: P[i] = float(p[1])
            if len(p) > 2: M[i] = float(p[2])
            if len(p) > 3: D[i] = float(p[3])
    K = np.zeros((N, N))
    with open(EDGE_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) < 3: continue
            u, v, w = int(p[0]), int(p[1]), float(p[2])
            K[u,v] = K[v,u] = w * KR
    theta_eq = np.zeros(N)
    with open(INIT_FILE) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                theta_eq[int(p[0])] = float(p[2])
    return P, M, K, theta_eq


# ================================================================
# iso-energy curve (4_plot.py 로직 참고)
# ================================================================
def potential_diff(target, theta_trial, theta_eq, P, K):
    dth = (theta_trial - theta_eq[target] + PI) % (2*PI) - PI
    V_trial = V_eq = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if K[i,j] > 0:
                if i == target or j == target:
                    nb = j if i == target else i
                    V_trial -= K[target, nb] * np.cos(theta_trial     - theta_eq[nb])
                    V_eq    -= K[target, nb] * np.cos(theta_eq[target] - theta_eq[nb])
    V_trial -= P[target] * dth
    return V_trial - V_eq


def wrap_to_pi(x):
    return (x + PI) % (2*PI) - PI


def compute_uep_dtheta(target, theta_eq, P, K):
    """Local mean-field 방법으로 UEP의 delta-theta를 계산 (attack_random_server.c와 동일 로직).
    반환: (uep_dtheta, valid). valid=False면 계산 불가(폴백, 제한 없음)."""
    sum_cos = 0.0; sum_sin = 0.0
    for nb in range(N):
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


def draw_iso_curve(ax, E_val, target, theta_eq, P, M, K, color, lw=2.4, excl_range=None):
    segs_pos, segs_neg = iso_energy_curve(E_val, target, theta_eq, P, M, K, excl_range=excl_range)
    for th_arr, w_arr in segs_pos + segs_neg:
        ax.plot(th_arr, w_arr, color=color, linewidth=lw, zorder=5, linestyle='-')


def _ax_setup(ax, excl_range=None):
    ax.set_facecolor('white')
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_{0}$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_{0}$')

    if excl_range is not None:
        ax.axvspan(excl_range[0], excl_range[1], facecolor='black', alpha=0.5,
                  zorder=25, edgecolor='none', linewidth=0)

    # 사각 테두리(spine)를 온도계 인셋(zorder=100)보다도 위로 올려서 맨 위에 그려지게
    for spine in ax.spines.values():
        spine.set_zorder(150)


# ================================================================
# 온도계 인셋 (단일색 버전): 2a=초록만(하한~E_danger), 2b=빨강만(E_safety~상한)
# - 산점도 왼쪽 안쪽, 세로 90% 높이, 가운데 정렬
# - 축 범위는 2a/2b가 공유하는 3×10^n 단위 (E_safe_max, E_danger_min 함께 고려)
# - 눈금: 10의 배수(major)만 라벨 표시(그림 중앙), 2~9배수(minor)는 짧은 틱만
# - 눈금/눈금값 모두 검은색, 배경색은 완전 불투명(투명도 없음)
# - 항상 다른 요소보다 위에 그려짐 (zorder=100)
# ================================================================
def soften_color(hex_color, lighten_factor=0.75, saturate_factor=0.30):
    """명도(L)를 흰색 쪽으로, 채도(S)를 낮추는 두 조절을 함께 적용 — 온도계 배경용
    lighten_factor: 0=원래 명도, 1=흰색에 가깝게
    saturate_factor: 0=완전 무채색, 1=원래 채도 그대로"""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1.0 - l) * lighten_factor
    s = s * saturate_factor
    return colorsys.hls_to_rgb(h, l, s)


def draw_thermometer_single(ax_th, thermo_lo, thermo_hi, boundary, fill_color, side):
    """side='below': [thermo_lo, boundary]를 fill_color로, 위는 흰색
       side='above': [boundary, thermo_hi]를 fill_color로, 아래는 흰색
       채워지는 영역은 채도를 낮춘 색으로, 경계값(boundary) 위치에는
       본문 그림의 등에너지선과 동일한 색·굵기의 강조선을 그어
       눈금/눈금값이 잘 보이면서도 실제 값의 위치는 뚜렷하게 표시한다."""
    ax_th.set_yscale('log')
    ax_th.set_ylim(thermo_lo, thermo_hi)
    ax_th.set_xlim(0, 1)
    ax_th.set_xticks([])

    soft_color = soften_color(fill_color, lighten_factor=THERMO_LIGHTEN_FACTOR,
                              saturate_factor=THERMO_SATURATE_FACTOR)
    boundary_color = THERMO_BOUNDARY_COLOR if THERMO_BOUNDARY_COLOR is not None else fill_color

    if np.isfinite(boundary):
        if side == 'below':
            ax_th.axhspan(thermo_lo, boundary, color=soft_color, zorder=1)
            ax_th.axhspan(boundary, thermo_hi, color='white', zorder=1)
        else:  # 'above'
            ax_th.axhspan(thermo_lo, boundary, color='white', zorder=1)
            ax_th.axhspan(boundary, thermo_hi, color=soft_color, zorder=1)
        # 경계값 위치: 본문의 등에너지선과 동일한 색·굵기로 강조
        ax_th.axhline(boundary, color=boundary_color, lw=THERMO_BOUNDARY_LW, zorder=8)
    else:
        ax_th.axhspan(thermo_lo, thermo_hi, color='white', zorder=1)

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


def thermo_range_from_values(*values):
    """여러 E 값으로부터 공통 3×10^n 온도계 범위 계산"""
    vals = [v for v in values if np.isfinite(v) and v > 0]
    if not vals:
        return (3e-2, 3e1)
    lo_val, hi_val = min(vals), max(vals)
    decade_lo = int(np.floor(np.log10(lo_val)))
    decade_hi = int(np.ceil(np.log10(hi_val)))
    if decade_lo == decade_hi:
        decade_hi = decade_lo + 1
    return (3 * 10**(decade_lo - 1), 3 * 10**decade_hi)


# ================================================================
# fig2a — E_safe_max
# ================================================================
def fig2a(df, P, M, K, theta_eq, E_safe_max, E_danger_min, attack_node=0):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    uep_dtheta, uep_valid = compute_uep_dtheta(attack_node, theta_eq, P, K)
    excl_range = excluded_theta_range(uep_dtheta, uep_valid)

    _ax_setup(ax, excl_range=excl_range)

    mask_sync   = df['num_synced'] >= N
    has_safe    = 'safe' in df.columns
    mask_safe   = mask_sync & (df['safe'] == 1) if has_safe else mask_sync

    # sync (secured) 점만
    ax.scatter(df.loc[mask_safe, 'attack_theta'],
               df.loc[mask_safe, 'attack_w'],
               c=COLOR_SAFE, s=20, linewidths=0, rasterized=True, zorder=3,
               label='sync (secured)')

    # E_safe_max 라인
    draw_iso_curve(ax, E_safe_max, attack_node, theta_eq, P, M, K,
                   color=COLOR_SAFE_DARK, lw=2.8, excl_range=excl_range)



    # 에너지 등고선: 4모서리 최대 에너지 기준으로 10개
    corners = [(-PI, -W_MAX), (-PI, W_MAX), (PI, -W_MAX), (PI, W_MAX)]
    E_corners = [potential_diff(attack_node,
                                theta_eq[attack_node] + cx,
                                theta_eq, P, K) + 0.5 * M[attack_node] * cy**2
                 for cx, cy in corners]
    E_max_corner = max(E_corners)
    for E_c in np.linspace(E_max_corner / N_CONTOURS, E_max_corner, N_CONTOURS):
        segs_p, segs_n = iso_energy_curve(E_c, attack_node, theta_eq, P, M, K,
                                          excl_range=excl_range)
        for th_arr, w_arr in segs_p + segs_n:
            ax.plot(th_arr, w_arr, color='#aaaaaa', linewidth=0.8,
                    zorder=15, linestyle='-')

    legend_handles = [
        mpatches.Patch(color=COLOR_SAFE, label='sync (secured)'),
        Line2D([0],[0], color=COLOR_SAFE_DARK, lw=2.4,
               label=r'$E_{0}^{\mathrm{U}} = ' + f'{E_safe_max:.3f}$'),
    ]
    leg = ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    leg.set_zorder(20)

    # 온도계 인셋: 빨강만(E_danger ~ 상한), 나머지는 흰색. 범위는 2a/2b 공유.
    thermo_lo, thermo_hi = thermo_range_from_values(E_safe_max, E_danger_min)
    ax_th = ax.inset_axes([0.05, 0.05, 0.09, 0.90], zorder=100)
    ax_th.set_zorder(100)
    ax_th.patch.set_zorder(100)
    draw_thermometer_single(ax_th, thermo_lo, thermo_hi, E_safe_max,
                            COLOR_SAFE_DARK, side='above')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2a.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: fig2a.pdf  |  E_safe_max = {E_safe_max:.4f}')


# ================================================================
# fig2b — E_danger_min
# ================================================================
def fig2b(df, P, M, K, theta_eq, E_danger_min, E_safe_max, attack_node=0):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    uep_dtheta, uep_valid = compute_uep_dtheta(attack_node, theta_eq, P, K)
    excl_range = excluded_theta_range(uep_dtheta, uep_valid)

    _ax_setup(ax, excl_range=excl_range)

    mask_sync   = df['num_synced'] >= N
    has_safe    = 'safe' in df.columns
    mask_danger = mask_sync & (df['safe'] != 1) if has_safe else pd.Series([False]*len(df))
    mask_desync = df['num_synced'] < N

    # sync (unsecured) 점
    ax.scatter(df.loc[mask_danger, 'attack_theta'],
               df.loc[mask_danger, 'attack_w'],
               c=COLOR_DANGER, s=20, linewidths=0, rasterized=True, zorder=3,
               label='sync (unsecured)')
    # desync 점
    ax.scatter(df.loc[mask_desync, 'attack_theta'],
               df.loc[mask_desync, 'attack_w'],
               c=COLOR_DESYNC, s=20, linewidths=0, rasterized=True, zorder=2,
               label='desync')

    # E_danger_min 라인
    draw_iso_curve(ax, E_danger_min, attack_node, theta_eq, P, M, K,
                   color=COLOR_EDANGER_LINE, lw=2.8, excl_range=excl_range)



    # 에너지 등고선: 4모서리 최대 에너지 기준으로 10개
    corners = [(-PI, -W_MAX), (-PI, W_MAX), (PI, -W_MAX), (PI, W_MAX)]
    E_corners = [potential_diff(attack_node,
                                theta_eq[attack_node] + cx,
                                theta_eq, P, K) + 0.5 * M[attack_node] * cy**2
                 for cx, cy in corners]
    E_max_corner = max(E_corners)
    for E_c in np.linspace(E_max_corner / N_CONTOURS, E_max_corner, N_CONTOURS):
        segs_p, segs_n = iso_energy_curve(E_c, attack_node, theta_eq, P, M, K,
                                          excl_range=excl_range)
        for th_arr, w_arr in segs_p + segs_n:
            ax.plot(th_arr, w_arr, color='#aaaaaa', linewidth=0.8,
                    zorder=15, linestyle='-')

    legend_handles = [
        mpatches.Patch(color=COLOR_DANGER, label='sync (unsecured)'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync (unsecured)'),
        Line2D([0],[0], color=COLOR_EDANGER_LINE, lw=2.4,
               label=r'$E_{0}^{\mathrm{S}} = ' + f'{E_danger_min:.3f}$'),
    ]
    leg = ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    leg.set_zorder(20)

    # 온도계 인셋: 초록만(하한 ~ E_safety), 나머지는 흰색. 범위는 2a/2b 공유.
    thermo_lo, thermo_hi = thermo_range_from_values(E_safe_max, E_danger_min)
    ax_th = ax.inset_axes([0.05, 0.05, 0.09, 0.90], zorder=100)
    ax_th.set_zorder(100)
    ax_th.patch.set_zorder(100)
    draw_thermometer_single(ax_th, thermo_lo, thermo_hi, E_danger_min,
                            COLOR_EDANGER_LINE, side='below')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2b.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: fig2b.pdf  |  E_danger_min = {E_danger_min:.4f}')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # E_safe_max, E_danger_min: 마지막 행에서 읽기
    last = df.iloc[-1]
    E_safe_max   = float(last['E_safe_max'])   if 'E_safe_max'   in df.columns else float('nan')
    E_danger_min = float(last['E_danger_min']) if 'E_danger_min' in df.columns else float('nan')
    print(f"  E_safe_max   = {E_safe_max:.4f}")
    print(f"  E_danger_min = {E_danger_min:.4f}")

    print("Loading network...")
    P, M, K, theta_eq = load_network()

    print("\n--- fig2a (E_safe_max) ---")
    fig2a(df, P, M, K, theta_eq, E_safe_max, E_danger_min, attack_node=0)

    print("\n--- fig2b (E_danger_min) ---")
    fig2b(df, P, M, K, theta_eq, E_danger_min, E_safe_max, attack_node=0)

    print("\nAll done. Output in:", OUT_DIR)