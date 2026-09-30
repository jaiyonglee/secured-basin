# fig_4.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_4/
#   fig4a.pdf  — 고장 모델 개념도 (인셋: P_i(t) 시계열)
#   fig4b.pdf  — UK54 node17 delta_t vs delta_P (raw4)

import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base      = Path(__file__).resolve().parent
NET_ROOT   = _base.parent / "02_network" / "99_test_UK_54"
NET_DIR    = NET_ROOT / "network"
INIT_DIR   = NET_ROOT / "init"
CSV_RAW1   = NET_ROOT / "result_raw1" / "54_analysis0_n_17_Kr_1.00.csv"
CSV_RAW3   = NET_ROOT / "result_raw3" / "54_analysis0_n_17_Kr_1.00.csv"
CSV_RAW4   = NET_ROOT / "result_raw4" / "54_analysis0_n_17_Kr_1.00.csv"
OUT_DIR    = _base / "fig_4"
OUT_DIR.mkdir(exist_ok=True)

# ── 파라미터 ────────────────────────────────────────────────────
NODE_NUM    = 54
SAMPLE_NUM  = 0
KR          = 1.00
ATTACK_NODE = 17

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
# fig4a — 고장 모델 개념도 + P_i(t) 인셋
# ================================================================
def fig4a():
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    dt_fault = 3.5
    P_star   = 0.28
    dP       = 0.38
    fs       = plt.rcParams['axes.labelsize'] * 2  # 2배

    # 신호선
    ax.plot([-0.15, 0],              [P_star, P_star],       color='black', lw=2.2)
    ax.plot([0, dt_fault],          [P_star+dP, P_star+dP], color='#D94946', lw=2.2)
    ax.plot([dt_fault, dt_fault+1.4], [P_star, P_star],     color='black', lw=2.2)

    # 수직 점선 (t=0, t=Δt): P_i^*+ΔP → 0 (t축까지)
    ls_v = dict(color='gray', lw=1.2, ls=(0, (4, 3)))
    ax.plot([0, 0],               [0, P_star+dP], **ls_v)
    ax.plot([dt_fault, dt_fault], [0, P_star+dP], **ls_v)

    # 가로 점선 (P_i^* 기준): 0 ~ Δt
    ls_h = dict(color='gray', lw=1.2, ls=(0, (4, 3)))
    ax.plot([0, dt_fault], [P_star, P_star], **ls_h)

    # Δt_i 수평 양방향 화살표
    ax.annotate('', xy=(dt_fault, P_star+dP+0.09),
                xytext=(0, P_star+dP+0.09),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax.text(dt_fault/2, P_star+dP+0.14, r'$\Delta t_i$',
            ha='center', va='bottom', fontsize=fs)

    x_max = dt_fault + 1.7
    y_max = P_star + dP + 0.32
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(0, y_max)

    # spine 사용: left/bottom만, zero에서 만나게
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 화살표 끝 (spine 끝에 맞게)
    ax.plot(x_max, 0, '>k', markersize=6, clip_on=False)
    ax.plot(0, y_max, '^k', markersize=6, clip_on=False)

    # ΔP_i 라벨 (Δt 위치 오른쪽, 수직)
    ax.annotate('', xy=(dt_fault+0.2, P_star+dP),
                xytext=(dt_fault+0.2, P_star),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
    ax.text(dt_fault+0.4, (P_star+P_star+dP)/2, r'$\Delta P_i$',
            ha='left', va='center', fontsize=fs)

    # 라벨
    ax.text(-0.18, P_star,    r'$P_i^*$',                 ha='right', va='center', fontsize=fs)
    ax.text(0,        -0.04, r'$0$',          ha='center', va='top', fontsize=fs)
    # ax.text(dt_fault, -0.04, r'$\Delta t_i$', ha='center', va='top', fontsize=fs)
    ax.text(x_max+0.05, -0.04, r'$t$',        ha='left',   va='top', fontsize=fs)
    ax.text(-0.1, y_max, r'$P_i(t)$',          ha='right',   va='top', fontsize=fs)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4a.pdf', dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('Saved: fig4a.pdf')



# ================================================================
# fig4b — delta_t vs delta_P (raw4) + analytic E lines + T/4 line
# ================================================================
def fig4b(P, M, D, K, theta_eq):
    # raw4 CSV
    df4 = pd.read_csv(CSV_RAW4)
    print(f"  raw4: {len(df4)} rows")

    # E_safe_max, E_danger_min from raw1
    E_safe_max = E_danger_min = np.nan
    print(CSV_RAW1)
    if CSV_RAW1.exists():
        df2   = pd.read_csv(CSV_RAW1)
        last2 = df2.iloc[-1]
        E_safe_max   = float(last2.get('E_safe_max',   np.nan))
        E_danger_min = float(last2.get('E_danger_min', np.nan))
    print(f"  E_safe_max={E_safe_max:.4f}  E_danger_min={E_danger_min:.4f}")

    # T/4 계산: ω_d = sqrt(k_i/M_i - (D_i/2M_i)²)
    M_i   = M[ATTACK_NODE]
    D_i   = D[ATTACK_NODE]
    k_i   = sum(K[ATTACK_NODE, j] * np.cos(theta_eq[ATTACK_NODE] - theta_eq[j])
                for j in range(NODE_NUM) if K[ATTACK_NODE, j] > 0)
    omega_n  = np.sqrt(k_i / M_i) if k_i > 0 else None
    alpha_   = D_i / (2.0 * M_i)
    quarter_T = None
    if omega_n and omega_n > alpha_:
        omega_d   = np.sqrt(omega_n**2 - alpha_**2)
        quarter_T = PI / (2.0 * omega_d)
        print(f"  T/4 = {quarter_T:.4f} s")

    # t 범위: 0 ~ T/4
    t_max = quarter_T if quarter_T else df4['attack_t'].max()

    # 카테고리
    has_safe    = 'safe' in df4.columns
    mask_sync   = df4['num_synced'] >= NODE_NUM
    mask_desync = df4['num_synced'] < NODE_NUM
    mask_safe   = mask_sync & (df4['safe'] == 1) if has_safe else mask_sync
    mask_danger = mask_sync & (df4['safe'] != 1) if has_safe else pd.Series([False]*len(df4))

    # t 범위 필터
    mask_t = df4['attack_t'] <= t_max

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.set_xscale('log')

    # scatter
    for mask, color in [
        (mask_desync & mask_t, COLOR_DESYNC),
        (mask_danger & mask_t, COLOR_DANGER),
        (mask_safe   & mask_t, COLOR_SAFE),
    ]:
        if mask.any():
            ax.scatter(df4.loc[mask, 'attack_t'],
                       df4.loc[mask, 'attack_P'],
                       c=color, s=20, linewidths=0, rasterized=True, alpha=0.7)

    # analytic curves: ΔP = sqrt(2E/M) * D / (1 - exp(-D/M * Δt))
    t_arr = np.logspace(np.log10(df4['attack_t'].min()+1e-6),
                        np.log10(t_max), 500)
    exp_term = np.exp(-D_i / M_i * t_arr)
    denom    = np.where(1.0 - exp_term < 1e-12, 1e-12, 1.0 - exp_term)

    for E_val, color, label in [
        (E_safe_max,   COLOR_SAFE_DARK,    r'$E_{17}^{\mathrm{U}}$'),
        (E_danger_min, COLOR_EDANGER_LINE, r'$E_{17}^{\mathrm{S}}$'),
    ]:
        if np.isfinite(E_val) and E_val > 0:
            dP = np.sqrt(2.0 * E_val / M_i) * D_i / denom
            ax.plot(t_arr,  dP, color=color, lw=2.4, zorder=5, label=label)
            ax.plot(t_arr, -dP, color=color, lw=2.4, zorder=5)

    # y축: symlog (양수+음수, 중간 linear 구간 좁게)
    p_all = df4.loc[mask_t, 'attack_P'].dropna()
    y_abs_max = np.percentile(np.abs(p_all), 98) * 1.2 if len(p_all) else 10
    p_min_nonzero = np.abs(p_all[p_all != 0]).min() if (p_all != 0).any() else 1e-2
    linthresh = p_min_nonzero * 0.5
    ax.set_yscale('symlog', linthresh=linthresh, linscale=0.5)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(-y_abs_max, y_abs_max)
    # 최소 눈금과 0 겹침 제거
    ax.figure.canvas.draw()  # tick 확정
    yticks = [t for t in ax.get_yticks()
              if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    pos_ticks = sorted([t for t in yticks if t > 0])
    neg_ticks = sorted([t for t in yticks if t < 0])
    zero      = [t for t in yticks if t == 0]
    if pos_ticks: pos_ticks = pos_ticks[1:]   # 가장 작은 양수 1개 제거
    if neg_ticks: neg_ticks = neg_ticks[:-1]  # 가장 큰 음수 1개 제거
    ax.set_yticks(zero + pos_ticks + neg_ticks)
    ax.set_xlim(1e-3, t_max)

    ax.set_xlabel(r'Disturbance duration $\Delta t_i$ (sec, log)')
    ax.set_ylabel(r'Power disturbance $\Delta P_i$ (symlog)')

    legend_handles = [
        mpatches.Patch(color=COLOR_SAFE,   label='sync (secured)'),
        mpatches.Patch(color=COLOR_DANGER, label='sync (unsecured)'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync (unsecured)'),
        Line2D([0],[0], color=COLOR_EDANGER_LINE, lw=2.4,
               label=r'$E_{17}^{\mathrm{S}}=' + f'{E_danger_min:.3f}$'),
        Line2D([0],[0], color=COLOR_SAFE_DARK, lw=2.4,
               label=r'$E_{17}^{\mathrm{U}}=' + f'{E_safe_max:.3f}$'),
    ]
    leg = ax.legend(handles=legend_handles, loc='center right',
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
    fig.savefig(OUT_DIR / 'fig4b.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig4b.pdf')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("--- fig4a (fault model schematic) ---")
    fig4a()

    print("\nLoading network...")
    P, M, D, K, theta_eq = load_network()

    print("\n--- fig4b (delta_t vs delta_P) ---")
    fig4b(P, M, D, K, theta_eq)

    print("\nAll done. Output in:", OUT_DIR)