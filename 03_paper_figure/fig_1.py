# fig_1.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_1/
#   fig1b.pdf  — Transient path (max link angle difference)
#   fig1c.pdf  — Basin stability view
#   fig1d.pdf  — Basin safety view

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pandas as pd
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base      = Path(__file__).resolve().parent
NET_ROOT   = _base.parent / "02_network" / "01_motif_3"
CSV_PATH   = NET_ROOT / "result_raw2" / "3_analysis0_n_0_Kr_10.00.csv"
NODE_FILE  = NET_ROOT / "network" / "3_sample0_nodes.txt"
EDGE_FILE  = NET_ROOT / "network" / "3_sample0_edges.txt"
INIT_FILE  = NET_ROOT / "init"    / "3_sample0_init_Kr_10.00.txt"
OUT_DIR    = _base / "fig_1"
OUT_DIR.mkdir(exist_ok=True)

# ── 파라미터 ────────────────────────────────────────────────────
T_SIM = 50.0
EARLY_FRAC_1A = 0.3   # fig1a: 전체 T_SIM 중 앞부분 이 비율만큼만 궤적으로 표시
TRAJ_LINEWIDTH = 1.0  # fig1a 궤적 선 굵기 (기존 2.2 대비 약 2.5배 얇게)
SYNC_OMEGA_TOL = 1.0  # 동기화 판정 시 |ω| 허용오차

PERTURBATIONS = [
    (-0.5 * np.pi, 3.0),
    (-0.5 * np.pi, 9.25),
    (-0.5 * np.pi, 15.5),
]
LABELS = ['A', 'B', 'C']

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

PI           = np.pi
W_MAX        = 20.0
KR           = 10.0
N            = 3
COLOR_SAFE   = '#4C9BE8'
COLOR_DANGER = '#E8954C'
COLOR_DESYNC = '#555555'
COLOR_LIMIT  = '#F5CBA7'
COLOR_NODE0  = '#D94946'
COLOR_NODE_OTHER = '#aaaaaa'


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
    return P, M, D, K, theta_eq


# ================================================================
# RK4 시뮬레이터
# ================================================================
def swing_rhs(state, P, M, D, K):
    theta = state[:N]; omega = state[N:]
    dtheta = omega.copy()
    domega = np.zeros(N)
    for i in range(N):
        coupling = np.sum(K[i] * np.sin(theta - theta[i]))
        domega[i] = (P[i] - D[i]*omega[i] + coupling) / M[i]
    return np.concatenate([dtheta, domega])

def rk4_step(state, dt, P, M, D, K):
    k1 = swing_rhs(state,           P, M, D, K)
    k2 = swing_rhs(state + dt/2*k1, P, M, D, K)
    k3 = swing_rhs(state + dt/2*k2, P, M, D, K)
    k4 = swing_rhs(state + dt*k3,   P, M, D, K)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def simulate(P, M, D, K, theta_eq, dth, dom, perturb_node=0,
             T_end=100.0, dt=1e-3, theta_c=PI):
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
        state = rk4_step(state, dt, P, M, D, K)
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

    synced = np.all(np.abs(state[N:]) < SYNC_OMEGA_TOL)
    return np.array(t_list), np.array(max_link_list), crossed, synced


def status_color(crossed, synced):
    if synced:
        status = 'sync (unsecured)' if crossed else 'sync (secured)'
    else:
        status = 'desync (unsecured)'
    color = {'sync (secured)': COLOR_SAFE,
             'sync (unsecured)': COLOR_DANGER,
             'desync (unsecured)': COLOR_DESYNC}[status]
    return status, color


# ================================================================
# 노드0 실제 에너지 계산 (fig_2.py 로직 재사용, motif_3용)
# ================================================================
def potential_diff_0(theta_trial, theta_eq, P, K):
    target = 0
    dth = (theta_trial - theta_eq[target] + PI) % (2*PI) - PI
    V_trial = V_eq = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if K[i, j] > 0:
                if i == target or j == target:
                    nb = j if i == target else i
                    V_trial -= K[target, nb] * np.cos(theta_trial      - theta_eq[nb])
                    V_eq    -= K[target, nb] * np.cos(theta_eq[target] - theta_eq[nb])
    V_trial -= P[target] * dth
    return V_trial - V_eq


def energy_0(theta_pert, omega_pert, theta_eq, P, M, K):
    """노드0의 위상섭동/각속도섭동 (Δθ, Δω)에서의 실제 에너지."""
    V = potential_diff_0(theta_eq[0] + theta_pert, theta_eq, P, K)
    T = 0.5 * M[0] * omega_pert**2
    return V + T


# ================================================================
# 노드0의 (Δθ, Δω) 궤적을 조밀하게 추적 (fig1a용)
# ================================================================
def simulate_dense_trace(P, M, D, K, theta_eq, dth, dom, T_show, dt=1e-3, ref_node=1):
    """노드0의 (Δθ, Δω) 궤적. 위상각은 ref_node(기본 노드1)에 대한 상대각으로
    표시한다 — 즉 ref_node의 위상각을 프레임으로 고정."""
    n_steps = int(T_show / dt)
    theta0 = theta_eq.copy(); omega0 = np.zeros(N)
    theta0[0] += dth; omega0[0] += dom
    state = np.concatenate([theta0, omega0])

    eq_rel = theta_eq[0] - theta_eq[ref_node]

    def rel_angle(th):
        d = (th[0] - th[ref_node]) - eq_rel
        return (d + PI) % (2*PI) - PI

    save_every = max(1, n_steps // 2000)
    d0_list = [rel_angle(theta0)]
    w0_list = [omega0[0]]

    for step in range(n_steps):
        state = rk4_step(state, dt, P, M, D, K)
        if (step + 1) % save_every == 0:
            th = state[:N]; om0 = state[N]
            d0_list.append(rel_angle(th))
            w0_list.append(om0)

    return np.array(d0_list), np.array(w0_list)




# ================================================================
# fig1a — Conceptual basin of attraction schematic
# ================================================================
def fig1a(P, M, D, K, theta_eq):
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for spine in ax.spines.values():
        spine.set_zorder(150)
    ax.set_facecolor('white')

    x = np.linspace(-PI, PI, 500)
    y = np.linspace(-W_MAX, W_MAX, 500)
    X, Y = np.meshgrid(x, y)

    # 실제 에너지 지형 (fig_2.py와 동일한 potential_diff + kinetic 계산)
    Z = energy_0(X, Y, theta_eq, P, M, K)

    # 원점(평형점)이 가장 낮음 → 정규화 후 0=흰색, 1=진한 초록
    z_min = Z.min()
    z_max = np.percentile(Z, 92)   # 너무 먼 고에너지 영역까지 늘어지지 않게 상위 percentile로 클립
    Z_sym = (Z - z_min) / max(z_max - z_min, 1e-6)
    Z_sym = np.clip(Z_sym, 0, 1)

    cmap_terrain = LinearSegmentedColormap.from_list(
        'summer_green', [
            '#ffffff',
            '#f1f8e9',
            '#dcedc8',
            '#c5e1a5',
            '#aed581',
            '#8bc34a',
            '#7cb342',
        ], N=512)
    levels_fill = np.linspace(0, 1, 80)
    ax.contourf(X, Y, Z_sym, levels=levels_fill,
                cmap=cmap_terrain, alpha=0.60, zorder=0)
    ax.contour(X, Y, Z_sym, levels=15,
               colors='#33691e', linewidths=0.4, alpha=0.25, zorder=1)

    # 평형점 (원점)
    ax.plot(0, 0, '.', ms=10, mew=2.5, color='#2c3e50', zorder=7)

    # ── 실제 시뮬레이션 궤적 A, B, C ─────────────────────────────
    T_show = T_SIM * EARLY_FRAC_1A
    for (dth, dom), lbl in zip(PERTURBATIONS, LABELS):
        # 분류(색상)는 전체 T_SIM 기준 시뮬레이션으로 판정
        _, _, crossed, synced = simulate(P, M, D, K, theta_eq, dth, dom,
                                         perturb_node=0, T_end=T_SIM)
        status, color = status_color(crossed, synced)
        print(f"  trajectory {lbl}: {status}  crossed={crossed}  synced={synced}")

        # 표시용 궤적은 앞부분 T_show 구간만 조밀하게 추적
        d0_arr, w0_arr = simulate_dense_trace(P, M, D, K, theta_eq, dth, dom, T_show)

        # 주기 경계(±π)를 넘나드는 지점에서 선이 끊기도록 세그먼트 분리
        jumps = np.where(np.abs(np.diff(d0_arr)) > PI)[0]
        seg_bounds = [0] + list(jumps + 1) + [len(d0_arr)]
        for si in range(len(seg_bounds) - 1):
            s0, s1 = seg_bounds[si], seg_bounds[si+1]
            if s1 - s0 < 2:
                continue
            ax.plot(d0_arr[s0:s1], w0_arr[s0:s1], color=color, lw=TRAJ_LINEWIDTH, zorder=5)

    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0',
                        r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle difference $(\theta_0-\theta_1)$')
    ax.set_ylabel(r'angular-velocity $\omega_0$')

    legend_handles = [
        plt.Line2D([0],[0], color=COLOR_SAFE,   lw=TRAJ_LINEWIDTH,
                   marker='>', markersize=7, label='A: sync (secured)'),
        plt.Line2D([0],[0], color=COLOR_DANGER, lw=TRAJ_LINEWIDTH,
                   marker='>', markersize=7, label='B: sync (unsecured)'),
        plt.Line2D([0],[0], color=COLOR_DESYNC, lw=TRAJ_LINEWIDTH,
                   marker='>', markersize=7, label='C: desync (unsecured)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right',
              framealpha=0.92, edgecolor='#aaaaaa', borderpad=0.7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1a.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig1a.pdf')



def fig1b(P, M, D, K, theta_eq):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for spine in ax.spines.values():
        spine.set_zorder(150)

    results  = []
    for (dth, dom), lbl in zip(PERTURBATIONS, LABELS):
        t_arr, max_link, crossed, synced = simulate(
            P, M, D, K, theta_eq, dth, dom, perturb_node=0, T_end=T_SIM)
        status, col = status_color(crossed, synced)
        results.append((lbl, t_arr, max_link, crossed, status, col))
        print(f"  {lbl}: {status}  crossed={crossed}  synced={synced}")

    # y축 최대값: A, B만 기준으로 (C는 desync라 값이 매우 커서 화면을 지배함)
    ylim_ref = [r for r in results if r[0] in ('A', 'B')] or results
    ylim_top = max(max(r[2].max() for r in ylim_ref) * 1.1, PI * 1.3)
    ylim_top = max(ylim_top, PI + 1.0)

    # limit exceeded 음영 (π 위)
    ax.fill_between([0, T_SIM], PI, ylim_top,
                    color=COLOR_LIMIT, alpha=0.40, zorder=0)

    for lbl, t_arr, max_link, crossed, status, col in results:
        ax.plot(t_arr, max_link, color=col, lw=2.2,
                label=f'{lbl}: {status}')

    # angle limit π
    ax.axhline(PI, color='black', lw=1.5, ls='--', zorder=4,
               label=r'phase-angle limit $\pi$')

    ax.set_xlim(0, T_SIM)
    ax.set_ylim(0, ylim_top)

    # x축 정수 눈금
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # y축 π 단위 눈금
    pi_max = int(np.ceil(ylim_top / PI))
    pi_ticks = [k * PI for k in range(pi_max + 1) if k * PI <= ylim_top * 1.01]
    pi_labels = []
    for k in range(pi_max + 1):
        if k * PI > ylim_top * 1.01: break
        if k == 0:   pi_labels.append('0')
        elif k == 1: pi_labels.append(r'$\pi$')
        else:        pi_labels.append(rf'${k}\pi$')
    ax.set_yticks(pi_ticks)
    ax.set_yticklabels(pi_labels)

    ax.set_xlabel('time (sec)', size=14)
    ax.set_ylabel(r'absolute phase-angle diff $|\theta_0 - \theta_1|$', size=14)
    # ax.set_title('b  Transient path decides safety', loc='left', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ── 네트워크 인셋 (오른쪽, π선 바로 위) ────────────────────
    # 인셋: axes 좌표 [x0, y0, width, height]
    # π / ylim_top = π선의 y 위치 비율
    pi_frac = PI / ylim_top
    inset_h = 0.22
    inset_y = pi_frac + 0.02   # π선 바로 위
    ax_in = ax.inset_axes([0.62, inset_y, 0.36, inset_h])
    ax_in.set_facecolor('none')
    ax_in.set_xlim(-0.6, 2.6)
    ax_in.set_ylim(-0.55, 0.45)
    ax_in.axis('off')

    # 체인: 1 - 0 - 2 (수평, 순서대로 x=0,1,2이지만 노드0을 가운데)
    node_pos = {0: (1, 0), 1: (0, 0), 2: (2, 0)}
    node_cols = {0: COLOR_NODE0, 1: COLOR_NODE_OTHER, 2: COLOR_NODE_OTHER}

    # edges: 1-0, 0-2
    for u, v in [(1, 0), (0, 2)]:
        x0, y0 = node_pos[u]; x1, y1 = node_pos[v]
        ax_in.plot([x0, x1], [y0, y1], color='#888', lw=2.0, zorder=1)

    for i in range(N):
        x, y = node_pos[i]
        ax_in.scatter(x, y, s=500, c=node_cols[i],
                      edgecolors='black', linewidths=1.2, zorder=3)
        ax_in.text(x, y, str(i), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', zorder=4)
        ax_in.text(x, -0.38, f'P={P[i]:+.1f}',
                   ha='center', va='top', fontsize=8, color='#444')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1b.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig1b.pdf')


# ================================================================
# fig1c — Basin stability view  (배경 흰색, desync 점 표시)
# ================================================================
def fig1c(df):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for spine in ax.spines.values():
        spine.set_zorder(150)
    ax.set_facecolor('white')
 
    mask_desync = df['num_synced'] < N
    mask_sync   = df['num_synced'] >= N
 
    ax.scatter(df.loc[mask_desync, 'attack_theta'],
               df.loc[mask_desync, 'attack_w'],
               c=COLOR_DESYNC, s=20, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(df.loc[mask_sync, 'attack_theta'],
               df.loc[mask_sync, 'attack_w'],
               c='#71bd58', s=20, linewidths=0, rasterized=True, zorder=3)
 
    # A, B 포인트: 별 모양
    for (dth, dom), lbl in zip(PERTURBATIONS, LABELS):
        ax.scatter(dth, dom, s=800, c='#D94946', zorder=8,
                   edgecolors='black', linewidths=1.5, marker='*')
        ax.text(dth, dom, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=9)
 
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0',
                        r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_0$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_0$')
    legend_handles = [
        mpatches.Patch(color='#71bd58',    label='sync'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1c.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig1c.pdf')
 
 
# ================================================================
# fig1d — Basin safety view  (배경 흰색, desync 점 표시)
# ================================================================
def fig1d(df):
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for spine in ax.spines.values():
        spine.set_zorder(150)
    ax.set_facecolor('white')
 
    has_safe    = 'safe' in df.columns
    mask_desync = df['num_synced'] < N
    mask_sync   = df['num_synced'] >= N
    if has_safe:
        mask_safe   = mask_sync & (df['safe'] == 1)
        mask_danger = mask_sync & (df['safe'] != 1)
    else:
        mask_safe   = mask_sync
        mask_danger = pd.Series([False]*len(df))
 
    ax.scatter(df.loc[mask_desync, 'attack_theta'],
               df.loc[mask_desync, 'attack_w'],
               c=COLOR_DESYNC, s=20, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(df.loc[mask_danger, 'attack_theta'],
               df.loc[mask_danger, 'attack_w'],
               c=COLOR_DANGER, s=20, linewidths=0, rasterized=True, zorder=3)
    ax.scatter(df.loc[mask_safe, 'attack_theta'],
               df.loc[mask_safe, 'attack_w'],
               c=COLOR_SAFE, s=20, linewidths=0, rasterized=True, zorder=4)
 
    # A, B 포인트: 별 모양
    for (dth, dom), lbl in zip(PERTURBATIONS, LABELS):
        ax.scatter(dth, dom, s=800, c='#D94946', zorder=8,
                   edgecolors='black', linewidths=1.5, marker='*')
        ax.text(dth, dom, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=9)
 
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-W_MAX, W_MAX)
    ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0',
                        r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.set_xlabel(r'phase-angle perturbation $\Delta\theta_0$')
    ax.set_ylabel(r'angular-velocity perturbation $\Delta\omega_0$')
    legend_handles = [
        mpatches.Patch(color=COLOR_SAFE,   label='sync (secured)'),
        mpatches.Patch(color=COLOR_DANGER, label='sync (unsecured)'),
        mpatches.Patch(color=COLOR_DESYNC, label='desync (unsecured)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1d.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig1d.pdf')
 
 
# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
 
    print("Loading network...")
    P, M, D, K, theta_eq = load_network()
    print(f"  P={P}  theta_eq={np.degrees(theta_eq).round(2)} deg")
 
    print("\n--- fig1b (transient) ---")
    fig1b(P, M, D, K, theta_eq)
 
    print("\n--- fig1a (conceptual schematic) ---")
    fig1a(P, M, D, K, theta_eq)
 
    print("\n--- fig1c (basin stability) ---")
    fig1c(df)
 
    print("\n--- fig1d (basin safety) ---")
    fig1d(df)
 
    # ── 요약 통계: sync 비율, safe 비율 (전체 점 개수 대비) ────────
    total       = len(df)
    n_sync      = int((df['num_synced'] >= N).sum())
    has_safe    = 'safe' in df.columns
    n_safe      = int(((df['num_synced'] >= N) & (df['safe'] == 1)).sum()) if has_safe else None
 
    sync_ratio = n_sync / total if total else float('nan')
    safe_ratio = n_safe / total if (has_safe and total) else float('nan')
 
    print("\n--- Summary ---")
    print(f"  Total points : {total}")
    print(f"  sync points  : {n_sync}  ({sync_ratio:.4%})")
    if has_safe:
        print(f"  safe points  : {n_safe}  ({safe_ratio:.4%})")
    else:
        print("  safe points  : N/A (no 'safe' column)")
 
    print("\nAll done. Output in:", OUT_DIR)