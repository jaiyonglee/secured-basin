# fig_5.py
# 출력: 2026_Safe_Zone/03_paper_figure/fig_5/
#   fig5a.pdf  — Basin stability vs Basin safety
#   fig5b.pdf  — Basin stability vs Energy safety mean (log, with error bars)
#   fig5c.pdf  — Basin safety vs Energy safety mean (log, no error bars)
#   fig5d.pdf  — Basin safety vs Energy safety deviation (log, no error bars)
#   fig5e.pdf  — Disparity 히스토그램: 좌(기하 σ_ES) vs 우(산술 Δ), x축 log

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# ── 경로 ────────────────────────────────────────────────────────
_base    = Path(__file__).resolve().parent
NET_ROOT = _base.parent / "02_network"
OUT_DIR  = _base / "fig_5"
OUT_DIR.mkdir(exist_ok=True)

# ── 데이터셋 정의 ────────────────────────────────────────────────
DATASETS = [
    ("04_UK_54",           "UK 54"),
    ("06_Swiss_161",       "Swiss 161"),
    ("02_South_Korea_193", "Korea 193"),
    ("05_UK_313",          "UK 313"),
    ("07_Germany_553",     "Germany 553"),
    ("08_Spain_838",       "Spain 838"),
    ("09_France_847",      "France 847"),
]

# ── 스타일 ──────────────────────────────────────────────────────
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

# ── 파라미터 ────────────────────────────────────────────────────
# True: ratio 축을 log 스케일로, r도 log 변환 후 피어슨
# False: 선형 스케일, r도 원래값 기준 피어슨
LOG_RATIO = True

# 특정 노드 강조 표시: (DATASETS의 인덱스, 노드 번호)
# 해당 노드만 검은색 + 2배 굵기로 점/에러바를 그린다.
ENABLE_HIGHLIGHT = False    # False로 하면 강조 표시 끔
HIGHLIGHT_NODE = (0, 17)   # 기본값: UK 54 (DATASETS[0]), node 17

COLORS  = ['#4C9BE8', '#E8954C', '#53AB38', '#D94946',
           '#9B59B6', '#16A085', '#E67E22']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


# ================================================================
# 데이터 로드
# ================================================================
def load_all():
    dfs = []
    for folder, label in DATASETS:
        fp = NET_ROOT / folder / "result" / "0_rank.csv"
        if not fp.exists():
            print(f"  WARNING: {fp} not found, skip")
            continue
        df = pd.read_csv(fp)
        df['dataset'] = label
        df['inv_safe_danger_ratio'] = np.where(
            df['safe_danger_ratio'] > 0,
            1 / df['safe_danger_ratio'],
            np.nan
        )
        dfs.append((label, df))
        print(f"  Loaded {label}: {len(df)} nodes")
    return dfs




# ================================================================
# 단일 scatter + 추세선 플랏 함수
# ================================================================
def scatter_plot(ax, dfs, xcol, ycol, xlabel, ylabel, log_y=False, exclude_one=False):
    all_x, all_y = [], []
    n_dfs = len(dfs)

    for idx, ((label, df), color) in enumerate(zip(dfs, COLORS)):
        cols = [xcol, ycol] + (['node'] if 'node' in df.columns else [])
        sub = df[cols].dropna()
        if sub.empty:
            continue
        if exclude_one:
            # basin_stability 또는 basin_safety가 1인 행 제외
            for col in ['basin_stability', 'basin_safety']:
                if col in sub.columns:
                    sub = sub[sub[col] < 1.0]
        x, y = sub[xcol].values, sub[ycol].values
        if log_y:
            valid = y > 0
            sub = sub[valid]
            x, y = x[valid], y[valid]

        # ── 데이터셋별 Pearson correlation ──
        valid_corr = np.isfinite(x) & np.isfinite(y)

        if valid_corr.sum() > 1:
            x_corr = x[valid_corr]
            y_corr = y[valid_corr]

            if log_y:
                y_corr = np.log10(y_corr)

            r_dataset = np.corrcoef(x_corr, y_corr)[0, 1]

            print(
                f"  {label:<12s}: "
                f"r = {r_dataset: .3f}, "
                f"R^2 = {r_dataset**2:.3f}, "
                f"N = {len(x_corr)}"
            )

        # DATASETS 순서대로 맨 앞이 맨 위에 오도록 zorder 부여
        z = 3 + (n_dfs - 1 - idx)
        ax.scatter(x, y, c=color, s=6, alpha=0.8,
                   edgecolors='white', linewidths=0.15,
                   marker='o', label=label, zorder=z)

        # 지정된 노드 강조 (검은색, 2배 굵기/크기, 항상 맨 위)
        if ENABLE_HIGHLIGHT and HIGHLIGHT_NODE is not None and HIGHLIGHT_NODE[0] == idx and 'node' in sub.columns:
            hl = sub[sub['node'] == HIGHLIGHT_NODE[1]]
            if not hl.empty:
                ax.scatter(hl[xcol].values, hl[ycol].values, c='black', s=12,
                          alpha=1.0, edgecolors='white', linewidths=0.3,
                          marker='o', zorder=100)

        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    if log_y:
        ax.set_yscale('log')

    # 전체 추세선 + 상관계수
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    mask  = np.isfinite(all_x) & np.isfinite(all_y)
    if mask.sum() > 1:
        _x = all_x[mask]
        _y = np.log10(all_y[mask]) if log_y else all_y[mask]
        r  = np.corrcoef(_x, _y)[0, 1]
        z  = np.polyfit(_x, _y, 1)
        p  = np.poly1d(z)
        x_line = np.linspace(_x.min(), _x.max(), 100)
        y_line = 10**p(x_line) if log_y else p(x_line)
        ax.plot(x_line, y_line, color='#333333', lw=1.8,
                ls='--', zorder=50,
                label=f'$r={r:.3f}$, $R^2={r**2:.3f}$')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


# ================================================================
# fig5a — Basin stability vs Basin safety
# ================================================================
def fig5a(dfs):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    scatter_plot(ax, dfs,
                 xcol='basin_stability', ycol='basin_safety',
                 xlabel=r'Basin stability $p_i^\mathbf{BS}$',
                 ylabel=r'Relative secured-basin volume $p_i^\mathbf{S}$',
                 log_y=False)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='upper left', framealpha=0.88, ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5a.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig5a.pdf')


# ================================================================
# fig5b/c 공용 — X vs Energy safety (log, with error bars)
# ================================================================
def energy_safety_plot(dfs, xcol, xlabel, out_name):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    all_x, all_logm = [], []
    n_dfs = len(dfs)

    for idx, ((label, df), color) in enumerate(zip(dfs, COLORS)):
        sub = df[['node', xcol, 'energy_safety_mean', 'energy_safety_dev']].dropna()
        sub = sub[(sub['energy_safety_mean'] > 0) & (sub['energy_safety_dev'] > 0)]
        if sub.empty:
            continue

        x_v   = sub[xcol].values
        m_v   = sub['energy_safety_mean'].values
        sig_v = sub['energy_safety_dev'].values
        node_v = sub['node'].values

        # ── 데이터셋별 Pearson correlation ──
        valid_corr = np.isfinite(x_v) & np.isfinite(m_v) & (m_v > 0)

        if valid_corr.sum() > 1:
            x_corr = x_v[valid_corr]
            logm_corr = np.log10(m_v[valid_corr])

            r_dataset = np.corrcoef(x_corr, logm_corr)[0, 1]

            print(
                f"  {label:<12s}: "
                f"r = {r_dataset: .3f}, "
                f"R^2 = {r_dataset**2:.3f}, "
                f"N = {len(x_corr)}"
            )

        # 사전계산된 m, sigma로부터 E^safety, E^danger 역산
        es_v = m_v / sig_v
        ed_v = m_v * sig_v

        # DATASETS 순서대로 맨 앞이 맨 위에 오도록 zorder 부여
        z_err = 1 + (n_dfs - 1 - idx)
        z_pt  = 3 + (n_dfs - 1 - idx)

        # 에러바: 옅은 회색, 얇게 (데이터가 많아 겹침이 심하므로)
        for i in range(len(x_v)):
            ax.plot([x_v[i], x_v[i]], [es_v[i], ed_v[i]],
                    color='#999999', lw=0.3, alpha=0.75, zorder=z_err)

        ax.scatter(x_v, m_v, c=color, s=6, alpha=0.8,
                   edgecolors='white', linewidths=0.15,
                   marker='o', label=label, zorder=z_pt)

        # 지정된 노드 강조 (검은색, 2배 굵기/크기 에러바+점, 항상 맨 위)
        if ENABLE_HIGHLIGHT and HIGHLIGHT_NODE is not None and HIGHLIGHT_NODE[0] == idx:
            hl_mask = node_v == HIGHLIGHT_NODE[1]
            if hl_mask.any():
                ax.plot([x_v[hl_mask][0], x_v[hl_mask][0]],
                        [es_v[hl_mask][0], ed_v[hl_mask][0]],
                        color='black', lw=0.6, alpha=1.0, zorder=100)
                ax.scatter(x_v[hl_mask], m_v[hl_mask], c='black', s=12,
                          alpha=1.0, edgecolors='white', linewidths=0.3,
                          marker='o', zorder=101)

        all_x.extend(x_v.tolist())
        all_logm.extend(np.log10(m_v).tolist())

    ax.set_yscale('log')

    # 전체 회귀 (log10(m_ES) 기준)
    all_x    = np.array(all_x)
    all_logm = np.array(all_logm)
    mask = np.isfinite(all_x) & np.isfinite(all_logm)
    if mask.sum() > 1:
        _x, _y = all_x[mask], all_logm[mask]
        r  = np.corrcoef(_x, _y)[0, 1]
        z  = np.polyfit(_x, _y, 1)
        p  = np.poly1d(z)
        x_line = np.linspace(_x.min(), _x.max(), 100)
        ax.plot(x_line, 10**p(x_line), color='#333333', lw=1.8,
                ls='--', zorder=50,
                label=f'$r={r:.3f}$, $R^2={r**2:.3f}$')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Mean barrier height $\overline{E}_i$ (log)')
    ax.grid(True, alpha=0.2)

    # legend: 점 + 세로선(에러바)을 함께 보여주는 errorbar 프록시 핸들
    legend_handles = []
    for (label, df), color in zip(dfs, COLORS):
        h = ax.errorbar([], [], yerr=1, fmt='o', color=color, markersize=4,
                        markeredgecolor='white', markeredgewidth=0.3,
                        elinewidth=0.8, capsize=0, alpha=0.9, label=label)
        legend_handles.append(h)
    reg_line_handles = [ln for ln in ax.get_lines() if ln.get_label().startswith('$r=')]
    legend_handles.extend(reg_line_handles)
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.88, ncol=1, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{out_name}.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_name}.pdf')


def fig5b(dfs):
    energy_safety_plot(dfs, xcol='basin_stability',
                       xlabel=r'Basin stability $p_i^\mathbf{BS}$', out_name='fig5b')


# ================================================================
# fig5c — Basin safety vs Energy safety mean (log, no error bars)
# ================================================================
def fig5c(dfs):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    scatter_plot(ax, dfs,
                 xcol='basin_safety', ycol='energy_safety_mean',
                 xlabel=r'Relative secured-basin volume $p_i^\mathbf{S}$',
                 ylabel=r'Mean barrier height $\overline{E}_i$ (log)',
                 log_y=True)
    ax.legend(loc='upper left', framealpha=0.88, ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5c.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig5c.pdf')


# ================================================================
# fig5d — Basin safety vs Energy safety deviation (log, no error bars)
# ================================================================
def fig5d(dfs):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    scatter_plot(ax, dfs,
                 xcol='basin_safety', ycol='energy_safety_dev',
                 xlabel=r'Relative secured-basin volume $p_i^\mathbf{S}$',
                 ylabel=r'Barrier height disparity $\Delta_{i}$ (log)',
                 log_y=True)
    ax.legend(loc='upper right', framealpha=0.88, ncol=1, fontsize=8)
    ax.set_ylim(top=1.1e1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5d.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig5d.pdf')


# ================================================================
# fig5e — disparity 정의 비교: 기하(σ_ES) vs 산술(Δ) 히스토그램
# ================================================================
def fig5e(dfs):
    m_geo_all   = []   # 기하평균: m_ES = sqrt(E_safety*E_danger)
    m_arith_all = []   # 산술평균: (E_safety+E_danger)/2
    sigma_all   = []   # 기하 disparity: σ_ES = sqrt(E_danger/E_safety)
    diff_all    = []   # 산술 disparity: Δ = (E_danger - E_safety)/2

    for label, df in dfs:
        sub = df[['energy_safety_mean', 'energy_safety_dev']].dropna()
        sub = sub[(sub['energy_safety_mean'] > 0) & (sub['energy_safety_dev'] > 0)]
        if sub.empty:
            continue
        m     = sub['energy_safety_mean'].values
        sigma = sub['energy_safety_dev'].values
        # 저장된 m(기하평균), sigma(곱셈형 편차)로부터 E_safety, E_danger 역산
        E_safety = m / sigma
        E_danger = m * sigma
        m_arith = (E_safety + E_danger) / 2.0
        diff    = (E_danger - E_safety) / 2.0

        m_geo_all.extend(m[m > 0].tolist())
        m_arith_all.extend(m_arith[m_arith > 0].tolist())
        sigma_all.extend(sigma[sigma > 0].tolist())
        diff_all.extend(diff[diff > 0].tolist())

    m_geo_all   = np.array(m_geo_all)
    m_arith_all = np.array(m_arith_all)
    sigma_all   = np.array(sigma_all)
    diff_all    = np.array(diff_all)
    print(f"  m_ES (geo mean):   {len(m_geo_all)} nodes  (range {m_geo_all.min():.4g} ~ {m_geo_all.max():.4g})")
    print(f"  mean (arith):      {len(m_arith_all)} nodes  (range {m_arith_all.min():.4g} ~ {m_arith_all.max():.4g})")
    print(f"  sigma_ES:          {len(sigma_all)} nodes  (range {sigma_all.min():.4g} ~ {sigma_all.max():.4g})")
    print(f"  Delta:             {len(diff_all)} nodes  (range {diff_all.min():.4g} ~ {diff_all.max():.4g})")

    def _hist(ax, data, color, xlabel):
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 40)
        ax.hist(data, bins=bins, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.2)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))

    # 위: 평균 (좌=기하, 우=산술)
    _hist(axes[0, 0], m_geo_all, COLORS[0],
         r'Geometric mean $m_i^{\mathrm{ES}}$ (log)')
    _hist(axes[0, 1], m_arith_all, COLORS[1],
         r'Arithmetic mean $\overline{E}_i$ (log)')

    # 아래: disparity (좌=기하, 우=산술)
    _hist(axes[1, 0], sigma_all, COLORS[0],
         r'Geometric disparity $\sigma_i^{\mathrm{ES}}$ (log)')
    _hist(axes[1, 1], diff_all, COLORS[1],
         r'Arithmetic disparity $\Delta_i$ (log)')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5e.pdf', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: fig5e.pdf')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("Loading datasets...")
    dfs = load_all()
    print(f"  Total: {len(dfs)} datasets\n")

    print("--- fig5a (basin stability vs basin safety) ---")
    fig5a(dfs)

    print("--- fig5b (basin stability vs energy safety) ---")
    fig5b(dfs)

    print("--- fig5c (basin safety vs energy safety mean) ---")
    fig5c(dfs)

    print("--- fig5d (basin safety vs energy safety deviation) ---")
    fig5d(dfs)

    print("--- fig5e (disparity: geometric vs arithmetic histogram) ---")
    fig5e(dfs)

    print("\nAll done. Output in:", OUT_DIR)