# fig_preprocess.py
# 각 데이터셋 폴더의 result_raw1/ 를 순회하여 result/0_rank.csv 를 (재)생성한다.
# 시간이 오래 걸리는 계산을 미리 끝내서, 이후 fig*.py 들은 이 CSV만 읽으면 되도록 한다.
#
# 출력 컬럼:
#   node, basin_stability, basin_safety, safe_danger_ratio,
#   rank_basin_stability, rank_basin_safety, rank_safe_danger_ratio,
#   energy_safety_mean, energy_safety_dev, rank_energy_safety_mean
#
# 주의: raw1 CSV의 컬럼 E_safe_max / E_danger_min 은 이름과 달리
#   E_safe_max   -> E^danger   (safe outcome 중 최댓값)
#   E_danger_min -> E^safety   (danger outcome 중 최솟값)
# 논문 정의를 따른다.
#
# ★ UEP 배제구간 처리 ★
# attack_random_server.c가 이제 UEP(local mean-field로 계산한 불안정평형점)
# 반대쪽(먼 쪽) 절반의 theta 구간은 애초에 샘플링하지 않는다(쉐이딩 영역).
# 이 배제구간은 "에너지 임계값(E_safety, E_danger)" 계산에만 적용한다:
#   - basin_stability, basin_safety는 전체 데이터(쉐이딩 영역 포함) 그대로 계산
#   - E_safety/E_danger(및 파생값인 safe_danger_ratio, energy_safety_mean/dev)는
#     노드별 UEP로 배제구간을 판정해 그 구간의 행을 뺀 뒤, 남은 행들의
#     E_temp에서 직접 재계산한다 (CSV의 누적 컬럼 E_safe_max/E_danger_min은
#     구버전 데이터라면 배제구간 샘플의 영향을 받았을 수 있어 신뢰하지 않는다)

import os
import glob
import re as _re
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ── 경로 ────────────────────────────────────────────────────────
_base    = Path(__file__).resolve().parent
NET_ROOT = _base.parent / "02_network"

PI = np.pi
SAMPLE_NUM = 0
KR = 1.00

# ── 데이터셋 정의 (label, folder) ─────────────────────────────────
DATASETS = [
    ("Korea 193",  "02_South_Korea_193"),
    ("UK 54",      "04_UK_54"),
    ("UK 313",     "05_UK_313"),
    ("Swiss 161",  "06_Swiss_161"),
    ("Germany 553","07_Germany_553"),
    ("Spain 838",  "08_Spain_838"),
    ("France 847", "09_France_847"),
]


def _node_count(net_dir):
    """네트워크 폴더 안 *_sample0_nodes.txt 파일명에서 노드 수 프리픽스 추출"""
    candidates = list(net_dir.glob("*_sample0_nodes.txt"))
    if not candidates:
        return None
    return int(candidates[0].name.split('_')[0])


# ================================================================
# 네트워크 로더 (P, K 인접리스트, theta_eq) — UEP 계산용
# ================================================================
def load_network(net_dir_parent, node_num):
    net_dir  = net_dir_parent / "network"
    init_dir = net_dir_parent / "init"

    P = np.zeros(node_num)
    with open(net_dir / f"{node_num}_sample{SAMPLE_NUM}_nodes.txt") as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            i = int(parts[0])
            if len(parts) > 1: P[i] = float(parts[1])

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

    return P, adj, theta_eq


def wrap_to_pi(x):
    return (x + PI) % (2*PI) - PI


def compute_uep_dtheta(target, theta_eq, P, adj):
    """Local mean-field 방법으로 UEP의 delta-theta 계산 (attack_random_server.c와 동일 로직).
    반환: (uep_dtheta, valid). valid=False면 계산 불가(배제구간 없음)."""
    sum_cos = 0.0; sum_sin = 0.0
    for nb, kij in adj[target]:
        sum_cos += kij * np.cos(theta_eq[nb])
        sum_sin += kij * np.sin(theta_eq[nb])

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


def process_dataset(label, folder):
    print(f"\n=== {label} ({folder}) ===")

    net_dir  = NET_ROOT / folder / "network"
    raw1_dir = NET_ROOT / folder / "result_raw1"
    out_dir  = NET_ROOT / folder / "result"
    out_dir.mkdir(exist_ok=True, parents=True)

    node_count = _node_count(net_dir)
    if node_count is None:
        print(f"  WARNING: cannot determine node count for {folder}, skip")
        return

    files = sorted(glob.glob(str(raw1_dir / "*_n_*_Kr_*.csv")))
    if not files:
        print(f"  WARNING: no raw1 CSV files found in {raw1_dir}, skip")
        return

    print("  Loading network (for UEP exclusion)...")
    P, adj, theta_eq = load_network(NET_ROOT / folder, node_count)

    n_excluded_total = 0
    rows = []
    for fp in tqdm(files, desc=f"  {label}", unit="node", leave=False):
        m = _re.search(r'_n_(\d+)_', os.path.basename(fp))
        if not m:
            continue
        n = int(m.group(1))
        if n >= node_count:
            continue

        try:
            df_full = pd.read_csv(fp)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df_full.empty:
            continue

        # ── basin_stability, basin_safety: 전체 데이터(쉐이딩 영역 포함) 기준 ──
        if 'num_synced' not in df_full.columns:
            continue
        total = len(df_full)
        mask_sync = df_full['num_synced'] >= node_count
        n_sync = int(mask_sync.sum())
        basin_stability = n_sync / total if total else np.nan

        if 'safe' in df_full.columns:
            n_safe = int((mask_sync & (df_full['safe'] == 1)).sum())
            basin_safety = n_safe / total if total else np.nan
        else:
            basin_safety = np.nan

        # ── UEP 배제구간 필터링: 에너지 계산에서만 이 구간의 행을 제외 ──
        uep_dtheta, uep_valid = compute_uep_dtheta(n, theta_eq, P, adj)
        excl_range = excluded_theta_range(uep_dtheta, uep_valid)
        df_energy = df_full
        if excl_range is not None and 'attack_theta' in df_full.columns:
            before = len(df_full)
            mask_excl = (df_full['attack_theta'] > excl_range[0]) & (df_full['attack_theta'] < excl_range[1])
            df_energy = df_full.loc[~mask_excl]
            n_excluded_total += before - len(df_energy)

        # ── E_safety, E_danger: 배제구간을 뺀 df_energy의 E_temp에서 직접 재계산 ──
        # (CSV의 누적 컬럼 E_safe_max/E_danger_min은 배제구간 샘플의 영향을
        #  받았을 수 있는 구버전 데이터일 가능성이 있어 신뢰하지 않는다)
        if not df_energy.empty and 'E_temp' in df_energy.columns:
            mask_sync_e = df_energy['num_synced'] >= node_count
            if 'safe' in df_energy.columns:
                mask_safe_ok_e = mask_sync_e & (df_energy['safe'] == 1)
            else:
                mask_safe_ok_e = mask_sync_e
            e_temp = df_energy['E_temp'].values.astype(float)
            safe_vals   = e_temp[mask_safe_ok_e.values]
            danger_vals = e_temp[(~mask_safe_ok_e).values]
            e_danger = float(np.max(safe_vals))   if len(safe_vals)   else np.nan
            e_safety = float(np.min(danger_vals)) if len(danger_vals) else np.nan
        else:
            # E_temp 컬럼이 없는 구버전 CSV는 최종행의 누적값으로 폴백
            last = df_full.iloc[-1]
            e_safety = float(last['E_danger_min']) if 'E_danger_min' in df_full.columns else np.nan
            e_danger = float(last['E_safe_max'])   if 'E_safe_max'   in df_full.columns else np.nan

        if np.isfinite(e_safety) and np.isfinite(e_danger) and e_safety > 0:
            safe_danger_ratio = e_danger / e_safety
            m_ES   = np.sqrt(e_safety * e_danger)
            sig_ES = np.sqrt(e_danger / e_safety)
        else:
            safe_danger_ratio = np.nan
            m_ES   = np.nan
            sig_ES = np.nan

        rows.append({
            'node': n,
            'basin_stability': basin_stability,
            'basin_safety': basin_safety,
            'safe_danger_ratio': safe_danger_ratio,
            'energy_safety_mean': m_ES,
            'energy_safety_dev': sig_ES,
        })

    if not rows:
        print(f"  WARNING: no valid rows produced for {folder}, skip")
        return

    out_df = pd.DataFrame(rows).sort_values('node').reset_index(drop=True)

    # ── rank 계산: 값이 클수록 rank=1 (내림차순) ────────────────
    for col in ['basin_stability', 'basin_safety', 'safe_danger_ratio', 'energy_safety_mean']:
        rank_col = f'rank_{col}'
        out_df[rank_col] = out_df[col].rank(ascending=False, method='min', na_option='bottom').astype(int)

    # 최종 컬럼 순서 정리
    out_df = out_df[[
        'node', 'basin_stability', 'basin_safety', 'safe_danger_ratio',
        'rank_basin_stability', 'rank_basin_safety', 'rank_safe_danger_ratio',
        'energy_safety_mean', 'energy_safety_dev', 'rank_energy_safety_mean',
    ]]

    out_fp = out_dir / "0_rank.csv"
    out_df.to_csv(out_fp, index=False)
    print(f"  Saved: {out_fp}  ({len(out_df)} nodes)")
    if n_excluded_total:
        print(f"  (UEP 배제구간으로 총 {n_excluded_total}개 행을 통계 계산에서 제외함)")


if __name__ == '__main__':
    n_datasets = len(DATASETS)
    for idx, (label, folder) in enumerate(DATASETS, start=1):
        print(f"[{idx}/{n_datasets}] ({idx/n_datasets:.0%}) {label}")
        process_dataset(label, folder)

    print("\nAll done.")