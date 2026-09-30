# impute_mass_from_power.py
#
# Purpose:
#   M=0, D=0 인 노드(load/변전소)에 대해 P 만으로 D와 M을 추정한다.
#   덴마크(convert_denmark_to_network.py), KPG193(convert_matpower_to_network.py)
#   처럼 발전기에는 gen_inertia(M>0)가 있지만 부하 버스는 M=D=0인 경우에 사용.
#
# 방법:
#   PanTaGruEl 4개국 (CH, DE, ES, UK) 데이터에서 도출한 회귀 계수 사용.
#
#   Step 1. D 추정 (P 부호에 따라 별도 회귀)
#     D_gen  = A_POS * |P|   (P > 0, 발전기 버스)   R² = 0.52~0.86
#     D_load = A_NEG * |P|   (P < 0, 수요 버스)      R² = 0.79~0.999
#
#   Step 2. M 추정 (gen 노드의 M~D 회귀)
#     M = C_MD * D_estimated                          R² = 0.92~0.99
#
#   국가별 계수:
#     CH : a_pos=0.008261  a_neg=0.004779  c_MD=4.3274
#     DE : a_pos=0.009329  a_neg=0.004782  c_MD=5.0546
#     ES : a_pos=0.008439  a_neg=0.004779  c_MD=4.9817
#     UK : a_pos=0.008940  a_neg=0.004805  c_MD=4.8428
#     평균: a_pos=0.008742  a_neg=0.004786  c_MD=4.8016
#     => M_pos_imputed ≈ 0.04198 * |P|
#     => M_neg_imputed ≈ 0.02298 * |P|
#
# Usage:
#   from impute_mass_from_power import impute_mass_power
#
#   nodes_txt = "173_sample0_nodes.txt"   # id  P  mass  gamma
#   impute_mass_power(nodes_txt, nodes_txt, country='mean')
#
# 또는 단독 실행:
#   python3 impute_mass_from_power.py   (상단 설정 참고)

from pathlib import Path
import numpy as np

# ===== 사용자 설정 =====
INPUT_FILE  = Path(__file__).resolve().parent / "193_sample0_nodes.txt"
OUTPUT_FILE = Path(__file__).resolve().parent / "193_sample0_nodes.txt"  # 덮어쓰기
COUNTRY     = "mean"   # "CH" | "DE" | "ES" | "UK" | "mean"

# M_FLOOR: 추정값이 이 값보다 작으면 강제로 올림
M_FLOOR = 1e-4
# ======================


# -----------------------------------------------------------------------
# 4개국 회귀 계수 (PanTaGruEl 데이터 기반)
# -----------------------------------------------------------------------
COEFFS = {
    #           a_pos      a_neg      c_MD
    "CH":  (0.008261,  0.004779,  4.3274),
    "DE":  (0.009329,  0.004782,  5.0546),
    "ES":  (0.008439,  0.004779,  4.9817),
    "UK":  (0.008940,  0.004805,  4.8428),
    "mean":(0.008742,  0.004786,  4.8016),
}


def impute_mass_power(input_path, output_path,
                      country="mean",
                      m_floor=1e-4,
                      verbose=True):
    """
    nodes.txt 파일을 읽어 M=0, D=0 노드에 D와 M을 추정해서 저장한다.

    Parameters
    ----------
    input_path  : str or Path — 입력 nodes.txt (id  P  mass  gamma)
    output_path : str or Path — 출력 nodes.txt (동일 포맷)
    country     : str — 회귀 계수 선택 ("CH"|"DE"|"ES"|"UK"|"mean")
    m_floor     : float — 최소 mass 하한
    verbose     : bool — 진행 상황 출력

    Returns
    -------
    nodes : list of [new_id, P, mass_new, gamma_new]
    stats : dict — 요약 통계
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if country not in COEFFS:
        raise ValueError(f"country must be one of {list(COEFFS.keys())}")
    a_pos, a_neg, c_MD = COEFFS[country]

    # ---- 읽기 ----
    nodes = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            nodes.append([int(v[0]), float(v[1]), float(v[2]), float(v[3])])
            # [id, P, mass, gamma]

    n_total   = len(nodes)
    gen_mask  = [r[2] > 1e-9 for r in nodes]   # M > 0 → 발전기
    n_gen     = sum(gen_mask)
    n_impute  = n_total - n_gen

    if verbose:
        print(f"  Input       : {input_path.name}  ({n_total} nodes)")
        print(f"  Country     : {country}  "
              f"(a_pos={a_pos:.6f}, a_neg={a_neg:.6f}, c_MD={c_MD:.4f})")
        print(f"  Gen (M>0)   : {n_gen}  → 그대로 유지")
        print(f"  Load (M=0)  : {n_impute}  → D·M 추정")
        print(f"  M_FLOOR     : {m_floor}")

    # ---- 추정 ----
    n_pos = n_neg = n_zero_p = n_floored = 0
    for r in nodes:
        if r[2] > 1e-9:          # gen: 그대로
            continue
        P = r[1]
        absP = abs(P)
        if absP < 1e-9:
            n_zero_p += 1
            r[3] = m_floor        # gamma
            r[2] = m_floor        # mass
            continue

        if P > 0:
            D_est = a_pos * absP
            n_pos += 1
        else:
            D_est = a_neg * absP
            n_neg += 1

        M_est = c_MD * D_est

        if M_est < m_floor:
            M_est = m_floor
            n_floored += 1
        if D_est < m_floor:
            D_est = m_floor

        r[2] = M_est   # mass
        r[3] = D_est   # gamma

    # ---- 통계 ----
    all_M = np.array([r[2] for r in nodes])
    all_D = np.array([r[3] for r in nodes])
    gen_M = all_M[np.array(gen_mask)]
    load_M = np.array([r[2] for r, g in zip(nodes, gen_mask) if not g])

    stats = dict(
        n_total=n_total, n_gen=n_gen, n_impute=n_impute,
        n_pos=n_pos, n_neg=n_neg, n_zero_p=n_zero_p, n_floored=n_floored,
        M_gen_range=(float(gen_M.min()), float(gen_M.max())) if n_gen else (0,0),
        M_load_range=(float(load_M.min()), float(load_M.max())) if n_impute else (0,0),
        M_all_range=(float(all_M.min()), float(all_M.max())),
    )

    if verbose:
        print(f"  Imputed P>0 : {n_pos} nodes")
        print(f"  Imputed P<0 : {n_neg} nodes")
        print(f"  P≈0 (floor) : {n_zero_p} nodes")
        print(f"  M_FLOOR hit : {n_floored} nodes")
        print(f"  Gen  M range: [{stats['M_gen_range'][0]:.5f}, "
              f"{stats['M_gen_range'][1]:.5f}]")
        print(f"  Load M range: [{stats['M_load_range'][0]:.5f}, "
              f"{stats['M_load_range'][1]:.5f}]")
        print(f"  All  M range: [{stats['M_all_range'][0]:.5f}, "
              f"{stats['M_all_range'][1]:.5f}]")

    # ---- 쓰기 ----
    with open(output_path, 'w') as f:
        for nid, P, mass, gamma in nodes:
            f.write(f"{nid} {P:.15f} {mass:.6f} {gamma:.6f}\n")

    if verbose:
        print(f"  Output      : {output_path.name}")

    return nodes, stats


# -----------------------------------------------------------------------
# 단독 실행
# -----------------------------------------------------------------------

def main():
    print("=" * 56)
    print("impute_mass_from_power.py")
    print("=" * 56)
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"Country: {COUNTRY}  (회귀 계수 선택)")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}\n"
            "INPUT_FILE 경로를 수정하거나 라이브러리로 import해서 사용하세요."
        )

    nodes, stats = impute_mass_power(
        INPUT_FILE, OUTPUT_FILE,
        country=COUNTRY,
        m_floor=M_FLOOR,
        verbose=True,
    )

    print()
    print("Notes:")
    print("  회귀 계수 출처: PanTaGruEl 4개국 (CH, DE, ES, UK)")
    print("  Step 1: D = a_pos*|P| (P>0) 또는 a_neg*|P| (P<0)")
    print("  Step 2: M = c_MD * D")
    print("  기존 M>0 노드(발전기)는 변경되지 않습니다.")
    print("\nDone.")


if __name__ == "__main__":
    main()
