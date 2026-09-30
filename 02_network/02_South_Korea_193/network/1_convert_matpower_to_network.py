# convert_matpower_to_network.py
#
# Purpose:
#   Convert a MATPOWER .m file (KPG193_ver1_5.m) into simulation-ready
#   network files:
#     network/193_sample0_nodes.txt   (id  P_i  mass  gamma)
#     network/193_sample0_edges.txt   (u   v    K_ij)
#   And optionally a GML file for visualization:
#     network/193_sample0.gml
#
# P_i   = (sum of Pg at bus - Pd) / baseMVA  [p.u.]
# K_ij  = 1 / x_ij  (susceptance, per-unit)
#          If parallel branches exist between the same pair, K_ij are summed.
# mass  = default 1.0  (override in MASS_DEFAULT or per-bus in MASS_OVERRIDE)
# gamma = default 0.1  (override in GAMMA_DEFAULT or per-bus in GAMMA_OVERRIDE)
#
# Usage:
#   python3 convert_matpower_to_network.py
#
# Dependencies: none (stdlib only)

import re
import os
import csv
from pathlib import Path
from collections import defaultdict

_here = Path(__file__).resolve().parent   # = network/
_base = _here.parent                       # = project root

# ===== 사용자 설정 =====
M_FILE       = _here / "KPG193_ver1_5.m"
LOC_FILE     = _here / "bus_location.csv"   # optional (for GML)
OUT_DIR      = _here                         # nodes/edges.txt → network/
SAMPLE_NUM   = 0
N_EXPECTED   = 193

BASE_MVA     = None    # None = read from file

# mass / gamma defaults (applied to all buses unless overridden)
MASS_DEFAULT  = 0.0  # 부하 버스: impute_mass_from_power.py로 별도 추정
GAMMA_DEFAULT = 0.1

# Per-bus overrides: {original_bus_id: value}  (1-indexed matpower bus ids)
MASS_OVERRIDE  = {}   # e.g. {3: 2.5, 8: 1.8}
GAMMA_OVERRIDE = {}   # e.g. {3: 0.5}

# Branch filter: skip branches where status != 1
FILTER_OFFLINE_BRANCHES = True

# Zero-injection buses: include or skip
# (buses where |P_i| < ZERO_TOL are kept as-is)
ZERO_TOL = 1e-9
# ======================


def parse_matpower(path):
    with open(path) as f:
        content = f.read()

    # baseMVA
    base_mva_match = re.search(r'mpc\.baseMVA\s*=\s*([\d.]+)', content)
    base_mva = float(base_mva_match.group(1)) if base_mva_match else 100.0

    # ---- bus data ----
    # columns: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    bus_block = re.search(r'mpc\.bus\s*=\s*\[(.*?)\];', content, re.DOTALL).group(1)
    buses = {}
    for line in bus_block.strip().splitlines():
        line = re.sub(r';.*', '', line).strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 3:
            continue
        bus_i = int(vals[0])
        Pd    = float(vals[2])   # MW load demand
        buses[bus_i] = {'Pd': Pd, 'Pg': 0.0}

    # ---- gen data ----
    # columns: bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin ...
    gen_block = re.search(r'mpc\.gen\s*=\s*\[(.*?)\];', content, re.DOTALL).group(1)
    for line in gen_block.strip().splitlines():
        line = re.sub(r';.*', '', line).strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 8:
            continue
        bus_i  = int(vals[0])
        Pg     = float(vals[1])
        status = int(float(vals[7]))
        if status != 1:
            continue
        if bus_i in buses:
            buses[bus_i]['Pg'] += Pg

    # ---- branch data ----
    # columns: fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
    branch_block = re.search(r'mpc\.branch\s*=\s*\[(.*?)\];', content, re.DOTALL).group(1)
    branches = []
    for line in branch_block.strip().splitlines():
        line = re.sub(r';.*', '', line).strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 11:
            continue
        fbus   = int(vals[0])
        tbus   = int(vals[1])
        x      = float(vals[3])
        status = int(float(vals[10]))
        if FILTER_OFFLINE_BRANCHES and status != 1:
            continue
        if abs(x) < 1e-12:
            print(f"  WARNING: branch {fbus}-{tbus} has near-zero reactance x={x}, skipped")
            continue
        K_ij = 1.0 / abs(x)
        branches.append((fbus, tbus, K_ij))

    return base_mva, buses, branches


def load_locations(path):
    """Returns dict: bus_id (1-indexed) -> (lat, lon)"""
    locs = {}
    if not path.exists():
        return locs
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bus_id = int(row['bus_id'])
            lat    = float(row['Latitude'])
            lon    = float(row['Longitude'])
            locs[bus_id] = (lat, lon)
    return locs


def build_node_list(buses, base_mva):
    """
    Returns list of (new_id_0indexed, orig_bus_id, P_i, mass, gamma)
    sorted by original bus id.
    """
    nodes = []
    sorted_buses = sorted(buses.keys())
    for new_id, orig_id in enumerate(sorted_buses):
        b  = buses[orig_id]
        Pi = (b['Pg'] - b['Pd']) / base_mva   # p.u.
        m  = MASS_OVERRIDE.get(orig_id, MASS_DEFAULT)
        g  = GAMMA_OVERRIDE.get(orig_id, GAMMA_DEFAULT)
        nodes.append((new_id, orig_id, Pi, m, g))
    return nodes


def build_edge_list(branches, orig_to_new):
    """
    Aggregate parallel branches (sum K_ij).
    Returns list of (u_new, v_new, K_ij_total), undirected (u < v).
    """
    edge_K = defaultdict(float)
    skipped = 0
    for fbus, tbus, K in branches:
        if fbus not in orig_to_new or tbus not in orig_to_new:
            skipped += 1
            continue
        u = orig_to_new[fbus]
        v = orig_to_new[tbus]
        if u > v:
            u, v = v, u
        edge_K[(u, v)] += K
    if skipped:
        print(f"  WARNING: {skipped} branches reference buses not in bus list (skipped)")
    return [(u, v, K) for (u, v), K in sorted(edge_K.items())]


def write_nodes(path, nodes):
    with open(path, 'w') as f:
        for new_id, orig_id, Pi, m, g in nodes:
            f.write(f"{new_id} {Pi:.15f} {m:.6f} {g:.6f}\n")
    print(f"  Written: {path}  ({len(nodes)} nodes)")


def write_edges(path, edges):
    with open(path, 'w') as f:
        for u, v, K in edges:
            f.write(f"{u} {v} {K:.15f}\n")
    print(f"  Written: {path}  ({len(edges)} edges)")


def write_gml(path, nodes, edges, locs, orig_ids):
    """Write GML file with geographic positions if available."""
    has_loc = bool(locs)
    with open(path, 'w') as f:
        f.write("graph [\n")
        f.write("  directed 0\n")
        for new_id, orig_id, Pi, m, g in nodes:
            f.write("  node [\n")
            f.write(f"    id {new_id}\n")
            f.write(f"    label \"{orig_id}\"\n")
            f.write(f"    power {Pi:.15f}\n")
            f.write(f"    mass {m:.6f}\n")
            f.write(f"    gamma {g:.6f}\n")
            if has_loc and orig_id in locs:
                lat, lon = locs[orig_id]
                f.write(f"    pos {lon:.6f}\n")
                f.write(f"    pos {lat:.6f}\n")
            f.write("  ]\n")
        for u, v, K in edges:
            f.write("  edge [\n")
            f.write(f"    source {u}\n")
            f.write(f"    target {v}\n")
            f.write(f"    weight {K:.15f}\n")
            f.write("  ]\n")
        f.write("]\n")
    print(f"  Written: {path}  (GML{'+ locations' if has_loc else ', no locations'})")


def summary(nodes, edges, base_mva):
    Pi_vals = [Pi for _, _, Pi, _, _ in nodes]
    K_vals  = [K for _, _, K in edges]
    gen_nodes   = sum(1 for Pi in Pi_vals if Pi > ZERO_TOL)
    load_nodes  = sum(1 for Pi in Pi_vals if Pi < -ZERO_TOL)
    zero_nodes  = sum(1 for Pi in Pi_vals if abs(Pi) <= ZERO_TOL)
    print(f"\n  baseMVA          : {base_mva}")
    print(f"  Nodes total      : {len(nodes)}")
    print(f"    Generator (P>0): {gen_nodes}")
    print(f"    Load      (P<0): {load_nodes}")
    print(f"    Zero injection : {zero_nodes}")
    print(f"  Edges (unique)   : {len(edges)}")
    print(f"  P_i range        : [{min(Pi_vals):.4f}, {max(Pi_vals):.4f}] p.u.")
    print(f"  K_ij range       : [{min(K_vals):.2f}, {max(K_vals):.2f}]")
    print(f"  sum(P_i)         : {sum(Pi_vals):.6f} p.u.  (should be ~0)")


def main():
    print("=" * 52)
    print("convert_matpower_to_network.py")
    print("=" * 52)
    print(f"Input  : {M_FILE}")
    print(f"LocFile: {LOC_FILE}")
    print(f"Output : {OUT_DIR}/193_sample{SAMPLE_NUM}_{{nodes,edges}}.txt")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse
    base_mva, buses, branches = parse_matpower(M_FILE)
    if BASE_MVA is not None:
        base_mva = BASE_MVA
    print(f"  Parsed: {len(buses)} buses, {len(branches)} active branches")

    if len(buses) != N_EXPECTED:
        print(f"  WARNING: expected {N_EXPECTED} buses, got {len(buses)}")

    # Load locations
    locs = load_locations(LOC_FILE)
    if locs:
        print(f"  Locations loaded: {len(locs)} buses")
    else:
        print(f"  No location file found — GML will have no coordinates")

    # Build
    nodes = build_node_list(buses, base_mva)
    orig_ids = [orig_id for _, orig_id, _, _, _ in nodes]
    orig_to_new = {orig: new for new, orig, _, _, _ in nodes}
    edges = build_edge_list(branches, orig_to_new)

    # Summary
    summary(nodes, edges, base_mva)
    print()

    # Write
    node_path = OUT_DIR / f"193_sample{SAMPLE_NUM}_nodes.txt"
    edge_path = OUT_DIR / f"193_sample{SAMPLE_NUM}_edges.txt"
    gml_path  = OUT_DIR / f"193_sample{SAMPLE_NUM}.gml"

    write_nodes(node_path, nodes)
    write_edges(edge_path, edges)
    write_gml(gml_path, nodes, edges, locs, orig_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()