#!/usr/bin/env python3
"""
MemProtMD-style per-chain topology from MemPrO (and MemPrOD) output.

Reproduces, without a CG simulation, what the MemProtMD pipeline produced from one:
the 2D secondary structure "arrow" diagram, the transmembrane strand/helix counts and
the membrane lines (the z histograms of each secondary structure class are in the JSON).

It is a Python 3 port of three mpm_jobs scripts:

* build_ss_topology.py: the arrow diagram (JSON key "topology").
* postprocess.py: transmembrane spans, chain placement and z histograms (keys "chains",
  "surface_residues", "z_histograms") and the per-residue leaflet heights (key
  "residue_leaflet_zs").
* calculate_contacts.py: lipid head/tail and solvent contacts. There is no trajectory
  here, so these are REPLACED BY A GEOMETRIC APPROXIMATION, see below.

Inputs:

* MemPrO's oriented_rank_N.pdb (atomistic; z along the membrane normal, membrane centre
  at z = 0, or at +-d/2 for -dm) and info_rank_N.txt (membrane thickness, -dm distance).
* Optionally a MemPrOD output folder. Its LocData gives the height of the upper and
  lower membrane surface on a grid, used for the bent membrane lines. Without it the
  lines are flat, at MemPrO's boundaries.

Differences from the reference, all recorded under "approximations" in the JSON:

* Contacts. A residue's solvent accessible surface (Shrake-Rupley, 1.4 A probe) is split
  by where it points: inside the membrane slab and reachable from outside the protein
  within the membrane plane it counts as lipid contact (head group contact within 5 A of
  a boundary, tail contact deeper); inside the slab but enclosed by protein it counts as
  solvent (a pore or cavity); outside the slab it counts as solvent. A residue is "in
  contact" where that part is at least --exposure-cutoff of its maximum accessibility.
  These stand in for the contact frequencies in the span "contact" type (lipid / solvent
  / protein) and in the chain placement type.
* Membrane boundaries are MemPrO's flat ones (+-thickness/2), where the reference used
  the mean height of the lipids around the protein in the simulation.
* The Bezier control points of each element are a least squares fit (deterministic)
  where the reference used a random search; at most 16 control points are used.
* Each element after the first of a segment starts at the last residue of the element
  before it, so that neighbouring elements join. The reference added the element's own
  first residue twice (same array lengths), which looks like a slip for this.
* A chain that starts (or resumes after a gap) within the middle half of the membrane is a span only if
  it crosses the midplane before leaving; the reference counted every such stretch.
* Residue indexes ("resIdxs", "surface_residues") count from 0 along "alignment".

Usage:
    mempro-topology -f Orient/Rank_1/oriented_rank_1.pdb -d Deformations/ -o Topology
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

__version__ = "0.1"

# ---------------------------------------------------------------------------
# Constants shared with the reference scripts
# ---------------------------------------------------------------------------

AA_RESNAMES = {
    "ILE": "I", "GLN": "Q", "GLY": "G", "MSE": "M", "GLU": "E", "CYS": "C", "ASP": "D",
    "LYS": "K", "PRO": "P", "ASN": "N", "VAL": "V", "THR": "T", "HIS": "H", "TRP": "W",
    "SER": "S", "PHE": "F", "ALA": "A", "MET": "M", "LEU": "L", "ARG": "R", "TYR": "Y",
}

# Maximum solvent accessibility of each residue (A^2; Tien et al. 2013, theoretical)
MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0, "GLN": 225.0,
    "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "MSE": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0, "THR": 172.0,
    "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

DSSP_SS_CODE_SHEET = "E"
DSSP_SS_CODE_HELIX_ALPHA = "H"
DSSP_SS_CODE_HELIX_310 = "G"
DSSP_SS_CODE_HELIX_PI = "I"
DSSP_SS_CODE_STRAND = "B"
DSSP_SS_CODE_BEND = "S"
DSSP_SS_CODE_COIL = "O"
DSSP_SS_CODE_BETA_TURN = "T"
DSSP_SS_CODE_NONE = ""

SPECIFIC_SMOOTHING_WINDOWS = {
    DSSP_SS_CODE_SHEET: 4.0,
    DSSP_SS_CODE_HELIX_ALPHA: 3.6,
    DSSP_SS_CODE_HELIX_310: 3.0,
    DSSP_SS_CODE_HELIX_PI: 4.1,
    DSSP_SS_CODE_STRAND: 4.0,
    DSSP_SS_CODE_BEND: 3.0,
    DSSP_SS_CODE_COIL: 3.0,
    DSSP_SS_CODE_BETA_TURN: 3.0,
    DSSP_SS_CODE_NONE: 3.0,
}

SS_LOOP = "ssNone"
SS_HELIX = "ssAlpha"
SS_STRAND = "ssBeta"

SS_SIMPLIFY_MAP = {
    DSSP_SS_CODE_SHEET: SS_STRAND,
    DSSP_SS_CODE_HELIX_ALPHA: SS_HELIX,
    DSSP_SS_CODE_HELIX_310: SS_HELIX,
    DSSP_SS_CODE_HELIX_PI: SS_HELIX,
    DSSP_SS_CODE_STRAND: SS_STRAND,
    DSSP_SS_CODE_BEND: SS_LOOP,
    DSSP_SS_CODE_COIL: SS_LOOP,
    DSSP_SS_CODE_BETA_TURN: SS_LOOP,
    DSSP_SS_CODE_NONE: SS_LOOP,
}

# (residues straightened at each end, residues the end's direction is taken from)
SPECIFIC_2D_EDGE_CANCELLATION = {
    SS_LOOP: (0, 1),
    SS_HELIX: (5, 4),
    SS_STRAND: (1, 2),
}

# Shorter helices and strands are drawn as loops
MIN_LENGTHS = {SS_LOOP: 3, SS_HELIX: 5, SS_STRAND: 4}

HEAD_WIDTH = 5.0       # A; MemPrO's head group region at each membrane boundary
MAX_BASIS_POINTS = 16


def warn(msg):
    print("WARNING: " + msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Reading MemPrO output
# ---------------------------------------------------------------------------

def read_pdb(path):
    """
    Reads the protein of a PDB file (first model, first alternate location, no hydrogens).
    Returns (residues, dummy_z): residues have chainID, resSeq, iCode, resName and atoms
    (name -> (element, xyz)); dummy_z are the z of MemPrO's dummy membrane beads (DUM).
    """
    residues, index, dummy_z = [], {}, []
    n_cg = 0
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM") or len(line) < 54:
                continue
            resname = line[17:21].strip()
            name = line[12:16].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if resname == "DUM":
                dummy_z.append(xyz[2])
                continue
            if resname == "PGL" or (rec == "HETATM" and resname != "MSE"):
                continue
            if line[16] not in " A1":
                continue
            if name in ("BB", "SC1"):
                n_cg += 1
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element or not element.isalpha():
                element = re.sub("[^A-Za-z]", "", name)[:1].upper()
            if element in ("H", "D") or re.match(r"^\d?H", name):
                continue
            key = (line[21], line[22:27])
            if key not in index:
                index[key] = len(residues)
                residues.append({
                    "chainID": line[21] if line[21] != " " else "_",
                    "resSeq": int(line[22:26]),
                    "iCode": line[26].strip(),
                    "resName": resname,
                    "atoms": {},
                })
            residues[index[key]]["atoms"].setdefault(name, (element, np.array(xyz)))
    complete = [r for r in residues if all(a in r["atoms"] for a in ("N", "CA", "C"))]
    if len(complete) == 0:
        if n_cg > 0:
            raise SystemExit("ERROR: " + path + " is coarse grained. DSSP needs the backbone atoms: "
                             "orient the atomistic structure with MemPrO, or put the atomistic structure in the oriented frame.")
        raise SystemExit("ERROR: No protein residues with N, CA and C atoms in " + path)
    if len(complete) < len(residues):
        warn("%d residues without N, CA and C atoms were left out" % (len(residues) - len(complete)))
    return complete, dummy_z


def read_info(path):
    """Membrane thicknesses, -dm distance and curvature from MemPrO's info_rank_N.txt."""
    info = {}
    patterns = {
        "distance": r"Iter-Membrane distance[^:]*:\s*([-+0-9.eE]+)",
        "curvature": r"Global curvature[^:]*:\s*([-+0-9.eE]+)",
        "thickness": r"\(Inner\) Membrane Thickness[^:]*:\s*([-+0-9.eE]+)",
        "outer_thickness": r"Outer Membrane Thickness[^:]*:\s*([-+0-9.eE]+)",
    }
    with open(path) as fh:
        text = fh.read()
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = float(match.group(1))
    return info


def get_membranes(info, dummy_z, thickness=None, outer_thickness=None, distance=None):
    """
    The flat membrane boundaries, as a list of dicts (name, mid, lower, upper, source).
    Command line values come first, then the info file, then the dummy beads of the oriented PDB.
    One membrane, or for MemPrO -dm the inner membrane (at -z) and the outer one (at +z).
    """
    source = "MemPrO info file"
    if thickness is not None:
        source = "command line"
    thickness = thickness if thickness is not None else info.get("thickness")
    distance = distance if distance is not None else info.get("distance", 0.0)
    outer_thickness = outer_thickness if outer_thickness is not None else info.get("outer_thickness", thickness)
    if thickness is None:
        levels = sorted(set(np.round(dummy_z, 2)))
        if len(levels) == 4:
            return [{"name": "membrane", "mid": float((levels[0] + levels[3]) / 2), "lower": float(levels[0]),
                     "upper": float(levels[3]), "source": "dummy membrane beads"}]
        if len(levels) == 8:
            return [{"name": name, "mid": float((lv[0] + lv[3]) / 2), "lower": float(lv[0]), "upper": float(lv[3]),
                     "source": "dummy membrane beads"} for name, lv in (("inner", levels[:4]), ("outer", levels[4:]))]
        raise SystemExit("ERROR: No membrane thickness. Give MemPrO's info_rank_N.txt with -i, or the thickness with -mt.")
    if abs(distance) > 1e-5:
        return [
            {"name": "inner", "mid": -distance / 2, "lower": -distance / 2 - thickness / 2,
             "upper": -distance / 2 + thickness / 2, "source": source},
            {"name": "outer", "mid": distance / 2, "lower": distance / 2 - outer_thickness / 2,
             "upper": distance / 2 + outer_thickness / 2, "source": source},
        ]
    return [{"name": "membrane", "mid": 0.0, "lower": -thickness / 2, "upper": thickness / 2, "source": source}]


# ---------------------------------------------------------------------------
# Reading MemPrOD output
# ---------------------------------------------------------------------------

def _npz(path):
    data = np.load(path)
    return np.array(data[data.files[0]])


def load_memprod(folder):
    """
    The membrane surfaces MemPrOD predicted: a list of dicts with the grid (xs, ys, in the frame turned
    by rot about z), the height of the upper and lower surface (top, bot), where each has membrane
    (top_ok, bot_ok) and the z of the membrane's centre in the input's frame (0 unless -dm).
    """
    if os.path.isdir(os.path.join(folder, "LocData")):
        parts = [("membrane", folder)]
    elif os.path.isdir(os.path.join(folder, "Inner", "LocData")) and os.path.isdir(os.path.join(folder, "Outer", "LocData")):
        parts = [("inner", os.path.join(folder, "Inner")), ("outer", os.path.join(folder, "Outer"))]
    else:
        raise SystemExit("ERROR: " + folder + " is not a MemPrOD output folder (no LocData, or Inner/ and Outer/ with LocData)")
    surfaces = []
    for name, part in parts:
        loc = os.path.join(part, "LocData")
        mem = os.path.join(part, "Membrane_Data")
        rot_fn = os.path.join(mem, "Rotation_matrix.npz")
        centre_fn = os.path.join(mem, "Membrane_centre.npz")
        surface = {
            "name": name,
            "xs": _npz(os.path.join(loc, "Xlin.npz")),
            "ys": _npz(os.path.join(loc, "Ylin.npz")),
            "top": _npz(os.path.join(loc, "TopData.npz")),
            "bot": _npz(os.path.join(loc, "BotData.npz")),
            "top_ok": _npz(os.path.join(loc, "TopBan.npz")) > 0.5,
            "bot_ok": _npz(os.path.join(loc, "BotBan.npz")) > 0.5,
            "rot": _npz(rot_fn)[:2, :2] if os.path.exists(rot_fn) else np.eye(2),
            "centre": float(_npz(centre_fn)) if os.path.exists(centre_fn) else 0.0,
        }
        if not os.path.exists(rot_fn):
            warn("No Membrane_Data/Rotation_matrix.npz in " + part + "; taking the MemPrOD grid to be in the protein's x,y axes")
        # Undeformed (bulk) height of each surface: the cells at the edge of the grid
        edge = np.zeros(surface["top"].shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        for key in ("top", "bot"):
            sel = edge & surface[key + "_ok"]
            if not sel.any():
                sel = surface[key + "_ok"]
            surface[key + "_bulk"] = float(np.median(surface[key][sel])) + surface["centre"] if sel.any() else None
        surfaces.append(surface)
    return surfaces


def surface_heights(surface, xy, key):
    """Height of the upper ("top") or lower ("bot") surface nearest to each x,y: inverse distance mean of 5 cells."""
    ok = surface[key + "_ok"]
    if not ok.any():
        return None
    gx, gy = np.meshgrid(surface["xs"], surface["ys"], indexing="ij")
    cells = np.column_stack([gx[ok], gy[ok]])
    zs = surface[key][ok]
    grid_xy = np.dot(xy, surface["rot"].T)
    k = min(5, len(zs))
    dist, idx = cKDTree(cells).query(grid_xy, k=k)
    dist, idx = dist.reshape(len(xy), k), idx.reshape(len(xy), k)
    weights = 1.0 / np.maximum(dist, 0.1)
    return (zs[idx] * weights).sum(axis=1) / weights.sum(axis=1) + surface["centre"]


# ---------------------------------------------------------------------------
# DSSP
# ---------------------------------------------------------------------------

def chain_breaks(residues):
    """True for each residue that starts a chain or follows a gap (C to N over 2.5 A)."""
    breaks = np.ones(len(residues), dtype=bool)
    for i in range(1, len(residues)):
        if residues[i]["chainID"] == residues[i - 1]["chainID"]:
            gap = np.linalg.norm(residues[i]["atoms"]["N"][1] - residues[i - 1]["atoms"]["C"][1])
            breaks[i] = gap > 2.5
    return breaks


def dssp_builtin(residues):
    """
    Secondary structure by the DSSP method (Kabsch & Sander 1983): backbone hydrogen bonds from the
    electrostatic energy, then helices (H, G, I) from n-turns and strands (E, B) from bridges, plus
    turns (T) and bends (S). Returns one code per residue ("" for none).
    """
    n = len(residues)
    N = np.array([r["atoms"]["N"][1] for r in residues])
    CA = np.array([r["atoms"]["CA"][1] for r in residues])
    C = np.array([r["atoms"]["C"][1] for r in residues])
    O = np.array([r["atoms"]["O"][1] if "O" in r["atoms"] else [np.nan] * 3 for r in residues])
    breaks = chain_breaks(residues)
    seg = np.cumsum(breaks)

    # Amide hydrogen, 1 A from N along the C=O of the residue before
    H = np.full((n, 3), np.nan)
    for i in range(1, n):
        if not breaks[i] and residues[i]["resName"] != "PRO":
            co = C[i - 1] - O[i - 1]
            H[i] = N[i] + co / np.linalg.norm(co)

    # Hydrogen bonds: (a, d) where the C=O of a accepts the N-H of d
    pairs = cKDTree(CA).query_pairs(9.0, output_type="ndarray")
    hb = set()
    if len(pairs) > 0:
        acc = np.concatenate([pairs[:, 0], pairs[:, 1]])
        don = np.concatenate([pairs[:, 1], pairs[:, 0]])
        keep = ~((don == acc + 1) & ~breaks[don])
        acc, don = acc[keep], don[keep]
        dist = lambda a, b: np.maximum(np.linalg.norm(a - b, axis=1), 0.5)
        with np.errstate(invalid="ignore"):
            energy = 27.888 * (1 / dist(O[acc], N[don]) + 1 / dist(C[acc], H[don])
                               - 1 / dist(O[acc], H[don]) - 1 / dist(C[acc], N[don]))
            bonded = energy < -0.5
        hb = set(zip(acc[bonded].tolist(), don[bonded].tolist()))

    ss = [""] * n
    turn = {k: np.array([i + k < n and seg[i] == seg[i + k] and (i, i + k) in hb for i in range(n)]) for k in (3, 4, 5)}

    # Alpha helix
    for i in range(1, n - 3):
        if turn[4][i - 1] and turn[4][i]:
            for j in range(i, i + 4):
                ss[j] = "H"

    # Bridges, between residues that each have a neighbour on both sides
    inner = np.zeros(n, dtype=bool)
    inner[1:-1] = (seg[:-2] == seg[2:])
    bridges = {"P": set(), "A": set()}
    for i, j in pairs.tolist():
        if j - i < 3 or not (inner[i] and inner[j]):
            continue
        if ((i - 1, j) in hb and (j, i + 1) in hb) or ((j - 1, i) in hb and (i, j + 1) in hb):
            bridges["P"].add((i, j))
        if ((i, j) in hb and (j, i) in hb) or ((i - 1, j + 1) in hb and (j - 1, i + 1) in hb):
            bridges["A"].add((i, j))
    strand = {}
    for kind, step in (("P", 1), ("A", -1)):
        found = bridges[kind]
        by_i = {}
        for i, j in found:
            by_i.setdefault(i, []).append(j)
            for r in (i, j):
                strand.setdefault(r, "B")
        for i, j in found:
            # A ladder: the next bridge along
            if (i + 1, j + step) in found or (i - 1, j - step) in found:
                strand[i] = strand[j] = "E"
            # A bulge: the next bridge is up to 4 residues on along one strand and 1 along the other
            for i2 in range(i + 1, i + 6):
                for j2 in by_i.get(i2, ()):
                    gap_i, gap_j = i2 - i - 1, (j2 - j) * step - 1
                    if gap_j < 0 or (gap_i == 0 and gap_j == 0):
                        continue
                    if (gap_i <= 4 and gap_j <= 1) or (gap_i <= 1 and gap_j <= 4):
                        for r in list(range(i, i2 + 1)) + list(range(min(j, j2), max(j, j2) + 1)):
                            strand[r] = "E"
    for r, code in strand.items():
        if ss[r] == "":
            ss[r] = code

    # 3-10 helices where nothing else is assigned; pi helices also replace alpha helix (as mkdssp from 2.1 on)
    for k, code, replaces in ((3, "G", ("", "G")), (5, "I", ("", "I", "H"))):
        for i in range(1, n - k + 1):
            if turn[k][i - 1] and turn[k][i] and all(ss[j] in replaces for j in range(i, i + k)):
                for j in range(i, i + k):
                    ss[j] = code
    # Turns and bends
    for k in (3, 4, 5):
        for i in range(n - k):
            if turn[k][i]:
                for j in range(i + 1, i + k):
                    if ss[j] == "":
                        ss[j] = "T"
    for i in range(2, n - 2):
        if ss[i] == "" and seg[i - 2] == seg[i + 2]:
            a, b = CA[i] - CA[i - 2], CA[i + 2] - CA[i]
            cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if cosine < math.cos(math.radians(70.0)):
                ss[i] = "S"
    return ss


def dssp_mkdssp(residues, exe):
    """Secondary structure from the mkdssp program (classic DSSP output), one code per residue."""
    backbone_first = ("N", "CA", "C", "O")
    with tempfile.TemporaryDirectory() as tmp:
        pdb_fn = os.path.join(tmp, "protein.pdb")
        out_fn = os.path.join(tmp, "protein.dssp")
        with open(pdb_fn, "w") as fh:
            # mkdssp 4 does not read a PDB file without a CRYST1 record
            fh.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
            serial = 0
            for i, r in enumerate(residues):
                names = [a for a in backbone_first if a in r["atoms"]] + [a for a in r["atoms"] if a not in backbone_first]
                for name in names:
                    element, xyz = r["atoms"][name]
                    serial += 1
                    padded = name if len(name) == 4 or len(element) == 2 else " " + name
                    fh.write("ATOM  %5d %-4s %3s %1s%4d%1s   %8.3f%8.3f%8.3f  1.00  0.00          %2s\n" % (
                        serial % 100000, padded, r["resName"][:3], r["chainID"], r["resSeq"], r["iCode"] or " ",
                        xyz[0], xyz[1], xyz[2], element))
                if i + 1 == len(residues) or residues[i + 1]["chainID"] != r["chainID"]:
                    fh.write("TER\n")
            fh.write("END\n")
        attempts = ([exe, "--output-format", "dssp", pdb_fn, out_fn], [exe, "-i", pdb_fn, "-o", out_fn])
        # mkdssp 4 needs libcifpp's dictionaries; point it at the ones installed beside it if it is not told where they are
        env = dict(os.environ)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "libcifpp")
        if "LIBCIFPP_DATA_DIR" not in env and os.path.isdir(data_dir):
            env["LIBCIFPP_DATA_DIR"] = data_dir
        errors = []
        for cmd in attempts:
            run = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if run.returncode == 0 and os.path.exists(out_fn) and os.path.getsize(out_fn) > 0:
                break
            errors.append((run.stderr.strip().splitlines() or ["no output"])[-1])
        else:
            raise RuntimeError("mkdssp failed: " + " | ".join(errors))
        codes = {}
        with open(out_fn) as fh:
            started = False
            for line in fh:
                if line.startswith("  #  RESIDUE"):
                    started = True
                    continue
                if started and len(line) > 16 and line[13] != "!":
                    codes[(line[11], int(line[5:10]), line[10].strip())] = line[16].strip()
    known = set(SPECIFIC_SMOOTHING_WINDOWS)
    ss = []
    for r in residues:
        code = codes.get((r["chainID"], r["resSeq"], r["iCode"]), "")
        ss.append(code if code in known else "")
    return ss


def run_dssp(residues, method="auto", exe=None):
    """Returns (codes, description of the method used)."""
    exe = exe or shutil.which("mkdssp") or shutil.which("dssp")
    if method == "mkdssp" and not exe:
        raise SystemExit("ERROR: mkdssp was not found. Give its path with --mkdssp, or use --dssp builtin.")
    if method in ("auto", "mkdssp") and exe:
        try:
            return dssp_mkdssp(residues, exe), "mkdssp (" + exe + ")"
        except (RuntimeError, OSError) as err:
            if method == "mkdssp":
                raise SystemExit("ERROR: " + str(err))
            warn(str(err) + "; using the built-in DSSP")
    return dssp_builtin(residues), "built-in (Kabsch & Sander hydrogen bond method)"


# ---------------------------------------------------------------------------
# Arrow diagram (port of build_ss_topology.py)
# ---------------------------------------------------------------------------

def window_mean(window_width, idx, values):
    """Mean of values over a (fractional) window of residues about idx, narrowed at the ends."""
    max_width = min(idx, values.shape[0] - idx - 1) + 1
    window_width = min(max_width, window_width)
    normed = 1 / window_width
    flank_width = (window_width - 1) / 2
    val = values[idx] * normed
    offset = 0
    while flank_width > 0:
        offset += 1
        cell_val = min(1, flank_width)
        val = val + cell_val * normed * values[idx + offset]
        val = val + cell_val * normed * values[idx - offset]
        flank_width -= cell_val
    return val


def bezier_matrix(n_nodes, vals):
    """Bernstein polynomials: row k gives the weight of each control point at vals[k]."""
    degree = n_nodes - 1
    vals = np.asarray(vals, dtype=float)[:, None]
    i = np.arange(n_nodes)[None, :]
    binomial = np.array([math.comb(degree, k) for k in range(n_nodes)], dtype=float)[None, :]
    return binomial * vals ** i * (1 - vals) ** (degree - i)


def calculate_basis_line(points, vals):
    """Points of the Bezier curve with control points `points` at each of vals (0 to 1)."""
    points = np.asarray(points, dtype=float)
    return np.dot(bezier_matrix(len(points), vals), points)


def fit_basis(points, n_nodes):
    """
    Bezier curve with n_nodes control points through `points`, the first and last control point fixed
    at the first and last point. Returns (mean distance of the points from the curve, control points).
    """
    points = np.asarray(points, dtype=float)
    vals = np.linspace(0, 1, len(points))
    matrix = bezier_matrix(n_nodes, vals)
    nodes = np.array([points[0] * (1 - x) + points[-1] * x for x in np.linspace(0, 1, n_nodes)])
    if n_nodes > 2 and len(points) > 2:
        target = points - np.outer(matrix[:, 0], points[0]) - np.outer(matrix[:, -1], points[-1])
        nodes[1:-1] = np.linalg.lstsq(matrix[:, 1:-1], target, rcond=None)[0]
    err = float(np.mean(np.linalg.norm(np.dot(matrix, nodes) - points, axis=1)))
    return err, nodes


def build_2d(smoothed, ss_type):
    """
    One element in 2D: x is the distance travelled in the membrane plane, y is z. The ends of helices and
    strands are straightened (smoothing pulls them in), then a Bezier curve is fitted.
    """
    window = SPECIFIC_SMOOTHING_WINDOWS[ss_type]
    ec_size, ec_sample_size = SPECIFIC_2D_EDGE_CANCELLATION[SS_SIMPLIFY_MAP[ss_type]]
    n = len(smoothed)
    if n < ec_size * 2 + ec_sample_size:
        third_len = n / 3
        ec_size = int(third_len)
        ec_sample_size = int(third_len) + 1

    element_y = np.array(smoothed[:, 2], dtype=float)
    element_x = np.zeros(n)
    for i in range(1, n):
        element_x[i] = np.linalg.norm(smoothed[i, :2] - smoothed[i - 1, :2]) + element_x[i - 1]

    if ec_sample_size > 1:
        start_window_0 = np.array([element_x[ec_size], element_y[ec_size]])
        start_window_1 = np.array([element_x[ec_size + ec_sample_size - 1], element_y[ec_size + ec_sample_size - 1]])
        start_vector = (start_window_1 - start_window_0) / (ec_sample_size - 1)
        end_window_1 = np.array([element_x[-ec_size - ec_sample_size], element_y[-ec_size - ec_sample_size]])
        end_window_0 = np.array([element_x[-ec_size - 1], element_y[-ec_size - 1]])
        end_vector = (end_window_1 - end_window_0) / (ec_sample_size - 1)
        for x in range(ec_size):
            start_point = start_window_0 - (x + 1) * start_vector
            end_point = end_window_0 - (x + 1) * end_vector
            element_x[ec_size - (x + 1)] = start_point[0]
            element_y[ec_size - (x + 1)] = start_point[1]
            element_x[-ec_size + x] = end_point[0]
            element_y[-ec_size + x] = end_point[1]

    element_x -= element_x.min()
    points = np.column_stack([element_x, element_y])

    # Fewest control points that follow the element to within 1 A on average
    best_err, best_basis = None, None
    for n_points in range(2, min(n + 1, MAX_BASIS_POINTS) + 1):
        err, basis_points = fit_basis(points, n_points)
        if best_err is None or err < best_err:
            best_err, best_basis = err, basis_points
        if err < 1:
            break
    basis_fitted = calculate_basis_line(best_basis, np.linspace(0, 1, n))
    return {
        "basisPoints": [{"x": float(x[0]), "y": float(x[1])} for x in best_basis],
        "basisFitted": [{"x": float(x[0]), "y": float(x[1])} for x in basis_fitted],
        "phase": [idx % (window * 1.0) for idx in range(n)],
    }


def build_chain_segment_ss_plot(positions, alignment, indexes):
    """The elements (runs of one DSSP code; all loop codes count as one) of an unbroken stretch of chain."""
    smoothed_positions = np.array(positions, dtype=float)
    for idx in range(len(positions)):
        window = SPECIFIC_SMOOTHING_WINDOWS[alignment[idx]["DSSP"]]
        smoothed_positions[idx, :] = window_mean(window, idx, positions)

    ss_elements = []
    prev_ss_code = None
    for idx, residue in enumerate(alignment):
        dssp_code = residue["DSSP"]
        if SS_SIMPLIFY_MAP[dssp_code] == SS_LOOP:
            dssp_code = ""
        if dssp_code == DSSP_SS_CODE_STRAND:
            dssp_code = DSSP_SS_CODE_SHEET
        if prev_ss_code != dssp_code:
            ss_elements.append({"DSSP": dssp_code, "idxs": [], "resIdxs": [], "DSSPClass": SS_SIMPLIFY_MAP[dssp_code]})
            if idx > 0:
                # Starts at the last residue of the element before, so that the two join
                ss_elements[-1]["idxs"].append(idx - 1)
        ss_elements[-1]["resIdxs"].append(indexes[idx])
        ss_elements[-1]["idxs"].append(idx)
        prev_ss_code = dssp_code
    for element in ss_elements:
        element["plot"] = build_2d(smoothed_positions[element["idxs"]], element["DSSP"])
        del element["idxs"]
    return ss_elements


def build_chain_ss_plot(positions, alignment, indexes):
    """Splits a chain into segments at gaps (CA atoms over 6 A apart) and builds each."""
    segments = [{"idxs": []}]
    for idx in range(len(positions)):
        if idx > 0 and np.linalg.norm(positions[idx - 1] - positions[idx]) > 6:
            segments.append({"idxs": []})
        segments[-1]["idxs"].append(idx)
    for segment in segments:
        idxs = segment.pop("idxs")
        segment["elements"] = build_chain_segment_ss_plot(
            positions[idxs], [alignment[x] for x in idxs], [indexes[x] for x in idxs])
    return segments


def build_ss_plot(ca_positions, alignment):
    """The arrow diagram of every chain: chains > segments > elements, as the MemProtMD "topology" result."""
    chains = {}
    for idx, aligned_residue in enumerate(alignment):
        key = (aligned_residue["chainID"], aligned_residue["chainInstance"])
        chains.setdefault(key, []).append(idx)
    result = []
    for (chain_id, chain_instance), idxs in sorted(chains.items()):
        result.append({
            "chainID": chain_id,
            "chainInstance": chain_instance,
            "segments": build_chain_ss_plot(ca_positions[idxs], [alignment[x] for x in idxs], idxs),
        })
    return result


# ---------------------------------------------------------------------------
# Geometric stand-in for the simulation contacts
# ---------------------------------------------------------------------------

def sphere_points(n=96):
    k = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * k / n)
    azimuth = math.pi * (1 + 5 ** 0.5) * k
    return np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)])


def accessible_dots(xyz, radii, probe=1.4, n_points=96):
    """Shrake-Rupley: the solvent accessible points of each atom. Returns (points, atom of each, area of each)."""
    sphere = sphere_points(n_points)
    tree = cKDTree(xyz)
    reach = radii + probe
    neighbours = tree.query_ball_point(xyz, reach + reach.max())
    dots, owner, area = [], [], []
    for i in range(len(xyz)):
        pts = xyz[i] + reach[i] * sphere
        nb = np.array([j for j in neighbours[i] if j != i], dtype=int)
        if len(nb) > 0:
            d2 = ((pts[:, None, :] - xyz[nb][None, :, :]) ** 2).sum(axis=2)
            pts = pts[(d2 >= reach[nb][None, :] ** 2).all(axis=1)]
        if len(pts) > 0:
            dots.append(pts)
            owner.append(np.full(len(pts), i))
            area.append(np.full(len(pts), 4 * math.pi * reach[i] ** 2 / n_points))
    if not dots:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0)
    return np.concatenate(dots), np.concatenate(owner), np.concatenate(area)


def lateral_access(xyz, lower, upper, exclusion=4.0, spacing=1.0, layer=2.0, reach=3.0):
    """
    Which space within the membrane slab a lipid could be in. The slab is cut into layers; in each, the
    space further than `exclusion` from every atom is free, and free space connected within the layer to
    the outside of the protein is lipid, the rest is enclosed (a pore or cavity).
    Returns a function giving, for points in the slab, 1 (faces lipid), 2 (faces enclosed space) or
    0 (faces neither: no free space within `reach` in the membrane plane).
    """
    near = xyz[(xyz[:, 2] > lower - exclusion - 1) & (xyz[:, 2] < upper + exclusion + 1)]
    if len(near) == 0:
        return lambda pts: np.ones(len(pts), dtype=int)
    pad = exclusion + 4.0
    x0, y0 = near[:, 0].min() - pad, near[:, 1].min() - pad
    nx = int((near[:, 0].max() + pad - x0) / spacing) + 2
    ny = int((near[:, 1].max() + pad - y0) / spacing) + 2
    zs = np.arange(lower, upper + 1e-6, layer)
    gz, gx, gy = np.meshgrid(zs, x0 + spacing * np.arange(nx), y0 + spacing * np.arange(ny), indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    dist, _ = cKDTree(near).query(cells, k=1, distance_upper_bound=exclusion, workers=-1)
    free = np.isinf(dist).reshape(len(zs), nx, ny)
    in_plane = np.zeros((3, 3, 3), dtype=int)
    in_plane[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labels, _ = ndimage.label(free, structure=in_plane)
    outside = np.unique(np.concatenate([labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                                        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    outside = outside[outside > 0]
    kind = np.where(free, np.where(np.isin(labels, outside), 1, 2), 0)
    # Each cell takes the kind of the nearest free cell of its layer, if that is within reach
    nearest = np.zeros_like(kind)
    for k in range(len(zs)):
        dist2d, idx = ndimage.distance_transform_edt(~free[k], return_indices=True)
        nearest[k] = np.where(dist2d * spacing <= reach, kind[k][idx[0], idx[1]], 0)

    def classify(pts):
        k = np.clip(np.rint((pts[:, 2] - lower) / layer).astype(int), 0, len(zs) - 1)
        ix = np.rint((pts[:, 0] - x0) / spacing).astype(int)
        iy = np.rint((pts[:, 1] - y0) / spacing).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.ones(len(pts), dtype=int)
        out[inside] = nearest[k[inside], ix[inside], iy[inside]]
        return out
    return classify


def geometric_contacts(residues, membranes, head_width=HEAD_WIDTH):
    """
    Per residue, the fraction of its maximum accessible surface that faces each environment:
    "surface" (all of it), "solvent" (outside the membranes, or enclosed space within one: "pore"),
    and per membrane the lipid head groups and tails of the upper and lower leaflet
    ("upper_head", "upper_tail", "lower_head", "lower_tail": arrays of membranes x residues).
    """
    xyz, radii, owner_res = [], [], []
    for i, r in enumerate(residues):
        for element, pos in r["atoms"].values():
            xyz.append(pos)
            radii.append(VDW_RADII.get(element, 1.8))
            owner_res.append(i)
    xyz, radii, owner_res = np.array(xyz), np.array(radii), np.array(owner_res)
    dots, owner, area = accessible_dots(xyz, radii)
    dot_res = owner_res[owner]
    n_res, n_mem = len(residues), len(membranes)
    max_sasa = np.array([MAX_SASA.get(r["resName"], 200.0) for r in residues])

    def per_residue(mask):
        return np.minimum(1.0, np.bincount(dot_res[mask], weights=area[mask], minlength=n_res) / max_sasa)

    result = {"surface": per_residue(np.ones(len(dots), dtype=bool))}
    zones = {key: np.zeros((n_mem, n_res)) for key in ("upper_head", "upper_tail", "lower_head", "lower_tail")}
    in_any = np.zeros(len(dots), dtype=bool)
    pore = np.zeros(len(dots), dtype=bool)
    for m, mem in enumerate(membranes):
        in_slab = (dots[:, 2] >= mem["lower"]) & (dots[:, 2] <= mem["upper"])
        in_any |= in_slab
        kind = np.zeros(len(dots), dtype=int)
        if in_slab.any():
            kind[in_slab] = lateral_access(xyz, mem["lower"], mem["upper"])(dots[in_slab])
        pore |= in_slab & (kind == 2)
        lipid = in_slab & (kind == 1)
        z = dots[:, 2]
        zones["upper_head"][m] = per_residue(lipid & (z > mem["upper"] - head_width))
        zones["upper_tail"][m] = per_residue(lipid & (z >= mem["mid"]) & (z <= mem["upper"] - head_width))
        zones["lower_tail"][m] = per_residue(lipid & (z < mem["mid"]) & (z >= mem["lower"] + head_width))
        zones["lower_head"][m] = per_residue(lipid & (z < mem["lower"] + head_width))
    result.update(zones)
    result["pore"] = per_residue(pore)
    result["solvent"] = per_residue(pore | ~in_any)
    result["head"] = np.minimum(1.0, (zones["upper_head"] + zones["lower_head"]).sum(axis=0))
    result["tail"] = np.minimum(1.0, (zones["upper_tail"] + zones["lower_tail"]).sum(axis=0))
    return result


# ---------------------------------------------------------------------------
# Transmembrane spans, chain placement, histograms (port of postprocess.py)
# ---------------------------------------------------------------------------

def get_bin(boundaries, val):
    if val >= boundaries[-1]:
        return len(boundaries)
    if val < boundaries[0]:
        return 0
    for idx, b in enumerate(boundaries):
        if b >= val:
            return idx


def relative_z_bins(mem):
    """Twelve boundaries: each leaflet in thirds, continued for a leaflet's thickness beyond the membrane."""
    lower = [mem["mid"] + (mem["lower"] - mem["mid"]) * k / 6 for k in (11, 9, 7, 5, 3, 1)]
    upper = [mem["mid"] + (mem["upper"] - mem["mid"]) * k / 6 for k in (1, 3, 5, 7, 9, 11)]
    return lower + upper


def new_span(state, row):
    return {"start_state": state, "start_side": row["side"], "start": (row["resSeq"], row["iCode"]), "alpha": 0, "beta": 0, "none": 0,
            "head": 0, "tail": 0, "solvent": 0, "len": 0, "sequence": ""}


def find_tm_spans(rows):
    """
    Transmembrane spans of one chain. Each row has a "state": -1 below the middle half of the membrane,
    0 within it, +1 above it. A span starts where the chain enters the middle half and ends at the first
    residue beyond it on the other side; one that turns back is not a span. The state machine is the
    reference's, and so are the totals kept for each span, except for stretches that start within the
    middle half (see the end). Each row also has a "side": -1 below the midplane, +1 above it.
    """
    tm_spans = []
    prev_state = None
    prev_position = None
    for row in rows:
        position = row["position"]
        if prev_position is None or np.linalg.norm(position - prev_position) > 5:
            prev_state = None
        prev_position = position

        if len(tm_spans) > 0:
            span = tm_spans[-1]
            span["len"] += 1
            span[row["ss"]] += 1
            span["solvent"] += row["solvent"]
            span["head"] += row["head"]
            span["tail"] += row["tail"]
            span["sequence"] += AA_RESNAMES.get(row["resName"], "X")
        state = row["state"]
        new_state = False
        if prev_state is not None and len(tm_spans) != 0:
            if state != prev_state:
                if state == 0 and prev_state == tm_spans[-1]["start_state"]:
                    tm_spans[-1].update(new_span(tm_spans[-1]["start_state"], row))
                if state != 0 and state != tm_spans[-1]["start_state"]:
                    tm_spans[-1]["end"] = (row["resSeq"], row["iCode"])
                    tm_spans[-1]["end_state"] = state
                    new_state = True
        else:
            new_state = True
        if new_state:
            tm_spans.append(new_span(state, row))
        prev_state = state
    # A stretch that starts within the middle half (at a chain end or after a gap) counts only if it starts in
    # the leaflet opposite the side it leaves by, so that it crosses the midplane (the reference counted them all)
    return [x for x in tm_spans if "end" in x and (x["start_state"] != 0 or x["start_side"] == -x["end_state"])]


def classify_span(span):
    """Secondary structure and contact type of a span, by the reference's rules."""
    alpha = span["alpha"] / span["len"]
    beta = span["beta"] / span["len"]
    diso = span["none"] / span["len"]
    if alpha > beta + diso:
        span_sstype = "alpha"
    elif beta > alpha + diso:
        span_sstype = "beta"
    elif alpha + beta > diso:
        span_sstype = "mixed"
    else:
        span_sstype = "none"
    sol = span["solvent"] / span["len"]
    head = span["head"] / span["len"]
    tail = span["tail"] / span["len"]
    span_contacttype = "protein"
    if sol > head + tail and sol > 0.3:
        span_contacttype = "solvent"
    if head + tail > sol and head + tail > 0.3:
        span_contacttype = "lipid"
    return span_sstype, span_contacttype


PLACEMENTS = ["no membrane", "touching", "single leaflet", "membrane embedded", "membrane spanning"]


def chain_placement(a_heads, b_heads, a_tails, b_tails):
    """The reference's placement of a chain from its contacts with the heads and tails of leaflets a and b."""
    if a_heads > 1 and b_heads > 1:
        return "membrane spanning"
    if a_heads > 1 or b_heads > 1:
        if a_tails > 1 and b_tails > 1:
            return "membrane embedded"
        if a_tails > 1 or b_tails > 1:
            return "single leaflet"
        return "touching"
    return "no membrane"


def z_histogram(zs, sss, boundaries, total):
    """Fraction of `total` residues in each bin of z, in all and by secondary structure."""
    bins = {}
    for z, ss in zip(zs, sss):
        b = get_bin(boundaries, z)
        counts = bins.setdefault(b, {"all": 0, "alpha": 0, "beta": 0, "none": 0})
        counts["all"] += 1
        counts[ss] += 1
    rows = []
    for b in sorted(bins):
        row = {"bin": b,
               "min_val": None if b == 0 else float(boundaries[b - 1]),
               "max_val": None if b == len(boundaries) else float(boundaries[b])}
        row.update({k: v / total for k, v in bins[b].items()})
        rows.append(row)
    return rows


def postprocess(residues, alignment, membranes, contacts, cutoff):
    """Transmembrane spans and placement of each chain, surface residues and z histograms."""
    n = len(residues)
    ca = np.array([r["atoms"]["CA"][1] for r in residues])
    ss = ["alpha" if a["DSSP"] in ("H", "I", "G") and a["DSSP"] != "" else ("beta" if a["DSSP"] in ("B", "E") else "none")
          for a in alignment]
    in_contact = {key: contacts[key] >= cutoff for key in ("solvent", "head", "tail")}
    bins = [relative_z_bins(mem) for mem in membranes]
    rel_z = np.array([[get_bin(b, z) for z in ca[:, 2]] for b in bins])

    chain_ids = []
    for a in alignment:
        if a["chainID"] not in chain_ids:
            chain_ids.append(a["chainID"])
    chain_summaries = {}
    for chain_id in chain_ids:
        idxs = [i for i in range(n) if alignment[i]["chainID"] == chain_id]
        spans, tm_type, placement = [], {}, "no membrane"
        for m, mem in enumerate(membranes):
            rows = [{
                "position": ca[i], "ss": ss[i], "resName": residues[i]["resName"], "resSeq": residues[i]["resSeq"],
                "iCode": residues[i]["iCode"],
                "solvent": float(in_contact["solvent"][i]), "head": float(in_contact["head"][i]),
                "tail": float(in_contact["tail"][i]),
                "state": -1 if rel_z[m][i] <= 4 else (1 if rel_z[m][i] >= 8 else 0),
                "side": 1 if ca[i, 2] > mem["mid"] else -1,
            } for i in idxs]
            for span in find_tm_spans(rows):
                span_sstype, span_contacttype = classify_span(span)
                spans.append({
                    "start": {"resSeq": span["start"][0], "iCode": span["start"][1]},
                    "end": {"resSeq": span["end"][0], "iCode": span["end"][1]},
                    "direction": span["end_state"],
                    "ss": span_sstype,
                    "contact": span_contacttype,
                    "sequence": span["sequence"],
                    "membrane": mem["name"],
                })
                for k in (span_sstype + "-" + span_contacttype, span_sstype, span_contacttype):
                    tm_type[k] = tm_type.get(k, 0) + 1
            # Leaflet a is the upper one, b the lower one
            counts = [int((contacts[key][m][idxs] >= cutoff).sum())
                      for key in ("upper_head", "lower_head", "upper_tail", "lower_tail")]
            this = chain_placement(*counts)
            if PLACEMENTS.index(this) > PLACEMENTS.index(placement):
                placement = this
        chain_summaries[chain_id] = {"spans": [spans], "count": 1, "tm_type": [tm_type], "type": placement}

    z_min, z_max = int(ca[:, 2].min()), int(ca[:, 2].max() + 1)
    z_bins = np.linspace(z_min, z_max, 1 + (z_max - z_min))
    histograms = {}
    for chain_id in chain_ids + ["_all"]:
        idxs = [i for i in range(n) if chain_id == "_all" or alignment[i]["chainID"] == chain_id]
        zs, sss = ca[idxs, 2], [ss[i] for i in idxs]
        histograms[chain_id] = {
            "absolute": z_histogram(zs, sss, z_bins, len(idxs)),
            "relative": {mem["name"]: z_histogram(zs, sss, bins[m], len(idxs)) for m, mem in enumerate(membranes)},
        }
    surface_residues = [int(i) for i in np.nonzero(contacts["surface"] >= cutoff)[0]]
    return chain_summaries, surface_residues, histograms, ss, rel_z


def residue_leaflet_zs(residues, membranes, surfaces):
    """
    Height of the upper and lower membrane surface at each residue: from MemPrOD's surfaces where there
    are any (the surface nearest in x,y, smoothed along each chain), otherwise the flat boundary.
    Returns one {"upper", "lower", "source"} per membrane.
    """
    ca = np.array([r["atoms"]["CA"][1] for r in residues])
    chain_of = np.array([r["chainID"] for r in residues])
    result = []
    for mem in membranes:
        surface = None
        if surfaces:
            surface = min(surfaces, key=lambda s: abs(s["centre"] - mem["mid"]))
            if abs(surface["centre"] - mem["mid"]) > 15:
                warn("No MemPrOD surface near the %s membrane at z = %.1f A; drawing it flat" % (mem["name"], mem["mid"]))
                surface = None
        heights = {"source": "MemPrOD" if surface else "flat (MemPrO boundary)"}
        for side, key in (("upper", "top"), ("lower", "bot")):
            zs = surface_heights(surface, ca[:, :2], key) if surface else None
            if zs is None:
                zs = np.full(len(residues), mem[side])
                if surface:
                    heights["source"] = "MemPrOD, " + side + " flat (no membrane cells)"
            else:
                for chain_id in set(chain_of):
                    sel = chain_of == chain_id
                    zs[sel] = ndimage.gaussian_filter1d(zs[sel], 3.0, mode="nearest")
            heights[side] = [float(z) for z in zs]
        result.append(heights)
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLOURS = {SS_HELIX: "#2a78d6", SS_STRAND: "#008300", SS_LOOP: "#8c8b84"}
INK, INK_SOFT, MEMBRANE_FILL, MEMBRANE_LINE = "#1f1f1d", "#6b6a64", "#f0eee6", "#a3a299"


def layout_chain(chain):
    """
    Places the elements of a chain one after another along x, as one unbroken line: each loop is moved to start
    and end exactly where the helices and strands either side of it are drawn (their ends are straightened, so
    they do not end at the residue's own position). Returns (pieces, residue_xy). Each piece is a dict with
    "kind" (SS_HELIX for a rectangle, SS_STRAND for an arrow, SS_LOOP for a line, "gap" for the join across missing residues),
    "curve" (points) and, for arrows, "first" and "last" (residue indexes). residue_xy maps a residue index
    to its x and y.
    """
    pieces, residue_xy = [], {}
    x0 = 0.0
    previous_end = None
    for segment in chain["segments"]:
        placed = []
        for element in segment["elements"]:
            basis = np.array([[p["x"], p["y"]] for p in element["plot"]["basisPoints"]])
            fitted = np.array([[p["x"], p["y"]] for p in element["plot"]["basisFitted"]])
            curve = calculate_basis_line(basis, np.linspace(0, 1, max(2, 4 * len(fitted))))
            curve[:, 0] += x0
            kind = element["DSSPClass"]
            if len(element["resIdxs"]) < MIN_LENGTHS[kind]:
                kind = SS_LOOP
            placed.append({"kind": kind, "curve": curve, "first": element["resIdxs"][0], "last": element["resIdxs"][-1]})
            own = fitted[len(fitted) - len(element["resIdxs"]):]
            for res_idx, point in zip(element["resIdxs"], own):
                residue_xy[res_idx] = (point[0] + x0, point[1])
            x0 += fitted[-1, 0]
        # Loops take up the difference at each end; a helix and a strand in a row are joined by a short line
        joined = []
        for number, piece in enumerate(placed):
            before = joined[-1] if joined else None
            after = placed[number + 1] if number + 1 < len(placed) else None
            if piece["kind"] == SS_LOOP:
                weight = np.linspace(0, 1, len(piece["curve"]))[:, None]
                if before is not None:
                    piece["curve"] = piece["curve"] + (1 - weight) * (before["curve"][-1] - piece["curve"][0])
                if after is not None and after["kind"] != SS_LOOP:
                    piece["curve"] = piece["curve"] + weight * (after["curve"][0] - piece["curve"][-1])
            elif before is not None and before["kind"] == piece["kind"]:
                # One helix that DSSP gives as several (alpha, then pi or 3-10): one arrow
                before["curve"] = np.concatenate([before["curve"], piece["curve"]])
                before["last"] = piece["last"]
                continue
            elif before is not None and before["kind"] != SS_LOOP:
                joined.append({"kind": SS_LOOP, "curve": np.array([before["curve"][-1], piece["curve"][0]])})
            joined.append(piece)
        if previous_end is not None and joined:
            pieces.append({"kind": "gap", "curve": np.array([previous_end, joined[0]["curve"][0]])})
        pieces.extend(joined)
        if joined:
            previous_end = joined[-1]["curve"][-1]
        x0 += 12.0
    return pieces, residue_xy


def plot_chain(chain, summary, membranes, leaflets, alignment, filename, title_note):
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    pieces, residue_xy = layout_chain(chain)
    order = sorted(residue_xy, key=lambda i: residue_xy[i][0])
    xs = np.array([residue_xy[i][0] for i in order])
    all_y = np.concatenate([p["curve"][:, 1] for p in pieces] + [[m["lower"] for m in membranes], [m["upper"] for m in membranes]])
    y_lo, y_hi = all_y.min() - 10, all_y.max() + 10
    x_hi = max(xs.max(), 1.0)
    height = 5.2
    width = float(np.clip(height * 0.75 * x_hi / (y_hi - y_lo), 6.5, 24.0))
    fig, ax = plt.subplots(figsize=(width + 1.2, height))
    fig.patch.set_facecolor("white")
    # Room on the right for the thickness of each membrane
    margin = 0.04 * x_hi + 6
    x_right = x_hi + margin + 0.9 * (x_hi + 2 * margin) / width

    # Membrane: the surface at each residue along the chain, or flat lines; its thickness at the right
    bent = False
    for mem, heights in zip(membranes, leaflets):
        upper = np.array([heights["upper"][i] for i in order])
        lower = np.array([heights["lower"][i] for i in order])
        bent_here = heights["source"].startswith("MemPrOD")
        bent = bent or bent_here
        ax.fill_between(xs, lower, upper, color=MEMBRANE_FILL, lw=0, zorder=0)
        ax.plot(xs, upper, color=MEMBRANE_LINE, lw=1.2, zorder=1)
        ax.plot(xs, lower, color=MEMBRANE_LINE, lw=1.2, zorder=1)
        if bent_here:
            ax.hlines([mem["lower"], mem["upper"]], xs[0], xs[-1], color=MEMBRANE_LINE, lw=0.8, linestyles=[(0, (2, 3))], zorder=1)
        x_mark = x_hi + 0.5 * margin
        ax.annotate("", (x_mark, mem["lower"]), (x_mark, mem["upper"]), zorder=2,
                    arrowprops={"arrowstyle": "<->", "color": INK_SOFT, "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
        label = "%.0f Å" % (mem["upper"] - mem["lower"])
        if bent_here:
            local = upper - lower
            label += "\n(%.0f–%.0f Å\nat the chain)" % (local.min(), local.max())
        ax.text(x_mark + 0.012 * (x_right + margin), (mem["lower"] + mem["upper"]) / 2, label, ha="left", va="center",
                color=INK, fontsize=8, linespacing=1.3)

    # The chain: loops as thin lines, helices as rectangles, strands as arrows, numbered at each end
    for piece in pieces:
        curve, kind = piece["curve"], piece["kind"]
        if kind == "gap":
            ax.plot(curve[:, 0], curve[:, 1], color=COLOURS[SS_LOOP], lw=1.0, ls=(0, (1, 2.5)), zorder=2)
            continue
        if kind == SS_LOOP:
            ax.plot(curve[:, 0], curve[:, 1], color=COLOURS[SS_LOOP], lw=1.3, zorder=2, solid_capstyle="round", solid_joinstyle="round")
            continue
        if kind == SS_HELIX:
            # A rectangle: a wide band with square ends
            ax.plot(curve[:, 0], curve[:, 1], color=COLOURS[kind], lw=8.0, zorder=3, solid_capstyle="butt", solid_joinstyle="round")
        else:
            # An arrow pointing along the chain
            ax.plot(curve[:-2, 0], curve[:-2, 1], color=COLOURS[kind], lw=4.0, zorder=3, solid_capstyle="butt")
            ax.add_patch(FancyArrowPatch(curve[-4], curve[-1], arrowstyle="-|>,head_length=0.7,head_width=0.42",
                                         mutation_scale=12.8, lw=0, color=COLOURS[kind], shrinkA=0, shrinkB=0, zorder=4))
        # Residue numbers just beyond each end, on the side the element comes from or goes to
        rising = curve[-1, 1] >= curve[0, 1]
        for res_idx, point, below in ((piece["first"], curve[0], rising), (piece["last"], curve[-1], not rising)):
            number = "%d%s" % (alignment[res_idx]["resSeq"], alignment[res_idx]["iCode"])
            ax.annotate(number, point, xytext=(0, -7 if below else 7), textcoords="offset points", ha="center",
                        va="top" if below else "bottom", color=INK, fontsize=6.5, zorder=5,
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75})
    first, last = pieces[0]["curve"][0], pieces[-1]["curve"][-1]
    ax.annotate("N", first, xytext=(-9, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=10, fontweight="bold")
    ax.annotate("C", last, xytext=(9, 0), textcoords="offset points", ha="left", va="center", color=INK, fontsize=10, fontweight="bold")

    ax.set_xlim(-margin, x_right)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("z (Å)", color=INK, fontsize=10)
    ax.set_xlabel("distance along the chain in the membrane plane (Å)", color=INK_SOFT, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MEMBRANE_LINE)
    ax.spines["bottom"].set_bounds(-margin, x_hi + 0.2 * margin)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= x_hi])
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)

    handles = [
        Line2D([], [], color=COLOURS[SS_HELIX], lw=8, solid_capstyle="butt", label="helix"),
        Line2D([], [], color=COLOURS[SS_STRAND], lw=4, label="strand"),
        Line2D([], [], color=COLOURS[SS_LOOP], lw=1.3, label="loop"),
    ]
    if any(p["kind"] == "gap" for p in pieces):
        handles.append(Line2D([], [], color=COLOURS[SS_LOOP], lw=1.0, ls=(0, (1, 2.5)), label="missing residues"))
    handles.append(Line2D([], [], color=MEMBRANE_LINE, lw=1.2, label="membrane surface (MemPrOD)" if bent else "membrane boundary (MemPrO)"))
    if bent:
        handles.append(Line2D([], [], color=MEMBRANE_LINE, lw=0.8, ls=(0, (2, 3)), label="membrane boundary (MemPrO)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8, labelcolor=INK_SOFT,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4)

    tm_type = summary["tm_type"][0]
    counts = []
    for key, name in (("alpha", "TM helix"), ("beta", "TM strand")):
        n_spans = tm_type.get(key, 0)
        plural = name if n_spans == 1 else {"TM helix": "TM helices"}.get(name, name + "s")
        counts.append("%d %s (%d lipid-facing)" % (n_spans, plural, tm_type.get(key + "-lipid", 0)))
    others = tm_type.get("mixed", 0) + tm_type.get("none", 0)
    if others:
        counts.append("%d other span%s" % (others, "" if others == 1 else "s"))
    fig.suptitle("Chain %s: %s" % (chain["chainID"], summary["type"]), x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.90, " · ".join(counts) + "   " + title_note, ha="left", va="center", color=INK_SOFT, fontsize=9)
    fig.subplots_adjust(left=0.75 / (width + 1.2) + 0.01, right=0.995, top=0.84, bottom=0.2)
    fig.savefig(filename, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_topology(pdb_fn, info_fn=None, memprod_dir=None, dssp="auto", mkdssp=None, thickness=None,
                   outer_thickness=None, distance=None, boundaries="mempro", cutoff=0.1):
    """Everything but the files: returns (results for the JSON, per-residue table rows, data for the plots)."""
    residues, dummy_z = read_pdb(pdb_fn)
    info = read_info(info_fn) if info_fn else {}
    membranes = get_membranes(info, dummy_z, thickness, outer_thickness, distance)
    approximations = [
        "Lipid and solvent contacts are geometric (solvent accessible surface by where it faces in the static "
        "structure), not contact frequencies from a simulation. They decide each span's contact type "
        "(lipid/solvent/protein), each chain's placement type and the surface residues.",
        "Membrane boundaries are flat planes; the reference used the mean height of the lipids around the protein.",
    ]
    if abs(info.get("curvature", 0.0)) > 1e-9:
        warn("MemPrO fitted a curved membrane (%g A^-1); flat boundaries at +-thickness/2 are used here" % info["curvature"])
        approximations.append("MemPrO's membrane is curved (%g A^-1) but is treated as flat." % info["curvature"])

    surfaces = load_memprod(memprod_dir) if memprod_dir else None
    if surfaces:
        for mem in membranes:
            surface = min(surfaces, key=lambda s: abs(s["centre"] - mem["mid"]))
            if surface["top_bulk"] is None or surface["bot_bulk"] is None or abs(surface["centre"] - mem["mid"]) > 15:
                continue
            bulk_mid = (surface["top_bulk"] + surface["bot_bulk"]) / 2
            mem["memprod_bulk"] = {"lower": surface["bot_bulk"], "upper": surface["top_bulk"], "mid": bulk_mid}
            if boundaries == "memprod":
                mem.update(lower=surface["bot_bulk"], upper=surface["top_bulk"], mid=bulk_mid, source="MemPrOD bulk membrane")
            elif abs(bulk_mid - mem["mid"]) > 2.0:
                warn("MemPrOD moved the membrane (%s) %.1f A from where MemPrO put it (centre %.1f A against %.1f A). "
                     "Spans are counted with MemPrO's boundaries, the membrane lines are MemPrOD's; "
                     "--boundaries memprod counts them with MemPrOD's bulk membrane."
                     % (mem["name"], bulk_mid - mem["mid"], bulk_mid, mem["mid"]))
    else:
        approximations.append("No MemPrOD surfaces: the membrane lines are flat, at MemPrO's boundaries.")

    codes, dssp_method = run_dssp(residues, dssp, mkdssp)
    alignment = [{"chainID": r["chainID"], "chainInstance": 0, "resSeq": r["resSeq"], "iCode": r["iCode"],
                  "resName": r["resName"], "DSSP": code} for r, code in zip(residues, codes)]
    ca = np.array([r["atoms"]["CA"][1] for r in residues])

    topology = build_ss_plot(ca, alignment)
    contacts = geometric_contacts(residues, membranes)
    chains, surface_residues, histograms, ss, rel_z = postprocess(residues, alignment, membranes, contacts, cutoff)
    leaflets = residue_leaflet_zs(residues, membranes, surfaces)

    membranes_out = []
    for mem, heights in zip(membranes, leaflets):
        entry = dict(mem)
        entry["thickness"] = mem["upper"] - mem["lower"]
        entry["residue_leaflet_zs"] = heights
        membranes_out.append(entry)
    results = {
        "generator": "mempro-topology " + __version__,
        "input": {"structure": os.path.abspath(pdb_fn), "info": os.path.abspath(info_fn) if info_fn else None,
                  "memprod": os.path.abspath(memprod_dir) if memprod_dir else None},
        "dssp": dssp_method,
        "exposure_cutoff": cutoff,
        "approximations": approximations,
        "membranes": membranes_out,
        "alignment": alignment,
        "topology": topology,
        "chains": chains,
        "surface_residues": surface_residues,
        "residue_leaflet_zs": {"upper": leaflets[0]["upper"], "lower": leaflets[0]["lower"]},
        "z_histograms": histograms,
    }
    table = []
    for i, r in enumerate(residues):
        row = {"idx": i, "chainID": r["chainID"], "resSeq": r["resSeq"], "iCode": r["iCode"], "resName": r["resName"],
               "DSSP": codes[i], "ss": ss[i], "x": ca[i, 0], "y": ca[i, 1], "z": ca[i, 2]}
        for m, mem in enumerate(membranes):
            tag = "" if len(membranes) == 1 else "_" + mem["name"]
            row["rel_z" + tag] = int(rel_z[m][i])
            row["leaflet_upper" + tag] = leaflets[m]["upper"][i]
            row["leaflet_lower" + tag] = leaflets[m]["lower"][i]
        for key in ("surface", "head", "tail", "solvent", "pore"):
            row[key] = contacts[key][i]
        table.append(row)
    plot_data = {"membranes": membranes, "leaflets": leaflets, "alignment": alignment}
    return results, table, plot_data


def find_info(pdb_fn):
    found = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(pdb_fn)), "info_rank_*.txt")))
    match = re.search(r"rank_(\d+)", os.path.basename(pdb_fn))
    if match:
        same = [f for f in found if re.search(r"info_rank_%s\.txt$" % match.group(1), f)]
        found = same or found
    return found[0] if len(found) == 1 else None


def write_topology(pdb_fn, out_dir, info_fn=None, memprod_dir=None, chains=None, formats=("png",), plot=True,
                   quiet=False, **options):
    """
    Writes topology.json, residues.csv and a diagram of each chain to out_dir. This is what mempro-topology runs,
    and what MemPrO and MemPrOD call for their own output. The options are those of build_topology.
    Returns False, with the reason printed, if the topology cannot be made (e.g. a coarse-grained structure).
    """
    info_fn = info_fn or find_info(pdb_fn)
    try:
        results, table, plot_data = build_topology(pdb_fn, info_fn, memprod_dir, **options)
    except SystemExit as err:
        print("Topology not written. " + str(err).replace("ERROR: ", ""))
        return False
    if info_fn and not quiet:
        print("Membrane from " + info_fn)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "topology.json"), "w") as fh:
        json.dump(results, fh)
    with open(os.path.join(out_dir, "residues.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader()
        for row in table:
            writer.writerow({k: ("%.3f" % v if isinstance(v, float) else v) for k, v in row.items()})

    if not quiet:
        print("DSSP: " + results["dssp"])
        for mem in results["membranes"]:
            print("%s: z = %.1f to %.1f A (%s); lines: %s" % (
                mem["name"].capitalize(), mem["lower"], mem["upper"], mem["source"], mem["residue_leaflet_zs"]["source"]))
    for chain_id, summary in results["chains"].items():
        tm_type = summary["tm_type"][0]
        print("Chain %s: %s; TM helices: %d (%d lipid-facing); TM strands: %d (%d lipid-facing)" % (
            chain_id, summary["type"], tm_type.get("alpha", 0), tm_type.get("alpha-lipid", 0),
            tm_type.get("beta", 0), tm_type.get("beta-lipid", 0)))
    print("NOTE: lipid-facing spans and chain types are a geometric approximation (static accessibility), not simulation contacts.")

    if plot:
        for chain in results["topology"]:
            if chains is not None and chain["chainID"] not in chains:
                continue
            for fmt in formats:
                plot_chain(chain, results["chains"][chain["chainID"]], plot_data["membranes"], plot_data["leaflets"],
                           plot_data["alignment"],
                           os.path.join(out_dir, "topology_chain_%s.%s" % (chain["chainID"], fmt.strip())),
                           "lipid-facing: geometric estimate")
    print("Topology written to " + out_dir)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mempro-topology",
        description="MemProtMD-style per-chain topology (arrow diagram, TM spans, membrane lines) "
                    "from MemPrO and MemPrOD output.")
    parser.add_argument("-f", "--file_name", required=True, help="Oriented atomistic structure from MemPrO (oriented_rank_N.pdb, with or without the dummy membrane)")
    parser.add_argument("-i", "--info", help="MemPrO's info_rank_N.txt (Default: the one in the folder of -f)")
    parser.add_argument("-d", "--deformations", help="MemPrOD output folder, for the local height of the membrane surfaces (Default: flat membrane lines)")
    parser.add_argument("-o", "--output", default="Topology", help="Output folder (Default: Topology)")
    parser.add_argument("-mt", "--membrane_thickness", type=float, help="Membrane thickness in A, instead of the info file's")
    parser.add_argument("-mt_o", "--outer_membrane_thickness", type=float, help="With -dm, thickness of the membrane at +z in A")
    parser.add_argument("-dm", "--dual_membrane", type=float, help="Distance between the centres of two membranes in A, instead of the info file's")
    parser.add_argument("--boundaries", choices=("mempro", "memprod"), default="mempro",
                        help="Boundaries the spans are counted with: MemPrO's, or the bulk membrane of MemPrOD, which moves the membrane (Default: mempro)")
    parser.add_argument("--dssp", choices=("auto", "mkdssp", "builtin"), default="auto",
                        help="auto: mkdssp if it is installed, otherwise the built-in DSSP (Default: auto)")
    parser.add_argument("--mkdssp", help="Path to mkdssp")
    parser.add_argument("--exposure_cutoff", type=float, default=0.1,
                        help="Relative accessibility from which a residue counts as in contact with lipid or solvent (Default: 0.1)")
    parser.add_argument("--chains", help="Comma separated chains to plot (Default: all)")
    parser.add_argument("--format", default="png", help="Comma separated plot formats, e.g. png,svg (Default: png)")
    parser.add_argument("--no_plot", action="store_true", help="Write the JSON and table only")
    args = parser.parse_args(argv)

    ok = write_topology(
        args.file_name, args.output, args.info, args.deformations, dssp=args.dssp, mkdssp=args.mkdssp,
        thickness=args.membrane_thickness, outer_thickness=args.outer_membrane_thickness, distance=args.dual_membrane,
        boundaries=args.boundaries, cutoff=args.exposure_cutoff, chains=args.chains.split(",") if args.chains else None,
        formats=args.format.split(","), plot=not args.no_plot)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
