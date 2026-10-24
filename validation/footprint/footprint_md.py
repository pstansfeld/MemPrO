"""
MD validation of Insane4MemPrO -footprint clash against -footprint map (docs/footprint-design.md, validation step 2).

For one protein:
    prepare   martinize2 the atomistic model (Martini 3, elastic network, chains merged into one molecule) and,
              if it is not already oriented, fit the coarse-grained model onto MemPrO's orientation
    build     build the system with each footprint method, with the same box and lipids
    run       energy-minimise, equilibrate (0.5 ns NPT, backbone restrained) and run semi-isotropic NPT
    analyse   lipid density of each leaflet beyond 3 nm from the protein, which does not depend on how the
              footprint is defined. The leaflets are matched when the two densities agree (target: within 1-2%).

Example (see README.md):
    python footprint_md.py prepare runs/QmcA --pdb ../../tests/data/QmcA/QmcA.pdb.gz \\
        --ref ../../tests/data/QmcA/protein-cg_rank_1.pdb.gz --info ../../tests/data/QmcA/info_rank_1.txt
    python footprint_md.py build runs/QmcA --ff /path/to/martini3
    python footprint_md.py run runs/QmcA/clash --ns 5
    python footprint_md.py run runs/QmcA/map --ns 5
    python footprint_md.py analyse runs/QmcA
"""

import argparse
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
MDP_DIR = os.path.join(HERE, "mdp")
METHODS = ("clash", "map")
SOLVENT = {"W", "SOL", "PW", "TW", "SW", "NA", "CL", "CA", "K", "MG", "ION"}
FF_FILES = ("martini_v3.0.0.itp", "martini_v3.0.0_ions_v1.itp", "martini_v3.0.0_solvents_v1.itp",
            "martini_v3.0.0_phospholipids_v1.itp")
# Bonded parameters that the Martini 3 lipidome phospholipid files refer to by name; included when present
FF_BONDED = "martini_v3.0.0_ffbonded_v2.itp"


def leaflet_check():
    """The installed mempro-leaflets module (frame reader, lipid areas and the footprint block)."""
    try:
        from mempro import leaflet_check as lc
    except ImportError:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "mempro"))
        import leaflet_check as lc
    return lc


def insane_path():
    return os.path.join(os.path.dirname(os.path.abspath(leaflet_check().__file__)), "Insane4MemPrO.py")


def run(cmd, cwd=None, stdin=None, log=None):
    """Runs a command, writing its output to log (in cwd), and stops with the end of the output if it fails."""
    print("  $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    out = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    if log:
        with open(os.path.join(cwd or ".", log), "w") as f:
            f.write(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit("Command failed (exit %d):\n%s" % (out.returncode, (out.stdout + out.stderr)[-4000:]))
    return out.stdout + out.stderr


def read_text(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return f.read()


def pdb_atoms(lines):
    return [l for l in lines if l.startswith(("ATOM", "HETATM"))]


def backbone_keys(atoms):
    """
    (chain ID, position among the backbone beads with that ID) of each backbone bead, and its index in atoms.
    Residue numbers are not used, as martinize2 renumbers chains sharing an ID when merging them.
    """
    keys, idx, count = [], [], {}
    for i, l in enumerate(atoms):
        if l[12:16].strip() != "BB":
            continue
        chain = l[21]
        count[chain] = count.get(chain, -1) + 1
        keys.append((chain, count[chain]))
        idx.append(i)
    return keys, idx


def chain_segments(lines):
    """
    Atom records split into chains: at TER records, where the chain ID changes, and where the residue numbering
    starts again (chains that share an ID without a TER between them). martinize2 keeps TER-separated chains apart
    even when they share an ID (e.g. beyond the 62 one-character IDs), but without a TER it groups atoms into residues
    by chain and residue number, so such chains become doubled residues and its graph repair runs for hours.
    """
    chains, current, last = [], None, None
    for line in lines:
        if line.startswith("TER"):
            current = None
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resid = int(line[22:26])
        if current is None or line[21] != current[0][21] or (last is not None and resid < last):
            current = []
            chains.append(current)
        current.append(line)
        last = resid
    return chains


def standard_pdb_line(line):
    """An atom record in standard columns. MemPrO's oriented files have wider B-factors, which push the element into
    the charge columns martinize2 reads."""
    rest = line[54:].split()
    element = rest[-1] if rest and rest[-1].isalpha() and len(rest[-1]) <= 2 else line[12:16].strip().lstrip("0123456789")[:1]
    return "%s%6.2f%6.2f          %2s" % (line[:54], 1.0, 0.0, element)


def pdb_xyz(atoms):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in atoms])


def kabsch(mobile, target):
    """Rotation R and translation t minimising |mobile @ R.T + t - target|."""
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    return rot, tc - mc @ rot.T


def mempro_thickness(info):
    """Phosphate-to-phosphate thickness (nm) from a MemPrO info_rank_N.txt, as MemPrO -bd passes it to Insane."""
    m = re.search(r"\(Inner\) Membrane Thickness: (\S+) A", read_text(info))
    return float(m.group(1)) / 10


# ---------------------------------------------------------------------------------------------------------------- prepare

def cmd_prepare(args):
    os.makedirs(args.dir, exist_ok=True)
    # Protein atoms only (MemPrO's orientation files also hold membrane dummies, DUM)
    skip = ("DUM", "HOH", "WAT", "SOL")
    chains = [[l for l in c if l[17:20].strip() not in skip] for c in chain_segments(read_text(args.pdb).splitlines())]
    chains = [c for c in chains if c]
    with open(os.path.join(args.dir, "atomistic.pdb"), "w") as f:
        for c in chains:
            f.write("\n".join(standard_pdb_line(l) for l in c) + "\nTER\n")
        f.write("END\n")
    print("%d chains (%d atoms), each ended by TER" % (len(chains), sum(len(c) for c in chains)))
    run([args.martinize, "-f", "atomistic.pdb", "-x", "martinized.pdb", "-o", "martinized.top", "-ff", "martini3001",
         "-dssp", *([args.dssp] if args.dssp else []), "-elastic", "-ef", "700", "-el", "0.5", "-eu", "0.9",
         "-p", "backbone", "-pf", "1000", "-merge", "all", "-name", "Protein", "-ignh", "-cys", "auto"],
        cwd=args.dir, log="martinize.log")
    # One molecule (chains merged), renamed Protein to match the topology Insane4MemPrO writes
    mols = re.findall(r"^#include \"(Protein\S*\.itp)\"", read_text(os.path.join(args.dir, "martinized.top")), re.M)
    if len(mols) != 1:
        sys.exit("martinize2 wrote %d molecules (%s); expected one" % (len(mols), ", ".join(mols)))
    itp = read_text(os.path.join(args.dir, mols[0]))
    itp = re.sub(r"(\[ *moleculetype *\][^\n]*\n(?:;[^\n]*\n)*)\s*\S+", r"\1Protein", itp, count=1)
    with open(os.path.join(args.dir, "protein-cg.itp"), "w") as f:
        f.write(itp)
    with open(os.path.join(args.dir, "martinized.pdb")) as f:
        cg = pdb_atoms(f.read().splitlines())
    xyz = pdb_xyz(cg)
    if args.ref:
        # Fit onto MemPrO's (coarse-grained) orientation by the backbone beads, in order
        # Backbone beads are matched by chain ID and their order within it, as martinize2 may write the chains in a
        # different order (sorted by chain ID)
        ref_lines = pdb_atoms(read_text(args.ref).splitlines())
        ref_keys, ref_idx = backbone_keys(ref_lines)
        cg_keys, cg_idx = backbone_keys(cg)
        ref_of = dict(zip(ref_keys, ref_idx))
        matched = [(i, ref_of[k]) for k, i in zip(cg_keys, cg_idx) if k in ref_of]
        if len(matched) != len(ref_keys) or len(matched) != len(cg_keys):
            sys.exit("Matched %d backbone beads of %d (reference) and %d (martinized); they must be the same protein"
                     % (len(matched), len(ref_keys), len(cg_keys)))
        mine, theirs = np.array(matched).T
        ref_xyz = pdb_xyz(ref_lines)
        rot, t = kabsch(xyz[mine], ref_xyz[theirs])
        xyz = xyz @ rot.T + t
        rmsd = np.sqrt(np.mean(np.sum((xyz[mine] - ref_xyz[theirs]) ** 2, axis=1))) / 10
        print("Fitted onto %s: backbone RMSD %.2f nm" % (args.ref, rmsd))
        if rmsd > 0.5:
            sys.exit("The fit is poor (backbone RMSD %.2f nm); the reference may be a different model" % rmsd)
    with open(os.path.join(args.dir, "protein-cg.pdb"), "w") as f:
        for l, (x, y, z) in zip(cg, xyz):
            f.write("%s%8.3f%8.3f%8.3f%s\n" % (l[:30], x, y, z, l[54:]))
        f.write("END\n")
    mt = args.mt if args.mt is not None else (mempro_thickness(args.info) if args.info else None)
    with open(os.path.join(args.dir, "system.json"), "w") as f:
        json.dump({"source": os.path.abspath(args.pdb), "ref": args.ref and os.path.abspath(args.ref), "mt": mt,
                   "beads": len(cg)}, f, indent=1)
    print("Prepared %s: %d beads, membrane thickness %s nm" % (args.dir, len(cg), mt))


# ------------------------------------------------------------------------------------------------------------------ build

def write_index(gro, ndx):
    """PROTEIN, LIPID and SOL_ION groups (as in md.mdp tc-grps) and System."""
    lc = leaflet_check()
    resid, resname, atomname, xyz, box = next(lc.read_gro_frames([gro]))
    prot = np.isin(resname, list(lc.AMINO_ACIDS))
    sol = np.isin(resname, list(SOLVENT))
    groups = {"System": np.ones(len(resname), dtype=bool), "PROTEIN": prot, "LIPID": ~prot & ~sol, "SOL_ION": sol}
    with open(ndx, "w") as f:
        for name, mask in groups.items():
            idx = np.where(mask)[0] + 1
            f.write("[ %s ]\n" % name)
            for k in range(0, len(idx), 15):
                f.write(" ".join(str(i) for i in idx[k:k + 15]) + "\n")
    return {k: int(v.sum()) for k, v in groups.items()}


def cmd_build(args):
    with open(os.path.join(args.dir, "system.json")) as f:
        system = json.load(f)
    missing = [n for n in FF_FILES if not os.path.exists(os.path.join(args.ff, n))]
    if missing:
        sys.exit("Missing from --ff %s: %s" % (args.ff, ", ".join(missing)))
    with open(os.path.join(args.dir, "protein-cg.pdb")) as f:
        xyz = pdb_xyz(pdb_atoms(f.read().splitlines())) / 10
    # Box as MemPrO -bd sizes it: 4 nm beyond the protein in x and y, 5 nm in z
    ext = xyz.max(0) - xyz.min(0)
    box = [ext[0] + 8, ext[1] + 8, ext[2] + 10] if not args.box else [float(v) for v in args.box.split(",")]
    mt = args.mt if args.mt is not None else system.get("mt")
    runs = [(method, method if args.reps == 1 else "%s_r%d" % (method, k + 1))
            for k in range(args.reps) for method in args.methods]
    for method, name in runs:
        out = os.path.join(args.dir, name)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, args.insane, "-f", "../protein-cg.pdb", "-x", "%.4f" % box[0], "-y", "%.4f" % box[1],
               "-z", "%.4f" % box[2], *shlex.split(args.lipids), "-sol", "W", "-negi_c0", "CL", "-posi_c0", "NA",
               "-footprint", method, "-o", "system.gro", "-p", "insane.top"]
        if mt is not None:
            cmd += ["-mt", str(mt)]
        log = run(cmd, cwd=out, log="insane.log")
        for n in FF_FILES:
            shutil.copy(os.path.join(args.ff, n), out)
        shutil.copy(os.path.join(args.dir, "protein-cg.itp"), out)
        top = read_text(os.path.join(out, "insane.top"))
        if os.path.exists(os.path.join(args.ff, FF_BONDED)):
            shutil.copy(os.path.join(args.ff, FF_BONDED), out)
            top = top.replace('#include "martini_v3.0.0.itp"\n', '#include "martini_v3.0.0.itp"\n#include "%s"\n' % FF_BONDED, 1)
        with open(os.path.join(out, "topol.top"), "w") as f:
            f.write(top)
        counts = write_index(os.path.join(out, "system.gro"), os.path.join(out, "index.ndx"))
        leaf = re.search(r"; (\d+) lipids in upper leaflet, (\d+) lipids in lower leaflet", log)
        print("Built %s: box %.2f x %.2f x %.2f nm, %d beads, %s upper / %s lower lipids"
              % (out, box[0], box[1], box[2], counts["System"], leaf.group(1), leaf.group(2)))


def cmd_control(args):
    """Protein-free bilayers with the same lipids, ions and run settings: the reference for analyse --reference."""
    missing = [n for n in FF_FILES if not os.path.exists(os.path.join(args.ff, n))]
    if missing:
        sys.exit("Missing from --ff %s: %s" % (args.ff, ", ".join(missing)))
    box = [float(v) for v in args.box.split(",")]
    os.makedirs(args.dir, exist_ok=True)
    for k in range(args.reps):
        out = os.path.join(args.dir, "control" if args.reps == 1 else "control_r%d" % (k + 1))
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, args.insane, "-x", str(box[0]), "-y", str(box[1]), "-z", str(box[2]),
               *shlex.split(args.lipids), "-sol", "W", "-negi_c0", "CL", "-posi_c0", "NA", "-o", "system.gro", "-p", "insane.top"]
        log = run(cmd, cwd=out, log="insane.log")
        top = read_text(os.path.join(out, "insane.top"))
        for n in FF_FILES:
            shutil.copy(os.path.join(args.ff, n), out)
        if os.path.exists(os.path.join(args.ff, FF_BONDED)):
            shutil.copy(os.path.join(args.ff, FF_BONDED), out)
            top = top.replace('#include "martini_v3.0.0.itp"\n', '#include "martini_v3.0.0.itp"\n#include "%s"\n' % FF_BONDED, 1)
        with open(os.path.join(out, "topol.top"), "w") as f:
            f.write(top)
        counts = write_index(os.path.join(out, "system.gro"), os.path.join(out, "index.ndx"))
        leaf = re.search(r"; (\d+) lipids in upper leaflet, (\d+) lipids in lower leaflet", log)
        print("Built %s: box %.2f x %.2f x %.2f nm, %d beads, %s upper / %s lower lipids"
              % (out, box[0], box[1], box[2], counts["System"], leaf.group(1), leaf.group(2)))


# -------------------------------------------------------------------------------------------------------------------- run

def separate_overlaps(gro_in, gro_out, min_dist=0.05, shift=0.1, seed=1):
    """
    Moves apart beads closer than min_dist (nm), which steepest descent cannot resolve (the force overflows).
    Insane4MemPrO traces each lipid's tails around the protein independently and does not check lipids against
    each other, so neighbouring tails occasionally end up on the same spot. Returns the number of beads moved.
    """
    with open(gro_in) as f:
        lines = f.read().splitlines()
    n = int(lines[1])
    box = np.array(lines[2 + n].split()[:3], dtype=float)
    xyz = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in lines[2:2 + n]])
    rng = np.random.default_rng(seed)
    moved = set()
    for _ in range(10):
        pairs = cKDTree(lc_wrap(xyz, box), boxsize=box).query_pairs(min_dist, output_type="ndarray")
        if len(pairs) == 0:
            break
        for j in np.unique(pairs[:, 1]):
            v = rng.normal(size=3)
            xyz[j] += shift * v / np.linalg.norm(v)
            moved.add(int(j))
    for i in moved:
        l = lines[2 + i]
        lines[2 + i] = "%s%8.3f%8.3f%8.3f%s" % (l[:20], xyz[i, 0], xyz[i, 1], xyz[i, 2], l[44:])
    with open(gro_out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(moved)


def em_max_force(log):
    m = re.findall(r"Maximum force\s*=\s*(\S+)", read_text(log))
    return float(m[-1]) if m else float("nan")


def drop_empty_protein_group(mdp, ndx):
    """
    For systems without protein (controls): removes the PROTEIN temperature-coupling group, and the backbone
    restraint defines, which grompp rejects when nothing in the topology uses them.
    """
    groups = re.findall(r"\[ (\S+) \]\n([^\[]*)", read_text(ndx))
    if any(name == "PROTEIN" and body.split() for name, body in groups):
        return
    text = read_text(mdp)
    text = re.sub(r"(?m)^define\s*=.*$", "; define removed: no protein to restrain", text)
    m = re.search(r"(?m)^tc-grps\s*=\s*(.*)$", text)
    if not m or "PROTEIN" not in m.group(1).split():
        with open(mdp, "w") as f:
            f.write(text)
        return
    names = m.group(1).split(";")[0].split()
    keep = [i for i, n in enumerate(names) if n != "PROTEIN"]
    for key in ("tc-grps", "tau_t", "ref_t"):
        km = re.search(r"(?m)^(%s\s*=\s*)([^;\n]*)(.*)$" % re.escape(key), text)
        values = km.group(2).split()
        text = text[:km.start()] + km.group(1) + " ".join(values[i] for i in keep) + km.group(3) + text[km.end():]
    with open(mdp, "w") as f:
        f.write(text)


def mdp_with_steps(src, dst, nsteps):
    with open(src) as f:
        text = f.read()
    text = re.sub(r"(?m)^nsteps\s*=.*$", "nsteps                   = %d" % nsteps, text)
    with open(dst, "w") as f:
        f.write(text)


def cmd_run(args):
    d = args.dir
    gmx = args.gmx
    mdrun = [gmx, "mdrun", "-ntmpi", "1", *shlex.split(args.mdrun_args)]
    if not os.path.exists(os.path.join(d, "em.gro")):
        moved = separate_overlaps(os.path.join(d, "system.gro"), os.path.join(d, "system_sep.gro"))
        print("Moved %d bead(s) within 0.05 nm of another apart by 0.1 nm before minimisation" % moved)
    stages = [("em", "em.mdp", "system_sep.gro"), ("eq", "eq.mdp", "em.gro"), ("md", "md.mdp", "eq.gro")]
    for name, mdp, start in stages:
        if os.path.exists(os.path.join(d, name + ".gro")):
            print("%s: done (%s.gro exists)" % (name, name))
            continue
        src = os.path.join(MDP_DIR, mdp)
        nsteps = {"em": args.em_steps, "eq": args.eq_steps, "md": int(round(args.ns * 1000 / 0.02))}[name]
        if nsteps:
            mdp_with_steps(src, os.path.join(d, mdp), nsteps)
        else:
            shutil.copy(src, os.path.join(d, mdp))
        drop_empty_protein_group(os.path.join(d, mdp), os.path.join(d, "index.ndx"))
        if os.path.exists(os.path.join(d, name + ".cpt")):
            run(mdrun + ["-deffnm", name, "-cpi", name + ".cpt"], cwd=d, log=name + ".mdrun.log")
            continue
        grompp = [gmx, "grompp", "-f", mdp, "-c", start, "-p", "topol.top", "-n", "index.ndx", "-o", name + ".tpr",
                  "-maxwarn", str(args.maxwarn)]
        if name == "eq":
            grompp += ["-r", start]
        run(grompp, cwd=d, log=name + ".grompp.log")
        run(mdrun + ["-deffnm", name], cwd=d, log=name + ".mdrun.log")
        if name == "em":
            fmax = em_max_force(os.path.join(d, "em.log"))
            if not np.isfinite(fmax):
                os.rename(os.path.join(d, "em.gro"), os.path.join(d, "em_failed.gro"))
                sys.exit("Minimisation failed (maximum force %s); see em.log" % fmax)
            print("em: maximum force %.1f kJ/mol/nm" % fmax)
        print("%s: done" % name)


# ---------------------------------------------------------------------------------------------------------------- analyse

REGIONS = ("bulk", "enclosed")


def analyse_method(d, args, lc, areas, fp):
    """
    Leaflet packing beyond args.cutoff of the protein, per frame. The far region is split into the bulk membrane (its
    largest connected part, periodic) and any regions the protein encloses (e.g. the inside of a ring), which cannot
    exchange lipids with the bulk over a short run and are reported separately.
    """
    frames_gro = os.path.join(d, "frames.gro")
    if not os.path.exists(frames_gro) or os.path.getmtime(frames_gro) < os.path.getmtime(os.path.join(d, "md.xtc")):
        run([args.gmx, "trjconv", "-f", "md.xtc", "-s", "md.tpr", "-b", str(args.skip_ns * 1000), "-o", "frames.gro"],
            cwd=d, stdin="0\n", log="trjconv.log")
    leaflet_of = None
    rows = []
    for resid, resname, atomname, xyz, box in lc.read_gro_frames([frames_gro]):
        prot = np.isin(resname, list(lc.AMINO_ACIDS))
        lip = ~prot & ~np.isin(resname, list(SOLVENT))
        mols = lc.split_molecules(resid, resname, lip)
        names = np.array([resname[m[0]] for m in mols])
        # Each lipid's phosphate (first PO bead, otherwise its first bead), as in mempro-leaflets
        po = np.array([next((i for i in m if atomname[i].startswith("PO")), m[0]) for m in mols])
        pz, pxy = xyz[po, 2], xyz[po, :2]
        if leaflet_of is None:
            # Leaflets from the first frame, split at the largest gap in phosphate height (lipids do not flip)
            order = np.argsort(pz)
            k = lc.leaflet_split(pz[order])
            leaflet_of = np.zeros(len(mols), dtype=bool)
            leaflet_of[order[k + 1:]] = True  # True: upper
        up, lo = leaflet_of, ~leaflet_of
        mid = (pz[up].mean() + pz[lo].mean()) / 2
        # Protein beads in the membrane (cap domains far from the bilayer do not exclude lipids)
        pb = xyz[prot]
        pb = pb[np.abs(pb[:, 2] - mid) < args.slab]
        b = box[:2]
        nx, ny = int(round(b[0] / 0.1)), int(round(b[1] / 0.1))
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) * b[0] / nx, (np.arange(ny) + 0.5) * b[1] / ny, indexing="ij")
        if len(pb):
            dist = cKDTree(lc_wrap(pb[:, :2], b), boxsize=b).query(np.column_stack([gx.ravel(), gy.ravel()]))[0]
            far = (dist > args.cutoff).reshape(nx, ny)
        else:
            far = np.ones((nx, ny), dtype=bool)
        bulk = fp["fp_main_component"](far)
        region_maps = {"bulk": bulk, "enclosed": far & ~bulk}
        row = {"box": float(b[0] * b[1])}
        for region, mask in region_maps.items():
            area = mask.mean() * b[0] * b[1]
            inside = fp["fp_lookup"](mask, pxy, b)
            row[region] = {"area": float(area)}
            for label, sel in (("upper", up), ("lower", lo)):
                s = sel & inside
                row[region][label] = {"n": int(s.sum()), "density": s.sum() / area if area > 0 else float("nan"),
                                      "packing": sum(areas["get_apl"](n) for n in names[s]) / area if area > 0 else float("nan")}
        for label, sel in (("upper", up), ("lower", lo)):
            row[label] = {"n": int(sel.sum()), "thickness": float(abs(pz[sel].mean() - mid))}
        rows.append(row)
    if not rows:
        sys.exit("No frames after %g ns in %s" % (args.skip_ns, d))
    res = {"frames": len(rows)}
    for label in ("upper", "lower"):
        res[label] = {"n": rows[0][label]["n"], "half_thickness": float(np.mean([r[label]["thickness"] for r in rows]))}
    block = max(1, int(round(1.0 / args.frame_ns)))
    for region in REGIONS:
        area = np.array([r[region]["area"] for r in rows])
        if area.mean() < 1.0:
            continue
        reg = {"area": float(area.mean())}
        for label in ("upper", "lower"):
            reg[label] = {"n": float(np.mean([r[region][label]["n"] for r in rows])),
                          "density": float(np.mean([r[region][label]["density"] for r in rows])),
                          "packing": float(np.mean([r[region][label]["packing"] for r in rows]))}
        ratio = np.array([r[region]["upper"]["packing"] / r[region]["lower"]["packing"] - 1 for r in rows])
        reg["diff"] = float(100 * ratio.mean())
        # Frames 0.1 ns apart are correlated: standard error from 1 ns blocks
        blocks = [ratio[i:i + block].mean() for i in range(0, len(ratio) - block + 1, block)]
        reg["diff_se"] = float(100 * np.std(blocks) / np.sqrt(len(blocks))) if len(blocks) > 1 else float("nan")
        res[region] = reg
    # mempro-leaflets on the same frames, for comparison
    out = run([sys.executable, lc.__file__, frames_gro, "-insane", insane_path()], cwd=d, log="leaflets.log")
    m = re.search(r"Differential strain \(upper - lower\): (\S+)%", out)
    res["mempro_leaflets_diff"] = float(m.group(1)) if m else None
    return res


def lc_wrap(xy, box):
    w = np.asarray(xy, dtype=float) % box
    return np.where(w >= box, 0.0, w)


def run_dirs(top):
    """Built systems in a protein directory with an MD run: {method: [(name, path), ...]}, for METHOD and METHOD_rN."""
    found = {}
    for name in sorted(os.listdir(top), key=lambda n: (n.split("_r")[0], int(n.split("_r")[1]) if "_r" in n and n.split("_r")[1].isdigit() else 0)):
        m = re.fullmatch(r"(%s)(?:_r(\d+))?" % "|".join(METHODS + ("control",)), name)
        if m and os.path.exists(os.path.join(top, name, "md.xtc")):
            found.setdefault(m.group(1), []).append((name, os.path.join(top, name)))
    return found


KB_T_310 = 1.380649e-23 * 310.0  # J


def box_area_series(d, args):
    """Box area (nm^2) over the analysed part of the production run, from md.edr."""
    xvg = os.path.join(d, "box.xvg")
    if not os.path.exists(xvg) or os.path.getmtime(xvg) < os.path.getmtime(os.path.join(d, "md.edr")):
        run([args.gmx, "energy", "-f", "md.edr", "-o", "box.xvg", "-b", str(args.skip_ns * 1000)], cwd=d,
            stdin="Box-X\nBox-Y\n\n", log="energy.log")
    rows = [l.split() for l in read_text(xvg).splitlines() if l and l[0] not in "#@"]
    xy = np.array([[float(r[1]), float(r[2])] for r in rows])
    return xy[:, 0] * xy[:, 1]


def reference_state(ref_dir, args, lc, areas, fp):
    """
    Tensionless reference from protein-free controls: the area-weighted packing of a leaflet (lipid area from the area
    table per nm^2 of box) and the bilayer area compressibility K_A = kT <A> / var(A) (mN/m), averaged over controls.
    """
    controls = run_dirs(ref_dir).get("control", [])
    if not controls:
        sys.exit("No control runs (control or control_rN with md.xtc) in %s" % ref_dir)
    packing, ka, diffs = [], [], []
    for name, d in controls:
        r = analyse_method(d, args, lc, areas, fp)
        g = r["bulk"]
        packing.append((g["upper"]["packing"] + g["lower"]["packing"]) / 2)
        diffs.append(g["diff"])
        area = box_area_series(d, args)
        ka.append(KB_T_310 * area.mean() / area.var() * 1e18 * 1e3)
    ref = {"controls": len(controls), "packing": float(np.mean(packing)), "ka": float(np.mean(ka)),
           "ka_sd": float(np.std(ka)), "control_diff": float(np.mean(diffs)),
           "control_diff_se": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")}
    print("Reference (%d protein-free control(s)): packing %.4f, K_A %.0f +/- %.0f mN/m, control up-lo %+.2f%% (se %.2f)"
          % (ref["controls"], ref["packing"], ref["ka"], ref["ka_sd"], ref["control_diff"], ref["control_diff_se"]))
    return ref


def differential_stress(region, ref):
    """
    Estimated differential stress (mN/m, upper minus lower leaflet tension) in a region: each leaflet's strain
    relative to the tensionless reference packing, times the monolayer area compressibility K_A/2. Linear
    elasticity, ignoring curvature; an estimate for comparing builds.
    """
    strain = {leaf: ref["packing"] / region[leaf]["packing"] - 1 for leaf in ("upper", "lower")}
    return strain["upper"], strain["lower"], ref["ka"] / 2 * (strain["upper"] - strain["lower"])


def cmd_analyse(args):
    lc = leaflet_check()
    areas = lc.load_lipid_areas(insane_path())
    fp = lc.load_footprint(insane_path())
    dirs = run_dirs(args.dir)
    if not dirs:
        sys.exit("No md.xtc in %s/{%s}[_rN]" % (args.dir, ",".join(METHODS)))
    results = {}
    for method in METHODS + ("control",):
        for name, d in dirs.get(method, []):
            print("Analysing %s ..." % d, flush=True)
            results[name] = analyse_method(d, args, lc, areas, fp)
    ref = reference_state(args.reference, args, lc, areas, fp) if args.reference else None
    if ref:
        for r in results.values():
            for region in REGIONS:
                if region in r:
                    eu, el, sigma = differential_stress(r[region], ref)
                    r[region].update({"strain_upper": eu, "strain_lower": el, "stress": sigma})
        results["_reference"] = ref
    with open(os.path.join(args.dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    results.pop("_reference", None)
    print("\nLipids beyond %.1f nm of the protein (%s, frames after %g ns)" % (args.cutoff, args.dir, args.skip_ns))
    print("%-9s %-9s %8s %6s %6s %8s %8s %10s %7s %13s%s" % ("run", "region", "area", "n_up", "n_lo", "rho_up", "rho_lo", "up-lo (%)", "se (%)", "leaflets (%)",
                                                        "  strain up/lo (%)  stress (mN/m)" if ref else ""))
    for name, r in results.items():
        for region in REGIONS:
            if region not in r:
                continue
            g = r[region]
            extra = "  %+6.2f / %+6.2f  %+12.2f" % (100 * g["strain_upper"], 100 * g["strain_lower"], g["stress"]) if ref else ""
            print("%-9s %-9s %8.0f %6.0f %6.0f %8.4f %8.4f %+10.2f %7.2f %13s%s" % (
                name, region, g["area"], g["upper"]["n"], g["lower"]["n"], g["upper"]["density"], g["lower"]["density"],
                g["diff"], g["diff_se"], "%+.2f" % r["mempro_leaflets_diff"] if region == "bulk" and r["mempro_leaflets_diff"] is not None else "", extra))
    if any(len(v) > 1 for v in dirs.values()):
        print("\nAcross replicates (mean, and standard error from the spread between replicates):")
        print("%-7s %-9s %4s %10s %7s %13s%s" % ("method", "region", "n", "up-lo (%)", "se (%)", "leaflets (%)", "  stress (mN/m)  se" if ref else ""))
        for method in METHODS + ("control",):
            names = [n for n, _ in dirs.get(method, [])]
            for region in REGIONS:
                vals = [results[n][region]["diff"] for n in names if region in results[n]]
                if not vals:
                    continue
                strain = [results[n]["mempro_leaflets_diff"] for n in names if results[n]["mempro_leaflets_diff"] is not None]
                se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
                extra = ""
                if ref:
                    st = [results[n][region]["stress"] for n in names if region in results[n]]
                    extra = "  %+13.2f %5.2f" % (np.mean(st), np.std(st, ddof=1) / np.sqrt(len(st)) if len(st) > 1 else float("nan"))
                print("%-7s %-9s %4d %+10.2f %7.2f %13s%s" % (method, region, len(vals), np.mean(vals), se,
                                                            "%+.2f" % np.mean(strain) if region == "bulk" and strain else "", extra))
    print("\nbulk: the largest connected region beyond the cut-off; enclosed: regions the protein encloses there (e.g. inside a")
    print("ring), which cannot exchange lipids with the bulk in a short run. area: nm^2; n: lipids in the region; rho: per nm^2.")
    print("up-lo: difference in area-weighted packing between the leaflets (0 = equal packing).")
    if ref:
        print("strain: each leaflet's area per lipid relative to the protein-free reference; stress: estimated differential")
        print("stress, upper minus lower leaflet tension, (K_A/2)(strain_up - strain_lo); 0 = zero differential stress.")
    print("se: per run, from 1 ns blocks (nan with less than 2 ns analysed); across replicates, from their spread.")
    print("leaflets: mempro-leaflets differential strain over the same frames (whole membrane).")


# ------------------------------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare", help="martinize2 and orient the protein")
    q.add_argument("dir", help="Directory for this protein (created)")
    q.add_argument("--pdb", required=True, help="Atomistic model (.pdb or .pdb.gz), oriented by MemPrO unless --ref is given")
    q.add_argument("--ref", help="MemPrO's oriented coarse-grained model of the same protein to fit onto (e.g. protein-cg_rank_1.pdb)")
    q.add_argument("--info", help="MemPrO info_rank_N.txt, for the membrane thickness (-mt)")
    q.add_argument("--mt", type=float, help="Membrane thickness (nm) between phosphate planes, instead of --info")
    q.add_argument("--martinize", default=os.environ.get("MARTINIZE2", "martinize2"), help="martinize2 executable (or $MARTINIZE2)")
    q.add_argument("--dssp", help="DSSP executable for martinize2 (default: martinize2's built-in, via mdtraj)")
    q.set_defaults(func=cmd_prepare)

    b = sub.add_parser("build", help="Build the clash and map systems")
    b.add_argument("dir", help="Prepared protein directory")
    b.add_argument("--ff", default=os.environ.get("MARTINI3_FF"), required=os.environ.get("MARTINI3_FF") is None,
                   help="Directory with the Martini 3 itp files (%s) (or $MARTINI3_FF)" % ", ".join(FF_FILES))
    b.add_argument("--lipids", default="-l POPE:8 -l POPG:2", help="Insane4MemPrO lipid options (Default: '-l POPE:8 -l POPG:2')")
    b.add_argument("--box", help="Box x,y,z (nm) (default: 4 nm beyond the protein in x and y, 5 nm in z, as MemPrO -bd)")
    b.add_argument("--mt", type=float, help="Override the membrane thickness from prepare")
    b.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    b.add_argument("--reps", type=int, default=1, help="Independent builds per method (Insane4MemPrO places lipids at random), "
                   "in METHOD_r1, METHOD_r2, ... (Default: 1, in METHOD)")
    b.add_argument("--insane", default=None, help="Insane4MemPrO.py (default: the installed copy)")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("control", help="Build protein-free bilayers (the reference for analyse --reference)")
    c.add_argument("dir", help="Directory for the controls (created)")
    c.add_argument("--ff", default=os.environ.get("MARTINI3_FF"), required=os.environ.get("MARTINI3_FF") is None,
                   help="Directory with the Martini 3 itp files (or $MARTINI3_FF)")
    c.add_argument("--lipids", default="-l POPE:8 -l POPG:2", help="Insane4MemPrO lipid options (Default: '-l POPE:8 -l POPG:2')")
    c.add_argument("--box", default="15,15,12", help="Box x,y,z (nm) (Default: 15,15,12)")
    c.add_argument("--reps", type=int, default=1, help="Independent builds (Default: 1)")
    c.add_argument("--insane", default=None, help="Insane4MemPrO.py (default: the installed copy)")
    c.set_defaults(func=cmd_control)

    r = sub.add_parser("run", help="Minimise, equilibrate and run NPT for one built system")
    r.add_argument("dir", help="Built system directory (e.g. runs/QmcA/map)")
    r.add_argument("--ns", type=float, default=5.0, help="Production length (ns) (Default: 5)")
    r.add_argument("--gmx", default=os.environ.get("GMX", "gmx"), help="GROMACS executable (or $GMX)")
    r.add_argument("--mdrun-args", default="", help="Extra gmx mdrun options, e.g. '-nb gpu -ntomp 8'")
    r.add_argument("--maxwarn", type=int, default=0, help="grompp -maxwarn (Default: 0)")
    # Shorter minimisation and equilibration, for testing the pipeline
    r.add_argument("--em-steps", type=int, default=None, help=argparse.SUPPRESS)
    r.add_argument("--eq-steps", type=int, default=None, help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyse", help="Compare leaflet densities beyond the cut-off for both methods")
    a.add_argument("dir", help="Protein directory holding clash/ and map/")
    a.add_argument("--skip-ns", type=float, default=1.0, help="Production time to skip as equilibration (ns) (Default: 1)")
    a.add_argument("--cutoff", type=float, default=3.0, help="Distance from the protein (nm) beyond which lipids are counted (Default: 3)")
    a.add_argument("--slab", type=float, default=2.5, help="Protein beads within this distance of the midplane (nm) count as in the membrane (Default: 2.5)")
    a.add_argument("--frame-ns", type=float, default=0.1, help="Time between frames (ns), for the error estimate (Default: 0.1)")
    a.add_argument("--gmx", default=os.environ.get("GMX", "gmx"), help="GROMACS executable (or $GMX)")
    a.add_argument("--reference", help="Directory with protein-free control runs (from control): adds each leaflet's strain and "
                   "the estimated differential stress, and the control's own up-lo as a baseline")
    a.set_defaults(func=cmd_analyse)

    args = p.parse_args(argv)
    if getattr(args, "insane", "unset") is None:
        args.insane = insane_path()
    args.func(args)


if __name__ == "__main__":
    main()
