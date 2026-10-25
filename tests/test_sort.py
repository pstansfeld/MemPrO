"""
Tests for lipid sorting around the protein (-sort model, docs/lipid-sorting-design.md) and the Martini 3 lipid A
template (LPA3).

Run from the repository root with:  python -m unittest discover tests
The model tests call the sort block directly (well under a second); the build tests run Insane4MemPrO on a small
synthetic protein (a few seconds each).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSANE = os.path.join(ROOT, "mempro", "Insane4MemPrO.py")
sys.path.insert(0, os.path.join(ROOT, "mempro"))
import leaflet_check  # noqa: E402

TMP_DIRS = []
NAMES = ("POPE", "POPG", "TOCL")
CHARGES = {"POPE": 0, "POPG": -1, "TOCL": -2}


def load_sort():
    """The sort block, with the area table and lipid charges it reads from the rest of Insane4MemPrO."""
    areas = leaflet_check.load_lipid_areas(INSANE)
    ns = {"np": np, "get_apl": areas["get_apl"], "charges": CHARGES}
    with open(INSANE) as f:
        src = f.read()
    exec(src[src.index("# BEGIN SORT"):src.index("# END SORT")], ns)
    return ns


SORT = load_sort()


def tearDownModule():
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


class TestSortModel(unittest.TestCase):
    def test_screening(self):
        # Martini 3 (epsilon_r 15) at 310 K and 0.15 M
        self.assertAlmostEqual(SORT["sort_bjerrum"](310.0), 3.59, places=2)
        self.assertAlmostEqual(SORT["sort_debye"](0.15, 310.0), 0.35, places=2)

    def test_probabilities_keep_composition(self):
        rng = np.random.default_rng(1)
        E = rng.normal(0, 2, (200, 3))
        target = [140, 40, 20]
        P = SORT["sort_probabilities"](E, target)
        np.testing.assert_allclose(P.sum(axis=1), 1, atol=1e-9)
        np.testing.assert_allclose(P.sum(axis=0), target, atol=1e-4)
        # No energy differences: every site gets the leaflet composition
        P0 = SORT["sort_probabilities"](np.zeros((50, 3)), target)
        np.testing.assert_allclose(P0, np.tile(np.array(target) / 200.0, (50, 1)), atol=1e-9)

    def test_charge_draws_anionic_lipids_to_basic_residues(self):
        qxyz, q = SORT["sort_charges"](["BB", "SC2", "SC1"], ["LYS", "LYS", "GLU"], [[0, 0, 0], [5, 5, 2], [0, 0, 9]])
        np.testing.assert_array_equal(q, [1, -1])
        pts = np.array([[5.5, 5, 2], [9, 9, 2]])
        phi = SORT["sort_potential"](pts, qxyz[:1], q[:1], (10, 10), 3.59, 0.34)
        self.assertGreater(phi[0], 1)  # 0.5 nm from the lysine: ~2.7 kT/e
        self.assertEqual(phi[1], 0)  # beyond the cut-off
        E = SORT["sort_energies"](NAMES, [0, 0], [0, 0], phi, ("charge",))
        P = SORT["sort_probabilities"](E, [1.4, 0.4, 0.2])
        self.assertGreater(P[0, 2], P[1, 2])
        self.assertGreater(P[0, 1], P[1, 1])
        self.assertLess(P[0, 0], P[1, 0])

    def test_curvature_draws_cone_lipids_to_negative_curvature(self):
        E = SORT["sort_energies"](NAMES, [0, 0], [-0.2, 0.2], [0, 0], ("curvature",))
        P = SORT["sort_probabilities"](E, [1.4, 0.4, 0.2])
        self.assertGreater(P[0, 2], P[1, 2])  # cardiolipin, c0 -0.6
        self.assertLess(P[0, 1], P[1, 1])  # PG, c0 0

    def test_mismatch_draws_shallow_lipids_to_thinned_membrane(self):
        self.assertLess(SORT["SORT_DEPTH"]["POPG"], SORT["SORT_DEPTH"]["POPE"])
        E = SORT["sort_energies"](NAMES, [-0.08, 0.0, 0.08], [0] * 3, [0] * 3, ("mismatch",), counts=[70, 20, 10])
        P = SORT["sort_probabilities"](E, [2.1, 0.6, 0.3])
        self.assertGreater(P[0, 1], P[1, 1])
        self.assertGreater(P[1, 1], P[2, 1])
        # Terms not asked for add nothing
        self.assertFalse(np.any(SORT["sort_energies"](NAMES, [0.1], [0.3], [2.0], ())))


def protein_pdb(path):
    """A transmembrane cylinder (radius 1.5 nm) with a ring of lysines at the upper head plane only."""
    lines, i = [], 0
    for z in np.arange(-2.4, 2.5, 0.4):
        for a in np.arange(0, 2 * np.pi, 0.4):
            i += 1
            lines.append(("BB", "LEU", 1.5 * np.cos(a), 1.5 * np.sin(a), z))
    for a in np.arange(0, 2 * np.pi, np.pi / 6):
        i += 1
        lines.append(("SC2", "LYS", 1.9 * np.cos(a), 1.9 * np.sin(a), 2.0))
    with open(path, "w") as f:
        for k, (atom, res, x, y, z) in enumerate(lines):
            f.write("ATOM  %5d  %-3s %-3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (k + 1, atom, res, k + 1, x * 10, y * 10, z * 10))
        f.write("END\n")


def build(args, protein=True):
    tmp = tempfile.mkdtemp(prefix="sort_test_")
    TMP_DIRS.append(tmp)
    if protein:
        protein_pdb(os.path.join(tmp, "prot.pdb"))
        args = ["-f", "prot.pdb"] + args
    cmd = [sys.executable, INSANE] + args + ["-x", "12", "-y", "12", "-z", "12", "-sol", "W", "-o", "sys.gro", "-p", "topol.top"]
    out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    log = out.stdout + out.stderr
    if out.returncode != 0:
        raise AssertionError("Insane4MemPrO failed:\n" + log[-3000:])
    counts = {}
    with open(os.path.join(tmp, "topol.top")) as f:
        for line in f.read().split("[ molecules ]")[1].splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith(";"):
                counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    return log, counts, os.path.join(tmp, "sys.gro")


def near_fractions(log):
    """{leaflet: {type: (fraction within 1 nm of the protein, overall fraction)}} from the -sort model report."""
    out = {}
    for leaf, parts in re.findall(r"-sort model, (\w+) leaflet, \d+ lipids within 1 nm of the protein: (.*)", log):
        out[leaf] = {n: (float(a) / 100, float(b) / 100) for n, a, b in re.findall(r"(\w+) (\d+)% \((\d+)% overall\)", parts)}
    return out


class TestSortBuild(unittest.TestCase):
    MIX = ["-l", "POPE:7", "-l", "POPG:2", "-l", "TOCL:1"]

    def test_sorting_keeps_counts_and_enriches_anionic_lipids(self):
        _, plain, _ = build(self.MIX + ["-sort", "none"])
        log, counts, _ = build(self.MIX + ["-sort", "model"])
        self.assertIn("-sort model: 12 charged side chains", log)
        for name in ("POPE", "POPG", "TOCL"):
            self.assertEqual(counts[name], plain[name])  # sorting moves lipids, never changes how many
        frac = near_fractions(log)
        self.assertEqual(set(frac), {"upper", "lower"})
        # Lysines at the upper head plane only
        up = frac["upper"]
        self.assertGreater(up["POPG"][0] + up["TOCL"][0], up["POPG"][1] + up["TOCL"][1] + 0.15)
        self.assertLess(up["POPE"][0], up["POPE"][1])

    def test_terms_option_is_checked(self):
        tmp = tempfile.mkdtemp(prefix="sort_test_")
        TMP_DIRS.append(tmp)
        protein_pdb(os.path.join(tmp, "prot.pdb"))
        out = subprocess.run([sys.executable, INSANE, "-f", "prot.pdb", "-x", "12", "-y", "12", "-z", "12", "-sol", "W"]
                             + self.MIX + ["-sort", "model", "-sort_terms", "charge,colour", "-o", "sys.gro", "-p", "topol.top"],
                             cwd=tmp, capture_output=True, text=True)
        self.assertIn("ERROR: -sort must be none or model", out.stdout + out.stderr)


class TestLipidA(unittest.TestCase):
    def test_lpa3_template(self):
        log, counts, gro = build(["-u", "LPA3", "-l", "POPE", "-posi_c0", "CA"], protein=False)
        self.assertGreater(counts["LPA3"], 0)
        with open(gro) as f:
            lines = f.read().splitlines()[2:-1]
        first = next(l[:5] for l in lines if l[5:10].strip() == "LPA3")
        beads = [l[10:15].strip() for l in lines if l[:5] == first and l[5:10].strip() == "LPA3"]
        self.assertEqual(beads, ["B%d" % i for i in range(37)])
        self.assertEqual(float(re.search(r"; Charge of membrane: (\S+)", log).group(1)), -4 * counts["LPA3"])
        # Area per lipid A 1.56 nm^2: 144 nm^2 of box
        up = int(re.search(r"; (\d+) lipids in upper leaflet", log).group(1))
        self.assertAlmostEqual(up, 144 / 1.56, delta=2)


if __name__ == "__main__":
    unittest.main()
