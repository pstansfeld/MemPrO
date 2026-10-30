"""
Regression tests for Insane4MemPrO membrane building and mempro-leaflets.

Run from the repository root with:  python -m unittest discover tests
Each test builds a small system in a temporary directory (a few seconds each).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSANE = os.path.join(ROOT, "mempro", "Insane4MemPrO.py")
LEAFLETS = os.path.join(ROOT, "mempro", "leaflet_check.py")
sys.path.insert(0, os.path.join(ROOT, "mempro"))
import leaflet_check  # noqa: E402

IONS = ["-sol", "W", "-negi_c0", "CL", "-posi_c0", "NA"]
TMP_DIRS = []


def tearDownModule():
    for d in TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def build(args, ions=True):
    """Builds a system and returns (log, molecule counts, path of the .gro)."""
    tmp = tempfile.mkdtemp(prefix="insane_test_")
    TMP_DIRS.append(tmp)
    cmd = [sys.executable, INSANE] + args + (IONS if ions else ["-sol", "W"]) + ["-o", "sys.gro", "-p", "topol.top"]
    out = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    log = out.stdout + out.stderr
    if out.returncode != 0:
        raise AssertionError("Insane4MemPrO failed:\n" + log[-3000:])
    counts = {}
    with open(os.path.join(tmp, "topol.top")) as f:
        mols = f.read().split("[ molecules ]")[1]
    for line in mols.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith(";"):
            counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    return log, counts, os.path.join(tmp, "sys.gro")


def leaflet_counts(log):
    """(upper, lower) lipid counts for each membrane, from the build log."""
    return [tuple(map(int, m)) for m in re.findall(r"; (\d+) lipids in upper leaflet, (\d+) lipids in lower leaflet", log)]


def leaflet_areas(log):
    return [tuple(map(float, m)) for m in re.findall(r"Lipid area \(nm\^2\): upper leaflet (\S+), lower leaflet (\S+)", log)]


class TestFlatFilling(unittest.TestCase):
    def test_flat_leaflet_gets_box_area_over_apl(self):
        log, counts, _ = build(["-x", "12", "-y", "10", "-z", "10", "-l", "POPE"])
        self.assertEqual(leaflet_counts(log), [(207, 207)])  # 120 nm^2 / 0.58
        self.assertEqual(counts["POPE"], 414)

    def test_square_grid_same_count(self):
        log, counts, _ = build(["-x", "12", "-y", "10", "-z", "10", "-l", "POPE", "-grid", "square"])
        self.assertEqual(leaflet_counts(log), [(207, 207)])

    def test_explicit_area(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE", "-a", "0.7"])
        self.assertEqual(leaflet_counts(log), [(143, 143)])  # 100 / 0.7

    def test_mixed_leaflet_fills_box_area(self):
        log, counts, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1"])
        for up, lo in leaflet_areas(log):
            self.assertAlmostEqual(up, 225, delta=2)
            self.assertAlmostEqual(lo, 225, delta=2)
        # composition ratio kept, cardiolipin counted per lipid
        self.assertAlmostEqual(counts["POPE"] / counts["CARD"], 7, delta=0.3)
        area = (counts["POPE"] * 0.58 + counts["POPG"] * 0.64 + counts["CARD"] * 1.30) / 2
        self.assertAlmostEqual(area, 225, delta=3)

    def test_apl_override(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE", "-apl", "POPE:0.5"])
        self.assertEqual(leaflet_counts(log), [(200, 200)])

    def test_temperature_and_m3(self):
        log, _, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC", "-temp", "310", "-apl_ff", "m3"])
        self.assertIn("Area per lipid (nm^2): upper 0.612, lower 0.612", log)


class TestDoubleMembraneAndCurvature(unittest.TestCase):
    def test_outer_membrane_lipid_a(self):
        log, counts, _ = build(["-x", "20", "-y", "20", "-z", "40", "-ps", "10", "-l", "POPE:8", "-l", "POPG:1", "-l", "CARD:1",
                                "-uo", "LIPA", "-negi_c1", "CL", "-posi_c1", "NA", "-negi_c2", "CL", "-posi_c2", "NA:1", "-posi_c2", "CA:4"])
        self.assertEqual(counts["LIPA"], 256)  # 400 / 1.56
        for up, lo in leaflet_areas(log):
            self.assertAlmostEqual(up / lo, 1, delta=0.01)

    def test_reversed_curvature_keeps_leaflet_areas(self):
        # Previously each leaflet was built with the other's density when the curvature was reversed
        log, _, _ = build(["-x", "30", "-y", "30", "-z", "25", "-l", "POPE", "-u", "POPG", "-au", "0.9", "-curv", "0.13,0.1,-1"])
        (up, lo), = leaflet_areas(log)
        self.assertAlmostEqual(up / lo, 1, delta=0.05)

    def test_short_return_arc_does_not_crash(self):
        build(["-x", "25", "-y", "25", "-z", "20", "-l", "POPE", "-curv", "0.1,0.3,-1"])


class TestIonsAndBalancing(unittest.TestCase):
    def test_no_ions_warns_instead_of_crashing(self):
        log, counts, _ = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE:8", "-l", "POPG:2"], ions=False)
        self.assertIn("WARNING: Compartment 0 is left with a net charge", log)

    def test_asym_counts_lipids(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1", "-asym", "20"])
        (up, lo), = leaflet_counts(log)
        self.assertEqual(lo - up, 20)

    def test_balance_number(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE", "-u", "LIPA", "-balance", "number",
                           "-posi_c0", "CA"])
        (up, lo), = leaflet_counts(log)
        self.assertEqual(up, lo)

    def test_mismatch_warning(self):
        log, _, _ = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE", "-asym", "30"])
        self.assertIn("WARNING: The leaflet lipid areas differ by more than 5%", log)


class TestLipidAreas(unittest.TestCase):
    def setUp(self):
        self.areas = leaflet_check.load_lipid_areas(INSANE)

    def test_table_and_fallbacks(self):
        get = self.areas["get_apl"]
        self.assertEqual(get("POPE"), 0.58)
        self.assertEqual(get("POPC.o"), 0.64)  # variant takes parent value
        self.assertEqual(get("XXPC"), 0.64)  # PC class
        self.assertEqual(get("DMPC"), 0.60)  # lipidome benchmark (experiment)
        self.assertEqual(get("DLPC"), 0.74)  # lipidome 18:2 PC, from the Martini 3 bilayers
        self.assertEqual(get("DLPC.o"), 0.61)  # Martini 3.0 12:0 PC
        self.assertAlmostEqual(get("DOEC"), 0.67 * 1.015)  # ether, from DOPC
        self.assertAlmostEqual(get("POLE"), 0.58 * 1.05)  # plasmalogen, from POPE
        self.assertEqual(get("NCER"), 0.40)  # ceramide class
        self.assertEqual(get("PSM"), 0.60)
        self.assertEqual(get("POP6"), 0.66)  # PI(4,5)P2
        self.assertEqual(get("XXXX"), 0.60)  # default

    def test_adjustments_skip_lps_and_sterols(self):
        self.areas["adjust_apls"](323.0, "m3")
        get = self.areas["get_apl"]
        self.assertAlmostEqual(get("POPC"), 0.64 * 1.05 * 0.94, places=6)
        self.assertEqual(get("LIPA"), 1.56)
        self.assertEqual(get("CHL1"), 0.40)


def membrane_charge(log):
    return float(re.search(r"; Charge of membrane: (\S+)", log).group(1))


def residue_beads(gro, resname):
    """Bead names of the first residue with the given name in a .gro file."""
    with open(gro) as f:
        lines = f.read().splitlines()[2:-1]
    beads, first = [], None
    for line in lines:
        if line[5:10].strip() == resname:
            if first is None:
                first = line[0:5]
            if line[0:5] != first:
                break
            beads.append(line[10:15].strip())
    return beads


class TestLipidome(unittest.TestCase):
    def test_lipidome_classes_build_with_itp_bead_names(self):
        log, counts, gro = build(["-x", "12", "-y", "12", "-z", "12", "-l", "POPC:4", "-l", "PSM:2", "-l", "CHOL:2", "-l", "DOEC:1",
                                  "-l", "POLE:1", "-u", "NCER:1", "-u", "TOCL:1", "-u", "DO2B:1", "-u", "DOTAP:1", "-u", "POPC:6"])
        for name in ("PSM", "CHOL", "DOEC", "POLE", "NCER", "TOCL", "DO2B", "DOTAP"):
            self.assertGreater(counts.get(name, 0), 0, name)
        # Bead names as in the Martini 3 lipidome itp files
        self.assertEqual(residue_beads(gro, "DOEC"), "NC3 PO4 ET1 ET2 C1A D2A C3A C4A C1B D2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "POLE"), "NH3 PO4 GL1 PL2 C1A D2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "PSM"), "NC3 PO4 OH1 AM2 T1A C2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(residue_beads(gro, "NCER")[:3], ["COH", "OH1", "AM2"])
        self.assertEqual(residue_beads(gro, "TOCL")[:4], ["GLC", "PO41", "GL11", "GL21"])
        expected = -2 * counts["TOCL"] - counts["DO2B"] + counts["DOTAP"]
        self.assertEqual(membrane_charge(log), expected)

    def test_phosphoinositide_charges(self):
        # PIPs were previously missing from the charge table, so they were not neutralised
        log, counts, gro = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC:8", "-l", "POP6:1", "-l", "SAPI:1"])
        self.assertEqual(membrane_charge(log), -4 * counts["POP6"] - counts["SAPI"])
        self.assertEqual(residue_beads(gro, "POP6"), "C1 C2 C3 C4 PO4 P4 P5 GL1 GL2 C1A D2A C3A C4A C1B C2B C3B C4B".split())

    def test_lipid_a_and_cardiolipin_charges(self):
        log, counts, _ = build(["-x", "12", "-y", "12", "-z", "12", "-l", "POPE:6", "-l", "CDL0:1", "-l", "CDL1:1", "-l", "CDL2:1",
                                "-u", "LIPA", "-posi_c0", "CA"])
        self.assertEqual(membrane_charge(log), -4 * counts["LIPA"] - counts["CDL1"] - 2 * counts["CDL2"])

    def test_lipid_itp_definitions_and_warning(self):
        tmp = tempfile.mkdtemp(prefix="insane_itp_")
        TMP_DIRS.append(tmp)
        itp = os.path.join(tmp, "lipids.itp")
        with open(itp, "w") as f:
            # A new lipid defined by its ;@INSANE line, and a POPC whose bead names differ from the template
            f.write(";@INSANE alhead=S P, allink=E E, altail=CCCC CCCC, alname=XXES, charge=-1.0\n"
                    "[ moleculetype ]\nXXES 1\n[ atoms ]\n")
            for i, b in enumerate("CNO PO4 ET1 ET2 C1A C2A C3A C4A C1B C2B C3B C4B".split()):
                f.write("%d C1 1 XXES %s %d %s\n" % (i + 1, b, i + 1, "-1.0" if b == "PO4" else "0"))
            f.write("[ moleculetype ]\nPOPC 1\n[ atoms ]\n1 Q1 1 POPC NC3 1 1.0\n2 Q5 1 POPC PHO 2 -1.0\n")
        log, counts, gro = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPC:1", "-l", "XXES:1", "-lipid_itp", itp])
        self.assertGreater(counts["XXES"], 0)
        self.assertEqual(residue_beads(gro, "XXES"), "CNO PO4 ET1 ET2 C1A C2A C3A C4A C1B C2B C3B C4B".split())
        self.assertEqual(membrane_charge(log), -counts["XXES"])
        self.assertIn("WARNING: The bead names of POPC", log)

    def test_lipid_itp_bead_names_by_position(self):
        """Martini 3 lipidome CARD has no ;@INSANE line and renamed beads; the names come from the itp."""
        tmp = tempfile.mkdtemp(prefix="insane_itp_")
        TMP_DIRS.append(tmp)
        itp = os.path.join(tmp, "card.itp")
        beads = ("GLC PO41 GL11 GL21 C1A1 D2A1 C3A1 C4A1 C1B1 C2B1 C3B1 C4B1 "
                 "PO42 GL12 GL22 C1A2 D2A2 C3A2 C4A2 C1B2 C2B2 C3B2 C4B2").split()
        with open(itp, "w") as f:
            f.write("[ moleculetype ]\nCARD 1\n[ atoms ]\n")
            for i, b in enumerate(beads):
                f.write("%d C1 1 CARD %s %d %s\n" % (i + 1, b, i + 1, "-1.0" if b.startswith("PO") else "0"))
        log, counts, gro = build(["-x", "10", "-y", "10", "-z", "10", "-l", "POPE:9", "-l", "CARD:1", "-lipid_itp", itp])
        self.assertGreater(counts["CARD"], 0)
        self.assertEqual(residue_beads(gro, "CARD"), beads)
        self.assertNotIn("WARNING: The bead names of CARD", log)


class TestProteinCharge(unittest.TestCase):
    """A lipoprotein chain: CYST has a neutral N-terminus, so the C-terminal -1 is not cancelled as residue names assume."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="insane_charge_")
        TMP_DIRS.append(self.dir)
        resnames = ["CYST"] + ["LEU"] * 19
        with open(os.path.join(self.dir, "prot.pdb"), "w") as f:
            for i, r in enumerate(resnames):
                f.write("ATOM  %5d  BB  %-4sA%4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (i + 1, r, i + 1, 50.0, 50.0, -28.5 + 3 * i))
            f.write("END\n")
        with open(os.path.join(self.dir, "chain.itp"), "w") as f:
            f.write("[ moleculetype ]\nchain 1\n\n[ atoms ]\n")
            for i, r in enumerate(resnames):
                f.write("%d P5 %d %s BB %d %s\n" % (i + 1, i + 1, r, i + 1, "-1" if i == len(resnames) - 1 else "0.0"))

    def write_top(self, copies):
        top = os.path.join(self.dir, "prot.top")
        with open(top, "w") as f:
            f.write('#include "martini.itp"\n#include "chain.itp"\n\n[ molecules ]\nchain %d\n' % copies)
        return top

    def ions(self, extra):
        log, counts, _ = build(["-f", os.path.join(self.dir, "prot.pdb"), "-x", "10", "-y", "10", "-z", "10", "-l", "POPE"] + extra)
        return log, counts.get("NA", 0) - counts.get("CL", 0)

    def test_residue_names_miss_terminus(self):
        log, excess = self.ions([])
        self.assertIn("Charge of protein: 0.0", log)
        self.assertEqual(excess, 0)

    def test_charge_from_topology(self):
        log, excess = self.ions(["-p_top", self.write_top(1)])
        self.assertIn("Charge of protein: -1.0", log)
        self.assertEqual(excess, 1)

    def test_mismatched_topology_falls_back(self):
        log, excess = self.ions(["-p_top", self.write_top(2)])
        self.assertIn("do not match the 20 protein beads", log)
        self.assertIn("Charge of protein: 0.0", log)


class TestLeafletCheck(unittest.TestCase):
    def run_check(self, gro):
        out = subprocess.run([sys.executable, LEAFLETS, gro, "-insane", INSANE], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_symmetric_and_asymmetric(self):
        _, _, gro = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1"])
        self.assertIn("no change suggested", self.run_check(gro))
        _, _, gro = build(["-x", "15", "-y", "15", "-z", "10", "-l", "POPE:7", "-l", "POPG:2", "-l", "CARD:1", "-asym", "30"])
        self.assertIn("-asym -30", self.run_check(gro))

    def test_lipid_out_of_the_membrane(self):
        # A lipid pulled out into the water must not become a leaflet (or a membrane) of its own
        _, _, gro = build(["-x", "15", "-y", "15", "-z", "20", "-l", "POPE"])
        with open(gro) as f:
            lines = f.read().splitlines()
        atoms = lines[2:-1]
        resids = sorted({l[0:5] for l in atoms if l[5:10].strip() == "POPE"}, key=int)
        height = {r: max(float(l[36:44]) for l in atoms if l[0:5] == r) for r in resids}
        # The highest lipid 4.6 nm further up (a gap larger than between the leaflets' heads), the lowest 8 nm down
        # (beyond the gap that separates membranes)
        moved = {max(resids, key=height.get): 4.6, min(resids, key=height.get): -8.0}
        for i, l in enumerate(atoms):
            if l[0:5] in moved and l[5:10].strip() == "POPE":
                atoms[i] = "%s%8.3f%s" % (l[:36], float(l[36:44]) + moved[l[0:5]], l[44:])
        with open(gro, "w") as f:
            f.write("\n".join(lines[:2] + atoms + lines[-1:]) + "\n")
        out = self.run_check(gro)
        self.assertNotIn("Membrane 2", out)
        self.assertIn("no change suggested", out)


if __name__ == "__main__":
    unittest.main()
