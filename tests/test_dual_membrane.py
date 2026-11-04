"""
Test of double-membrane orientation (-dm) on MacAB-TolC (tests/data/MacAB-TolC).

Run from the repository root with:  MEMPRO_SLOW_TESTS=1 python -m unittest tests.test_dual_membrane
It runs MemPrO -dm on a 4,582-residue complex (about a minute with -ng 4), so it is skipped unless MEMPRO_SLOW_TESTS
is set. The checks use the structure only (strands or helices in each membrane, where MacB's soluble mass is), not
MemPrO's own sidedness rules.
"""

import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMPRO = os.path.join(ROOT, "mempro", "MemPrO.py")
CG = os.path.join(ROOT, "tests", "data", "MacAB-TolC", "5NIK-cg.pdb.gz")


def rise(xyz, idx):
    """Median distance between backbone beads two residues apart (about 5.5 A in helices, 6.5 A in strands), over
    the runs of beads idx (indices in file order) that are bonded neighbours (steps under 4.5 A)."""
    out = []
    for a, b, c in zip(idx[:-2], idx[1:-1], idx[2:]):
        if b == a + 1 and c == b + 1 and np.linalg.norm(xyz[b] - xyz[a]) < 4.5 and np.linalg.norm(xyz[c] - xyz[b]) < 4.5:
            out.append(np.linalg.norm(xyz[c] - xyz[a]))
    return float(np.median(out)) if out else 0.0


@unittest.skipUnless(os.environ.get("MEMPRO_SLOW_TESTS"), "set MEMPRO_SLOW_TESTS=1 to run (about a minute)")
class TestDualMembrane(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cg = os.path.join(cls.tmp, "5NIK-cg.pdb")
        with gzip.open(CG, "rt") as src, open(cg, "w") as dst:
            dst.write(src.read())
        out = os.path.join(cls.tmp, "Orient") + "/"
        cls.log = subprocess.run([sys.executable, MEMPRO, "-f", cg, "-o", out, "-ng", "4", "-ni", "150", "-dm"],
                                 capture_output=True, text=True, timeout=1800).stdout
        with open(os.path.join(out, "Rank_1", "info_rank_1.txt")) as f:
            cls.info = f.read()
        with open(os.path.join(out, "Rank_1", "oriented_rank_1.pdb")) as f:
            cls.records = [l for l in f if l.startswith("ATOM")]
        lines = [l for l in cls.records if "DUM" not in l and l[12:16].strip() == "BB"]
        cls.xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
        cls.zdist = float(re.search(r"Iter-Membrane distance \(for -dm only\): ([\d.]+)", cls.info).group(1)) / 2

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pdb_columns(self):
        # Protein and dummy-membrane records alike read back from the fixed PDB columns (no tabs, B-factor six wide)
        for line in self.records:
            self.assertNotIn("\t", line)
            for a, b in ((30, 38), (38, 46), (46, 54), (54, 60), (60, 66)):
                float(line[a:b])
        dums = [float(l[46:54]) for l in self.records if l[17:20] == "DUM"]
        self.assertGreater(len(dums), 100)
        # The dummy membranes sit about the membrane centres at +/- half the separation
        self.assertAlmostEqual(max(dums), -min(dums), delta=5)

    def test_long_axis_start(self):
        self.assertIn("Long-axis start added", self.log)

    def test_membrane_separation(self):
        # The periplasm MacA spans: about 272 A between the membrane centres (Insane4MemPrO tutorial 3)
        self.assertAlmostEqual(2 * self.zdist, 272, delta=15)

    def test_barrel_outside_helices_inside(self):
        # The membrane at +z holds TolC's beta-barrel, the one at -z MacB's helices
        z = self.xyz[:, 2]
        upper = np.where(np.abs(z - self.zdist) < 10)[0]
        lower = np.where(np.abs(z + self.zdist) < 10)[0]
        self.assertGreater(len(upper), 30)
        self.assertGreater(len(lower), 30)
        self.assertGreater(rise(self.xyz, upper), 6.0)
        self.assertLess(rise(self.xyz, lower), 6.0)

    def test_cytoplasm_below(self):
        # MacB's nucleotide-binding domains hang below the inner membrane; nothing is beyond the outer membrane but
        # the short extracellular end of TolC
        z = self.xyz[:, 2]
        below = (z < -self.zdist - 20).sum()
        above = (z > self.zdist + 20).sum()
        self.assertGreater(below, 300)
        self.assertLess(above, below / 5)


if __name__ == "__main__":
    unittest.main()
