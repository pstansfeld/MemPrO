"""
Tests for additional residues (-res / -res_itp).

Run from the repository root with:  python -m unittest discover tests
The test runs MemPrO on the QmcA CG structure and stops while the additional residues are read (a few seconds).
"""

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMPRO = os.path.join(ROOT, "mempro", "MemPrO.py")
QMCA_CG = os.path.join(ROOT, "tests", "data", "QmcA", "QmcA-cg.pdb.gz")


class TestAdditionalResidues(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cg = os.path.join(self.tmp, "QmcA-cg.pdb")
        with gzip.open(QMCA_CG, "rt") as src, open(self.cg, "w") as dst:
            dst.write(src.read())

    def test_residue_missing_from_itp(self):
        """A -res residue with no [ atoms ] entries in -res_itp stops with a clear error and a non-zero exit code."""
        itp = os.path.join(self.tmp, "residues.itp")
        with open(itp, "w") as f:
            f.write("[ moleculetype ]\nPOPE 1\n\n[ atoms ]\n1 Q5 1 POPE NH3 1 1.0\n")

        result = subprocess.run([sys.executable, MEMPRO, "-f", self.cg, "-o", os.path.join(self.tmp, "out"),
                                 "-res", "CYST", "-res_itp", itp],
                                capture_output=True, text=True, timeout=600)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Additional residue CYST has no [ atoms ] entries", result.stdout)
        self.assertNotIn("IndexError", result.stderr)


if __name__ == "__main__":
    unittest.main()
