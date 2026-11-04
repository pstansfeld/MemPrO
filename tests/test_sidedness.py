"""
Tests for MemPrO -side auto (which way up an outer-membrane protein goes).

Run from the repository root with:  python -m unittest discover tests
The tests call the sidedness block of MemPrO.py directly on synthetic proteins (no JAX, well under a second).
"""

import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMPRO = os.path.join(ROOT, "mempro", "MemPrO.py")


def load_sidedness():
    with open(MEMPRO) as f:
        src = f.read()
    ns = {}
    exec(src[src.index("# BEGIN SIDEDNESS"):src.index("# END SIDEDNESS")], ns)
    return ns


SIDE = load_sidedness()
HALF = 14.0  # half the hydrophobic thickness (A)


def barrel(strands=8, radius=12.0, out_loop=8, in_turn=2, c_tail=4):
    """BB beads of a beta-barrel: strands 3.3 A apart along z (6.6 A two residues apart), joined by loops of out_loop
    residues at +z and turns of in_turn at -z; the chain ends with c_tail residues below the membrane (periplasm), or
    with c_tail=0 halfway along its last strand (missing residues)."""
    pts = []
    for k in range(strands):
        a = 2 * np.pi * k / strands
        x, y = radius * np.cos(a), radius * np.sin(a)
        zs = np.arange(-15, 15.1, 3.3)
        if k % 2:
            zs = zs[::-1]
        pts += [(x, y, z) for z in zs]
        if k == strands - 1:
            if c_tail == 0:
                pts = pts[:-5]
            pts += [(x, y, -18 - 3.5 * j) for j in range(c_tail)]
        else:
            b = 2 * np.pi * (k + 0.5) / strands
            n, z0 = (out_loop, 18.0) if k % 2 == 0 else (in_turn, -17.0)
            # loop beads 3.5 A apart, out of the membrane and back
            pts += [(radius * np.cos(b) * (1 + 0.02 * j), radius * np.sin(b), z0 + np.sign(z0) * 3.5 * min(j, n - 1 - j))
                    for j in range(n)]
    xyz = np.array(pts)
    n = len(xyz)
    return xyz, ["BB"] * n, ["LEU"] * n, list(range(1, n + 1)), ["A"] * n


def turned(protein):
    xyz, *rest = protein
    return (xyz * [1, -1, -1], *rest)


def side(protein):
    return SIDE["membrane_side"](*protein, HALF)


class TestSidedness(unittest.TestCase):

    def test_barrel_c_terminus(self):
        # The C-terminal strand leaves the membrane at -z: the periplasmic side is already down
        s, reasons, warning = side(barrel())
        self.assertEqual(s, 1)
        self.assertIsNone(warning)
        self.assertIn("1 beta-barrel chain end(s) leave the membrane towards -z, 0 towards +z", reasons)
        # Turned over, it is turned back
        self.assertEqual(side(turned(barrel()))[0], -1)

    def test_loops_when_the_c_terminus_is_missing(self):
        # A chain ending in the membrane (missing residues) leaves the decision to the loops: long ones are outside
        self.assertEqual(side(barrel(c_tail=0))[0], 1)
        self.assertEqual(side(turned(barrel(c_tail=0)))[0], -1)
        # Loops and turns of the same length decide nothing
        self.assertEqual(side(barrel(c_tail=0, out_loop=3, in_turn=3))[0], 0)

    def test_chains_merged_without_breaks_in_numbering(self):
        # martinize2 -merge all numbers merged chains straight through; a periplasmic domain far from the barrel's
        # end is a separate segment, and does not count as where the barrel chain leaves the membrane
        xyz, names, resn, num, ch = barrel()
        domain = np.array([[40.0 + 3.5 * j, 0.0, 30.0] for j in range(30)])
        merged = (np.vstack([xyz, domain]), names + ["BB"] * 30, resn + ["ALA"] * 30,
                  num + list(range(num[-1] + 1, num[-1] + 31)), ch + ["A"] * 30)
        self.assertEqual(side(merged)[0], 1)

    def test_positive_inside(self):
        # Seven helices across the membrane (1.5 A rise per residue, 5.5 A two residues apart) joined by loops, with
        # Lys/Arg in the loops at -z: the cytoplasmic side (-l) is already down
        def bundle(k_down=8, k_up=1):
            pts, names = [], []
            for h in range(7):
                cx, cy = 10 * np.cos(h), 10 * np.sin(h)
                t = np.arange(26)
                zs = -19 + 1.5 * t if h % 2 == 0 else 19 - 1.5 * t
                pts += [(cx + 2.3 * np.cos(1.745 * j), cy + 2.3 * np.sin(1.745 * j), z) for j, z in zip(t, zs)]
                names += ["LEU"] * 26
            xyz = np.array(pts)
            names = np.array(names)
            down = np.where(xyz[:, 2] < -15)[0]
            up = np.where(xyz[:, 2] > 15)[0]
            names[down[:k_down]] = "LYS"
            names[up[:k_up]] = "ARG"
            n = len(xyz)
            return xyz, ["BB"] * n, list(names), list(range(1, n + 1)), ["A"] * n
        s, reasons, warning = side(bundle())
        self.assertEqual(s, 1)
        self.assertIn("helical: 8 Lys/Arg within 1 nm of the membrane at -z, 1 at +z", reasons)
        self.assertEqual(side(turned(bundle()))[0], -1)
        # Too few, or too even, to decide (as for KcsA without its cytoplasmic domains)
        self.assertEqual(side(bundle(k_down=3, k_up=1))[0], 0)
        self.assertEqual(side(bundle(k_down=7, k_up=4))[0], 0)
        # Lipid anchors do not decide for a helical protein: in the inner membrane they would be at +z (-u)
        xyz, names, resn, num, ch = bundle(k_down=0, k_up=0)
        anchor = [(np.array([40.0, 3.0 * k, -8.0]), b) for k in range(4) for b in ("BB", "SC1", "C1A", "C2A")]
        anchored = (np.vstack([xyz, [a[0] for a in anchor]]), names + [a[1] for a in anchor],
                    resn + ["CYST"] * len(anchor), num + [1000 + k for k in range(4) for _ in range(4)],
                    ch + ["B"] * len(anchor))
        self.assertEqual(side(anchored)[0], 0)

    def test_lipid_anchors(self):
        # Triacyl Cys tails in the lower leaflet (as PqiC): periplasmic side down; in the upper leaflet: turn over
        def anchors(z, resname="CYST", beads=("C1A", "C2A", "C3A", "C1B", "C2B")):
            rows = [(np.array([3.0 * k, 0.0, z]), bead, resname, k + 1) for k in range(8) for bead in ("BB", "SC1") + beads]
            return (np.array([r[0] for r in rows]), [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                    ["A"] * len(rows))
        self.assertEqual(side(anchors(-8.0))[0], 1)
        self.assertEqual(side(anchors(8.0))[0], -1)
        # A CYS* name in no list is an anchor if it has beads beyond the Cys side chain; plain Cys is not
        self.assertEqual(side(anchors(8.0, "CYSQ"))[0], -1)
        self.assertEqual(side(anchors(8.0, "CYSH", beads=()))[0], 0)

    def test_anchors_and_barrel_disagree(self):
        xyz, names, resn, num, ch = barrel()
        anchor = [(np.array([30.0, 3.0 * k, 8.0]), b) for k in range(3) for b in ("BB", "SC1", "C1A", "C2A")]
        protein = (np.vstack([xyz, [a[0] for a in anchor]]), names + [a[1] for a in anchor],
                   resn + ["CYST"] * len(anchor), num + [1000 + k for k in range(3) for _ in range(4)],
                   ch + ["B"] * len(anchor))
        s, reasons, warning = side(protein)
        self.assertEqual(s, 0)
        self.assertIn("disagree", warning)


if __name__ == "__main__":
    unittest.main()
