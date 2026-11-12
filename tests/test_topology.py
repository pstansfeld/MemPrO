"""
Tests for mempro-topology (MemProtMD-style topology from MemPrO output).

Run from the repository root with:  python -m unittest discover tests
The QmcA test puts the atomistic model in MemPrO's frame (a rigid fit of its CA atoms onto the backbone beads of the
oriented coarse-grained model) and takes about 15 s; the others are instant.
"""

import gzip
import json
import os
import tempfile
import unittest

import numpy as np

from mempro import topology

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "QmcA")


def row(z, ss="alpha", state=None):
    return {"position": np.array([0.0, 0.0, z]), "ss": ss, "resName": "LEU", "resSeq": int(z) + 100, "iCode": "",
            "solvent": 0.0, "head": 0.0, "tail": 1.0, "side": 1 if z > 0 else -1, "state": state if state is not None else (-1 if z < -9.5 else (1 if z > 9.5 else 0))}


class TestSpans(unittest.TestCase):
    def test_crossing_is_a_span(self):
        spans = topology.find_tm_spans([row(z) for z in np.arange(-20, 21, 1.5)])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["start_state"], -1)
        self.assertEqual(topology.classify_span(spans[0]), ("alpha", "lipid"))

    def test_turning_back_is_not_a_span(self):
        zs = list(np.arange(-20, 5, 1.5)) + list(np.arange(5, -20, -1.5))
        self.assertEqual(topology.find_tm_spans([row(z) for z in zs]), [])

    def test_chain_starting_in_the_middle(self):
        # Leaves by the leaflet it starts in: not a span. Crosses the midplane first: a span
        self.assertEqual(topology.find_tm_spans([row(z) for z in np.arange(2, 21, 1.5)]), [])
        self.assertEqual(len(topology.find_tm_spans([row(z) for z in np.arange(-8, 21, 1.5)])), 1)

    def test_two_crossings(self):
        zs = list(np.arange(-20, 21, 1.5)) + list(np.arange(20, -21, -1.5))
        spans = topology.find_tm_spans([row(z) for z in zs])
        self.assertEqual([s["start_state"] for s in spans], [-1, 1])

    def test_placement(self):
        self.assertEqual(topology.chain_placement(3, 3, 0, 0), "membrane spanning")
        self.assertEqual(topology.chain_placement(3, 0, 2, 2), "membrane embedded")
        self.assertEqual(topology.chain_placement(3, 0, 2, 0), "single leaflet")
        self.assertEqual(topology.chain_placement(3, 0, 0, 0), "touching")
        self.assertEqual(topology.chain_placement(0, 0, 5, 5), "no membrane")


class TestMembranes(unittest.TestCase):
    def test_info_file(self):
        info = topology.read_info(os.path.join(DATA, "info_rank_1.txt"))
        (mem,) = topology.get_membranes(info, [])
        self.assertEqual((mem["lower"], mem["upper"]), (-19.0, 19.0))

    def test_dual_membrane(self):
        inner, outer = topology.get_membranes({"thickness": 38.0, "outer_thickness": 34.0, "distance": 200.0}, [])
        self.assertEqual((inner["lower"], inner["upper"]), (-119.0, -81.0))
        self.assertEqual((outer["lower"], outer["upper"]), (83.0, 117.0))

    def test_dummy_beads(self):
        (mem,) = topology.get_membranes({}, [-19.0, -14.0, 14.0, 19.0] * 3)
        self.assertEqual((mem["lower"], mem["upper"]), (-19.0, 19.0))


class TestBezier(unittest.TestCase):
    def test_fit_follows_points(self):
        x = np.linspace(0, 30, 25)
        points = np.column_stack([x, 8 * np.sin(x / 8)])
        err2, _ = topology.fit_basis(points, 2)
        err5, nodes = topology.fit_basis(points, 5)
        self.assertLess(err5, 0.5)
        self.assertLess(err5, err2)
        np.testing.assert_allclose(nodes[[0, -1]], points[[0, -1]])


class TestQmcA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(os.path.join(DATA, "QmcA.pdb.gz"), "rt") as fh:
            lines = [l for l in fh if l.startswith("ATOM")]
        with gzip.open(os.path.join(DATA, "protein-cg_rank_1.pdb.gz"), "rt") as fh:
            bb = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in fh if l.startswith("ATOM") and l[12:16].strip() == "BB"])
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
        ca = xyz[[l[12:16].strip() == "CA" for l in lines]]
        assert len(ca) == len(bb)
        u, _, vt = np.linalg.svd(np.dot((ca - ca.mean(0)).T, bb - bb.mean(0)))
        d = np.sign(np.linalg.det(np.dot(u, vt)))
        rot = np.dot(u * [1, 1, d], vt)
        xyz = np.dot(xyz - ca.mean(0), rot) + bb.mean(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pdb = os.path.join(cls.tmp.name, "oriented.pdb")
        with open(cls.pdb, "w") as fh:
            for l, p in zip(lines, xyz):
                fh.write(l[:30] + "%8.3f%8.3f%8.3f" % tuple(p) + l[54:])
        cls.results, cls.table, cls.plot_data = topology.build_topology(
            cls.pdb, os.path.join(DATA, "info_rank_1.txt"), dssp="builtin")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_lipid_facing_tm_helix_per_chain(self):
        self.assertEqual(len(self.results["chains"]), 23)
        for chain_id, summary in self.results["chains"].items():
            self.assertEqual(summary["tm_type"][0].get("alpha"), 1, chain_id)
            self.assertEqual(summary["tm_type"][0].get("alpha-lipid"), 1, chain_id)
            self.assertIn(summary["type"], ("membrane spanning", "membrane embedded"), chain_id)

    def test_mostly_helix_and_strand(self):
        codes = [a["DSSP"] for a in self.results["alignment"]]
        self.assertGreater(codes.count("H") / len(codes), 0.4)
        self.assertGreater(codes.count("E") / len(codes), 0.05)

    def test_topology_covers_every_residue_once(self):
        idxs = [i for chain in self.results["topology"] for seg in chain["segments"] for el in seg["elements"] for i in el["resIdxs"]]
        self.assertEqual(sorted(idxs), list(range(len(self.results["alignment"]))))
        for chain in self.results["topology"]:
            for seg in chain["segments"]:
                for number, el in enumerate(seg["elements"]):
                    self.assertEqual(len(el["plot"]["basisFitted"]), len(el["resIdxs"]) + (1 if number > 0 else 0))

    def test_flat_lines_and_json(self):
        self.assertEqual(set(self.results["residue_leaflet_zs"]["upper"]), {19.0})
        json.dumps(self.results)

    def test_plot(self):
        chain = self.results["topology"][0]
        out = os.path.join(self.tmp.name, "chain.png")
        topology.plot_chain(chain, self.results["chains"][chain["chainID"]], self.plot_data["membranes"],
                            self.plot_data["leaflets"], self.plot_data["alignment"], out, "")
        self.assertGreater(os.path.getsize(out), 10000)


if __name__ == "__main__":
    unittest.main()
